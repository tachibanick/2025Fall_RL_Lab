import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from env.gridworld_env import Action, TileType
from env.gridworld_mdp import GridWorldMDP
import random

ARROWS = {Action.UP: "↑", Action.DOWN: "↓", Action.LEFT: "←", Action.RIGHT: "→"}
NO_OF_ACTIONS = Action.__len__()
EPSILON = 0.2


# plot value and policy per every iteration
def plot_value_and_policy(
    V, policy, grid, iteration, width, height, policy_name, prefix="vi"
):
    # dir
    save_dir = os.path.join("outputs", f"{prefix}_{policy_name}")
    os.makedirs(save_dir, exist_ok=True)

    # plotting
    value_grid = np.full((height, width), np.nan)
    policy_grid = np.full((height, width), "", dtype=object)

    first_val = next(iter(policy.values()))
    if isinstance(first_val, dict):
        # stochastic policy (dict of Action to probaility for that action)
        deterministic_policy = {
            s: max(a_probs, key=a_probs.get) for s, a_probs in policy.items()
        }
    else:
        # deterministic policy
        deterministic_policy = policy
    policy = deterministic_policy
    for (y, x), v in V.items():
        value_grid[y][x] = v
        policy_grid[y][x] = ARROWS[policy[(y, x)]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    im = axes[0].imshow(value_grid, cmap="coolwarm", interpolation="nearest")
    for y in range(height):
        for x in range(width):
            if not np.isnan(value_grid[y, x]):
                axes[0].text(
                    x,
                    y,
                    f"{value_grid[y, x]:.1f}",
                    ha="center",
                    va="center",
                    color="black",
                )
    axes[0].set_title(f"Value Function - Iteration {iteration}")

    axes[1].imshow(np.ones_like(value_grid), cmap="gray", vmin=0, vmax=1)
    for y in range(height):
        for x in range(width):
            if policy_grid[y][x]:
                axes[1].text(
                    x, y, policy_grid[y][x], ha="center", va="center", fontsize=16
                )
            if grid[y][x] == TileType.WALL:
                axes[1].add_patch(
                    plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="black")
                )
            elif grid[y][x] == TileType.TRAP:
                axes[1].add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="red"))
            elif grid[y][x] == TileType.GOAL:
                axes[1].add_patch(
                    plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="green")
                )
    axes[1].set_title(f"Policy - Iteration {iteration}")

    for ax in axes:
        ax.set_xticks(np.arange(width))
        ax.set_yticks(np.arange(height))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(False)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/iteration_{iteration}.png")
    plt.close()


# -----Policy Iteration-----#
# Takes policy, returns map of state -> value
def policy_evaluation(policy, mdp, gamma=0.95, theta=1e-3):
    V = {s: 0 for s in mdp.states}  # initial value as zero
    while True:
        delta = 0
        for s in mdp.states:
            V_new = 0
            for a, p_a in policy[s].items():  # action with current state and policy
                for p_env, next_s, reward, done in mdp.get_transition_probabilities(
                    s, a
                ):
                    V_new += (
                        p_env * p_a * (reward + gamma * V.get(next_s, 0) * (not done))
                    )  # Bellman Equation
                    # p(s,a) = p_a * p. very confusing.

            delta = max(delta, abs(V[s] - V_new))
            V[s] = V_new

        if delta < theta:
            break
    print(
        f"[Policy Evaluation] s: {s}, a: {a}, p: {p_env*p_a}, next_s: {next_s}, reward: {reward}, done: {done}, V_new: {V_new}"
    )
    print(f"Value Function: {V}")
    return V


def policy_improvement(V, mdp, gamma=0.95):
    # takes value function, returns updated policy
    policy_stable = True
    policy = {}

    for s in mdp.states:
        old_action = policy.get(s)
        action_values = {}

        # compute q(s,a) for all actions possible for the current state
        for a in mdp.actions:  # don't need p_a here. also wouldn't make sense
            q = 0
            for p, next_s, reward, done in mdp.get_transition_probabilities(s, a):
                q += p * (
                    reward + gamma * V.get(next_s, 0) * (not done)
                )  # Bellman Equation
            action_values[a] = q

        best_action = max(action_values, key=action_values.get)  # search max q

        policy[s] = {
            a: EPSILON / len(mdp.actions) for a in mdp.actions
        }  # everything, at least epsilon probability

        policy[s][best_action] += 1.0 - EPSILON  # make best action more probable

        # if updated, not converged
        if old_action is not None and old_action != best_action:
            policy_stable = False

    return policy, policy_stable


def policy_eval_get_Q(policy, mdp, gamma=0.95, theta=1e-3):
    # policy, mdp -> Q[(s,a)] = expected value of taking action a in state s under policy.. policy
    Q = {(s, a): 0 for s in mdp.states for a in mdp.actions}
    while True:
        delta = 0
        new_Q = Q.copy()
        for s in mdp.states:
            for a in mdp.actions:
                q = 0
                for p, s_next, r, done in mdp.get_transition_probabilities(s, a):
                    exp_next = 0
                    for a_next, pi_ap in policy[s_next].items():
                        exp_next += (
                            pi_ap * Q[(s_next, a_next)]
                        )  # expected value of next state
                    q += p * (r + gamma * exp_next * (not done))  # bellman equation
                new_Q[(s, a)] = q
                delta = max(delta, abs(Q[(s, a)] - q))
        Q = new_Q
        if delta < theta:
            break
    return Q


def policy_improvement_from_Q(Q, mdp):
    nA = len(mdp.actions)
    new_policy = {}
    for s in mdp.states:
        q_values = {a: Q[(s, a)] for a in mdp.actions}  # all q values of this state
        best_a = max(q_values, key=q_values.get)
        new_policy[s] = {a: EPSILON / nA for a in mdp.actions}
        new_policy[s][best_a] += 1 - EPSILON
    return new_policy


def policy_iteration_from_Q(
    mdp, policy_name, gamma=0.95, theta=1e-6, max_iterations=80
):
    nA = len(mdp.actions)
    policy = {s: {a: 1 / nA for a in mdp.actions} for s in mdp.states}

    for i in range(max_iterations):
        Q = policy_eval_get_Q(policy, mdp, gamma, theta)  # get expected values
        old_policy = policy.copy()
        policy = policy_improvement_from_Q(
            Q, mdp
        )  # from expected values, get new policy

        # PLOTTING
        plot_value_and_policy(
            {s: max(Q[(s, a)] for a in mdp.actions) for s in mdp.states},
            policy,
            mdp.env.grid,
            i,
            mdp.width,
            mdp.height,
            policy_name,
            prefix="pi_q",
        )

        if all(
            abs(policy[s][a] - old_policy[s][a]) < theta  # arbitrary small number
            for s in mdp.states
            for a in mdp.actions
        ):
            break
    return Q, policy


def value_iteration_q(mdp, policy_name, gamma=0.95, theta=1e-3, max_iterations=80):
    Q = {(s, a): 0 for s in mdp.states for a in mdp.actions}

    for iteration in range(max_iterations):
        delta = 0
        new_Q = Q.copy()
        for s in mdp.states:
            for a in mdp.actions:
                q = 0
                for p, s_next, r, done in mdp.get_transition_probabilities(s, a):
                    max_next = max(
                        Q[(s_next, a_next)] for a_next in mdp.actions
                    )  # "recursive". assumee best action taken
                    q += p * (r + gamma * max_next * (not done))  # bellman equation
                new_Q[(s, a)] = q  # becomes your new best estimate of Q
                delta = max(
                    delta, abs(Q[(s, a)] - q)
                )  # how much Q changed. see value_iteration haha

        Q = new_Q

        # for plotting
        V = {s: max(Q[(s, a)] for a in mdp.actions) for s in mdp.states}
        best_policy = {}
        for s in mdp.states:
            best_a = max(mdp.actions, key=lambda a: Q[(s, a)])
            best_policy[s] = {a: EPSILON / len(mdp.actions) for a in mdp.actions}
            best_policy[s][best_a] += 1 - EPSILON

        plot_value_and_policy(
            V,
            best_policy,
            mdp.env.grid,
            iteration,
            mdp.width,
            mdp.height,
            policy_name,
            prefix="vi_q",
        )

        if delta < theta:
            break

    # derive deterministic policy
    policy = {}
    for s in mdp.states:
        best_a = max(mdp.actions, key=lambda a: Q[(s, a)])
        policy[s] = {
            a: EPSILON / len(mdp.actions) for a in mdp.actions
        }  # becaue everyone has at least epsilon prob
        policy[s][best_a] += 1 - EPSILON  # make best action more probable
    return Q, policy


def policy_iteration(mdp, policy_name, gamma=0.95, theta=1e-03, max_iterations=80):
    # # uncomment if you want to set initial policy as random movement
    # policy = {s: random.choice(mdp.actions) for s in mdp.states}
    # set initial policy as moving right, for faster convergence.
    nA = len(mdp.actions)
    policy = {s: {a: 1 / nA for a in mdp.actions} for s in mdp.states}
    iteration = 0
    policy_changes = []

    while iteration < max_iterations:
        # value function on current policy
        V = policy_evaluation(policy, mdp, gamma, theta)

        old_policy = policy.copy()
        # compute policy improvement based on current value function
        policy, _ = policy_improvement(V, mdp, gamma)

        # measure how much policy has changed for each state
        changed = sum(old_policy[s] != policy[s] for s in mdp.states)
        print(f"Policy changed for {changed} states.")
        policy_changes.append(changed)

        plot_value_and_policy(
            V,
            policy,
            mdp.env.grid,
            iteration,
            mdp.width,
            mdp.height,
            policy_name,
            prefix="pi",
        )
        # print(f"[PI Iter {iteration}] policy changes: {changed}")

        # check convergence
        if changed == 0:
            break

        iteration += 1

    return V, policy


# -----Value Iteration-----#
def value_iteration(mdp, policy_name, gamma=0.95, theta=1e-3, max_iterations=80):
    V = {s: 0 for s in mdp.states}  # initial zero
    policy = {
        s: {a: a == Action.UP for a in mdp.actions} for s in mdp.states
    }  # set moving up as initial policy
    iteration = 0
    deltas = []

    while iteration < max_iterations:
        delta = 0
        new_V = V.copy()

        for s in mdp.states:
            max_value = float("-inf")
            best_action = None

            # compute q(s,a) for all actions possible for the current state
            for a in mdp.actions:
                next_s, reward, done = mdp.get_transition(s, a)
                value = reward + gamma * (0 if done else V[next_s])  # Bellman equation

                # select action with highest q value
                if value > max_value:
                    max_value = value
                    best_action = a

            # update V[s] and policy
            new_V[s] = max_value
            policy[s] = best_action

            delta = max(delta, abs(V[s] - new_V[s]))  # how much v decreased

        V = new_V
        deltas.append(delta)
        plot_value_and_policy(
            V,
            policy,
            mdp.env.grid,
            iteration,
            mdp.width,
            mdp.height,
            policy_name,
            prefix="vi",
        )

        print(f"[VI Iter {iteration}] max Δ: {delta:.5f}")

        # if delta is smaller than theta, then converged
        if delta < theta:
            break

        iteration += 1

    return V, policy

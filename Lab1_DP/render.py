# render.py

import pickle
import time
import argparse
import random
from env.gridworld_env import GridWorldEnv, Action
import os
import numpy as np


def load_policy(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy", type=str, required=False, help="Path to policy .pkl file"
    )
    parser.add_argument("--map_size", type=int, default=6)
    parser.add_argument(
        "--random", action="store_true", help="Use randomly generated map"
    )
    args = parser.parse_args()

    map_name = None
    if not args.random:
        map_name = f"map_{args.map_size}.json"

    env = GridWorldEnv(width=args.map_size, height=args.map_size, map_file=map_name)

    policy = None
    if args.policy:
        policy_pkl_path = os.path.join(f"./checkpoints/{args.policy}")
        policy = load_policy(policy_pkl_path)

    state = env.reset()

    while True:
        if policy:
            actions = policy.get(tuple(state), None)

            if actions is None:
                break

            if isinstance(actions, dict):
                # stochastic policy
                action = np.random.choice(
                    list(actions.keys()), p=list(actions.values())
                )
                highest_prob_action = max(actions, key=actions.get)
            else:
                # deterministic policy
                action = actions
                highest_prob_action = actions

            if random.random() < 0.1:
                action = random.choice(list(Action))
                print(f"Taking random action {action} instead of {highest_prob_action}")
            else:
                action = highest_prob_action
        else:
            action = random.choice(list(Action))
        state, _, done = env.step(action.value)
        env.render()
        print(f"State: {state}, Action: {action}, Done: {done}")
        time.sleep(0.25)
        print("\n" * 5)
        if done:
            time.sleep(1.0)
            state = env.reset()


if __name__ == "__main__":
    main()

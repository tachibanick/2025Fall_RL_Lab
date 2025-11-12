# render.py

import pickle
import time
import argparse
import random
from env.gridworld_env import GridWorldEnv, Action
import os
def load_policy(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, required=False, help="Path to policy .pkl file")
    parser.add_argument('--map_name', type=str, default="cliff")
    parser.add_argument('--random', action='store_true', help="Use randomly generated map")
    args = parser.parse_args()

    map_name = None
    if not args.random:
        map_name = f"map_{args.map_name}.json"

    env = GridWorldEnv(map_file=map_name)
    
    policy = None
    if args.policy:
        policy_pkl_path = os.path.join(f"./checkpoints/{args.policy}")
        policy = load_policy(policy_pkl_path)
    
    state = env.reset()

    while True:
        if policy:
            action = policy.get(tuple(state), None)
            if action is None:
                break
        else:
            action = random.choice(list(Action))
        state, _, done = env.step(action.value)
        env.render()
        time.sleep(0.25)
        print(f"State: {state}, Action: {action}, Done: {done}")
        if done:
            time.sleep(1.0)
            state = env.reset()

if __name__ == "__main__":
    main()

# env/gridworld_mdp.py

from env.gridworld_env import TileType, Action
import random


class GridWorldMDP:
    def __init__(self, env):
        self.env = env
        self.height = env.height
        self.width = env.width
        self.actions = list(Action)
        self.isRandom = True

        # define all tiles as states (if the tile is not a wall)
        self.states = [
            (y, x)
            for y in range(self.height)
            for x in range(self.width)
            if env.grid[y][x] != TileType.WALL
        ]
        # print(self.states)

    def get_transition_probabilities(self, state, action):
        return [
            (
                0.025 + (0.9 * (a == action)),
                *self.get_transition(state, a, stochastic=False),
            )
            for a in self.actions
        ]

    # state transtion: (state, action) → (next_state, reward, done)
    def get_transition(self, state, action, stochastic=True):
        y, x = state

        # PA1 10% random action
        if stochastic and random.random() < 0.1:
            action = random.choice(self.actions)

        move = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1),
        }[action]

        new_y, new_x = y + move[0], x + move[1]

        # if the next position is outside the map
        if not (0 <= new_y < self.height and 0 <= new_x < self.width):
            return state, -1, False

        tile = self.env.grid[new_y][new_x]
        if tile == TileType.WALL:
            return state, -1, False
        elif tile == TileType.TRAP:
            return (new_y, new_x), -100, True
        elif tile == TileType.GOAL:
            return (new_y, new_x), 100, True
        else:
            return (new_y, new_x), -1, False

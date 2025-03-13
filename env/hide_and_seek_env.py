"""
env/hide_and_seek_env.py

This module defines the HideAndSeekEnv environment for a multi-agent hide and seek project.
It leverages the Room class (from env.room) for room layout and door properties,
and uses agent classes from the agents package (Seeker and Hider) for managing agent state.
The grid is 10x10 and the room is a 4x4 area at the bottom right corner.
Debug and informational output is standardized via the custom logger.

Reward system (very simple):
1. Vision-Based Reward (when the seeker is active):
   - In each step, if the active seeker sees the hider (i.e. the hider's cell is in the
     seeker's computed visible cells), then the seeker gets +1 reward and the hider gets -1 reward.
2. In-Room Reward for the Hider:
   - Every step, if the hider is inside the room (as determined by room.is_inside(cell)), the hider receives +1 reward.

Door actions ("toggle_door", "lock", "unlock") are still available to both agents for interaction,
but they no longer affect the reward.
The hider is spawned at reset, while the seeker is spawned after 10 steps.
When the seeker is inactive, its state and FOV are not rendered.
The step() method returns a tuple: (observation, done, rewards).
"""

import gym
from gym import spaces
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

from env.room import Room
from agents.seeker import Seeker
from agents.hider import Hider
from utils.logger import log_debug, log_info
from utils.visualization import render_environment

class HideAndSeekEnv(gym.Env):
    def __init__(self):
        super(HideAndSeekEnv, self).__init__()

        # Grid dimensions
        self.grid_size = 10

        # Initialize a 4x4 room at the bottom right corner (top_left=(6,6))
        self.room = Room(top_left=(self.grid_size - 4, self.grid_size - 4), width=4, height=4, door_side="left")
        # log_debug(f"Using room at {self.room.top_left} with door at {self.room.door.position}")

        # Define observation space: each agent's state as (x, y, direction)
        self.observation_space = spaces.Dict({
            "seeker": spaces.Box(low=0, high=self.grid_size - 1, shape=(3,), dtype=np.int32),
            "hider": spaces.Box(low=0, high=self.grid_size - 1, shape=(3,), dtype=np.int32)
        })

        # Both agents: 0-3 for movement; 4->"toggle_door", 5->"lock", 6->"unlock"
        self.action_space = spaces.Dict({
            "seeker": spaces.Discrete(7),
            "hider": spaces.Discrete(7)
        })

        # Mapping of movement actions to (dx, dy)
        self.moves = {
            0: (0, -1),
            1: (1, 0),
            2: (0, 1),
            3: (-1, 0)
        }

        # Mapping for door actions (for both agents)
        self.door_actions = {4: "toggle_door", 5: "lock", 6: "unlock"}

        self.max_steps = 100
        self.step_count = 0

        # Initially, only spawn the hider; the seeker is spawned after 10 steps.
        self.hider = None
        self.seeker = None
        self.seeker_active = False

        self.viewer_initialized = False
        self.fig = None
        self.ax = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # log_debug(f"Using device: {self.device}")

    def compute_visible_cells(self, state, max_distance=10, num_rays=7):
        """
        Compute a triangular (90° wedge) field of view for an agent.
        The agent's cell is visible, and for each row i (starting at 1),
        the width of visible cells is (2*i+1), unless occluded by a wall or closed door.

        Args:
            state (tuple): (x, y, d) representing the agent's state.
            max_distance (int): Maximum number of rows to check.
            num_rays (int): Not used here.

        Returns:
            list: List of (x, y) tuples that are visible.
        """
        x, y, d = state
        visible = {(x, y)}
        for i in range(1, max_distance + 1):
            if d == 0:  # facing up
                row_y = y - i
                for dx in range(-i, i + 1):
                    cell = (x + dx, row_y)
                    if not (0 <= cell[0] < self.grid_size and 0 <= cell[1] < self.grid_size):
                        continue
                    visible.add(cell)
                    if self.room.blocks_vision(cell):
                        break
            elif d == 1:  # facing right
                row_x = x + i
                for dy in range(-i, i + 1):
                    cell = (row_x, y + dy)
                    if not (0 <= cell[0] < self.grid_size and 0 <= cell[1] < self.grid_size):
                        continue
                    visible.add(cell)
                    if self.room.blocks_vision(cell):
                        break
            elif d == 2:  # facing down
                row_y = y + i
                for dx in range(-i, i + 1):
                    cell = (x + dx, row_y)
                    if not (0 <= cell[0] < self.grid_size and 0 <= cell[1] < self.grid_size):
                        continue
                    visible.add(cell)
                    if self.room.blocks_vision(cell):
                        break
            elif d == 3:  # facing left
                row_x = x - i
                for dy in range(-i, i + 1):
                    cell = (row_x, y + dy)
                    if not (0 <= cell[0] < self.grid_size and 0 <= cell[1] < self.grid_size):
                        continue
                    visible.add(cell)
                    if self.room.blocks_vision(cell):
                        break
        return list(visible)

    def reset(self):
        # log_debug("Resetting environment...")
        self.room.door.is_open = True
        self.room.door.is_locked = False
        self.step_count = 0
        self.seeker_active = False

        def get_valid_position():
            while True:
                x = np.random.randint(0, self.grid_size)
                y = np.random.randint(0, self.grid_size)
                if not self.room.is_wall((x, y)):
                    return x, y

        # Spawn hider first
        hx, hy = get_valid_position()
        hider_dir = np.random.randint(0, 4)
        self.hider = Hider(hx, hy, hider_dir)
        self.seeker = None

        # log_debug(f"Hider initial state: {self.hider.get_state()}")
        return {
            "seeker": {
                "state": (-1, -1, -1),  # Dummy state for inactive seeker
                "visible": []
            },
            "hider": {
                "state": self.hider.get_state(),
                "visible": self.compute_visible_cells(self.hider.get_state())
            }
        }

    def spawn_seeker(self):
        def get_valid_position():
            while True:
                x = np.random.randint(0, self.grid_size)
                y = np.random.randint(0, self.grid_size)
                if not self.room.is_wall((x, y)):
                    return x, y
        sx, sy = get_valid_position()
        seeker_dir = np.random.randint(0, 4)
        self.seeker = Seeker(sx, sy, seeker_dir)
        # log_debug(f"Seeker spawned with state: {self.seeker.get_state()}")

    def is_valid_move(self, x, y, dx, dy):
        new_x, new_y = x + dx, y + dy
        if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size):
            # log_debug(f"Invalid move: ({new_x}, {new_y}) is out of bounds.")
            return False
        new_pos = (new_x, new_y)
        if self.room.is_wall(new_pos):
            if self.room.is_door(new_pos) and self.room.door.is_open and not self.room.door.is_locked:
                return True
            else:
                # log_debug(f"Invalid move: ({new_x}, {new_y}) is blocked by a room wall.")
                return False
        return True

    def step(self, actions):
        self.step_count += 1
        # log_debug(f"Step {self.step_count} starting...")
        rewards = {"seeker": 0.0, "hider": 0.0}

        # Spawn seeker after 10 steps if not active
        if not self.seeker_active and self.step_count >= 10:
            self.spawn_seeker()
            self.seeker_active = True

        # Process door actions for both agents (but no reward is given for door actions in the new system)
        for agent_key in ["seeker", "hider"]:
            action = actions.get(agent_key)
            if agent_key == "seeker" and not self.seeker_active:
                continue
            agent = self.seeker if agent_key == "seeker" else self.hider
            x, y, _ = agent.get_state()
            if self.room.is_door((x, y)):
                if action in self.door_actions.values():
                    self.room.door.toggle() if action == "toggle_door" else None
                    self.room.door.lock() if action == "lock" else None
                    self.room.door.unlock() if action == "unlock" else None
                    # No door action rewards in the new simple reward system

        # Process movement actions
        for agent_key in ["seeker", "hider"]:
            action = actions.get(agent_key)
            if agent_key == "hider" and isinstance(action, int) and action >= 4:
                continue
            if agent_key == "seeker" and not self.seeker_active:
                continue
            if isinstance(action, int) and action in self.moves:
                dx, dy = self.moves[action]
                agent = self.seeker if agent_key == "seeker" else self.hider
                x, y, _ = agent.get_state()
                if self.is_valid_move(x, y, dx, dy):
                    new_x, new_y = x + dx, y + dy
                    agent.update_state(x=new_x, y=new_y, direction=action)
            # else:
            #     log_debug(f"{agent_key.capitalize()} action ({action}) is not a movement command.")

        # Vision-based reward: if seeker is active and sees the hider, seeker gets +1 and hider gets -1.
        if self.seeker_active:
            visible_seeker = set(self.compute_visible_cells(self.seeker.get_state()))
            hider_cell = self.hider.get_state()[:2]
            if hider_cell in visible_seeker:
                rewards["seeker"] += 1.0
                rewards["hider"] += -1.0

        # In-Room Reward for hider: if hider is inside the room, add +1 reward.
        if self.room.is_inside(self.hider.get_state()[:2]):
            rewards["hider"] += 1.0

        done = (self.step_count >= self.max_steps)
        obs = {
            "seeker": {
                "state": self.seeker.get_state() if self.seeker_active else (-1, -1, -1),
                "visible": self.compute_visible_cells(self.seeker.get_state()) if self.seeker_active else []
            },
            "hider": {
                "state": self.hider.get_state(),
                "visible": self.compute_visible_cells(self.hider.get_state())
            }
        }
        return obs, done, rewards

    def render(self, mode='human'):
        if not self.viewer_initialized:
            self.fig, self.ax = plt.subplots()
            self.viewer_initialized = True
        render_environment(self.ax, self)
        plt.pause(0.001)


if __name__ == "__main__":
    env = HideAndSeekEnv()
    observation = env.reset()
    env.render()
    log_info("Environment visualized. Starting random actions loop...")  # Uncomment for debugging
    try:
        while True:
            hider_random = np.random.choice(list(range(7)))  # 0-3 movement; 4-6 door actions.
            if hider_random >= 4:
                hider_action = {4: "toggle_door", 5: "lock", 6: "unlock"}[hider_random]
            else:
                hider_action = hider_random
            seeker_action = np.random.choice([0, 1, 2, 3]) if env.seeker_active else 0
            actions = {
                "seeker": seeker_action,
                "hider": hider_action
            }
            observation, done, rewards = env.step(actions)
            env.render()
            # log_debug(f"Rewards: {rewards}")  # Uncomment for debugging
            if done:
                log_info("Episode finished. Resetting environment...")  # Uncomment for debugging
                env.reset()
            plt.pause(0.5)
    except KeyboardInterrupt:
        log_info("Exiting visualization loop.")  # Uncomment for debugging
        plt.close()
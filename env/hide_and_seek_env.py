"""
env/hide_and_seek_env.py

This module defines the HideAndSeekEnv environment for a multi-agent hide and seek project.
It leverages the Room class (from env.room) for room layout and door properties,
and uses agent classes from the agents package (Seeker and Hider) for managing agent state.
The grid is 10x10 and the room is a 4x4 area at the bottom right corner.
Debug and informational output is standardized via the custom logger.

Additionally, this version supports door actions: "toggle_door", "lock", and "unlock".
For door actions performed by either agent while at the door cell:
  - For the hider:
      * If toggling an open door (thus closing it): +2 reward.
      * If locking a closed door: +4 reward.
  - For the seeker:
      * If locking an open door: +5 reward.
In each step, if the seeker is active:
  - If the seeker sees the hider, then the seeker gets +10 reward, the hider gets -10, and the episode terminates.
  - Otherwise, the hider receives +1 reward.

The step() method returns a tuple: (observation, done, door_rewards).
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
        log_debug(f"Using room at {self.room.top_left} with door at {self.room.door.position}")  # Uncomment for debugging

        # Define observation space: each agent's state as (x, y, direction)
        self.observation_space = spaces.Dict({
            "seeker": spaces.Box(low=0, high=self.grid_size - 1, shape=(3,), dtype=np.int32),
            "hider": spaces.Box(low=0, high=self.grid_size - 1, shape=(3,), dtype=np.int32)
        })

        # Define action space:
        # Both agents: 0-3 for movement; 4->"toggle_door", 5->"lock", 6->"unlock"
        self.action_space = spaces.Dict({
            "seeker": spaces.Discrete(7),
            "hider": spaces.Discrete(7)
        })

        # Mapping of movement actions to (dx, dy)
        self.moves = {
            0: (0, -1),   # Up
            1: (1, 0),    # Right
            2: (0, 1),    # Down
            3: (-1, 0)    # Left
        }

        # Mapping for door actions (for both agents)
        self.door_actions = {4: "toggle_door", 5: "lock", 6: "unlock"}

        self.max_steps = 100
        self.step_count = 0

        # Initially, only the hider is spawned; seeker will be spawned after 10 steps.
        self.hider = None
        self.seeker = None
        self.seeker_active = False

        self.viewer_initialized = False
        self.fig = None
        self.ax = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_debug(f"Using device: {self.device}")  # Uncomment for debugging

    def compute_visible_cells(self, state, max_distance=10, num_rays=7):
        """
        Compute a triangular field of view (90° wedge) for an agent.
        The agent's current cell is always visible. For each row i (1-indexed), the row in front
        of the agent has (2*i + 1) cells with the agent in the middle.

        Args:
            state (tuple): (x, y, d) representing the agent's state.
            max_distance (int): Maximum number of rows to check.
            num_rays (int): Not used in this implementation.

        Returns:
            list: List of (x, y) tuples that are visible.
        """
        x, y, d = state
        visible = {(x, y)}
        for i in range(1, max_distance + 1):
            if d == 0:  # Facing up
                row_y = y - i
                for dx in range(-i, i + 1):
                    cell = (x + dx, row_y)
                    if not (0 <= cell[0] < self.grid_size and 0 <= cell[1] < self.grid_size):
                        continue
                    visible.add(cell)
                    if self.room.blocks_vision(cell):
                        break
            elif d == 1:  # Facing right
                row_x = x + i
                for dy in range(-i, i + 1):
                    cell = (row_x, y + dy)
                    if not (0 <= cell[0] < self.grid_size and 0 <= cell[1] < self.grid_size):
                        continue
                    visible.add(cell)
                    if self.room.blocks_vision(cell):
                        break
            elif d == 2:  # Facing down
                row_y = y + i
                for dx in range(-i, i + 1):
                    cell = (x + dx, row_y)
                    if not (0 <= cell[0] < self.grid_size and 0 <= cell[1] < self.grid_size):
                        continue
                    visible.add(cell)
                    if self.room.blocks_vision(cell):
                        break
            elif d == 3:  # Facing left
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
        log_debug("Resetting environment...")  # Uncomment for debugging
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

        # Seeker is not spawned yet; its state is dummy.
        self.seeker = None

        log_debug(f"Hider initial state: {self.hider.get_state()}")  # Uncomment for debugging
        return {
            "seeker": {
                "state": (-1, -1, -1),  # Indicates inactive seeker
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
        log_debug(f"Seeker spawned with state: {self.seeker.get_state()}")  # Uncomment for debugging

    def is_valid_move(self, x, y, dx, dy):
        new_x, new_y = x + dx, y + dy
        if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size):
            log_debug(f"Invalid move: ({new_x}, {new_y}) is out of bounds.")  # Uncomment for debugging
            return False
        new_pos = (new_x, new_y)
        if self.room.is_wall(new_pos):
            if self.room.is_door(new_pos) and self.room.door.is_open and not self.room.door.is_locked:
                return True
            else:
                log_debug(f"Invalid move: ({new_x}, {new_y}) is blocked by a room wall.")  # Uncomment for debugging
                return False
        return True

    def step(self, actions):
        self.step_count += 1
        log_debug(f"Step {self.step_count} starting...")  # Uncomment for debugging
        door_rewards = {"seeker": 0.0, "hider": 0.0}

        # Spawn seeker after 10 steps if not active
        if not self.seeker_active and self.step_count >= 10:
            self.spawn_seeker()
            self.seeker_active = True

        # Process door actions for agents on the door cell
        for agent_key in ["seeker", "hider"]:
            action = actions.get(agent_key)
            # Skip door processing for seeker if not active
            if agent_key == "seeker" and not self.seeker_active:
                continue
            agent = self.seeker if agent_key == "seeker" else self.hider
            x, y, _ = agent.get_state()
            if self.room.is_door((x, y)):
                if action == "toggle_door":
                    if agent_key == "hider":
                        if self.room.door.is_open:
                            self.room.door.toggle()
                            if not self.room.door.is_open:
                                door_rewards["hider"] += 2.0
                        else:
                            self.room.door.toggle()
                    else:
                        self.room.door.toggle()
                elif action == "lock":
                    if agent_key == "hider":
                        if not self.room.door.is_open:
                            self.room.door.lock()
                            if self.room.door.is_locked:
                                door_rewards["hider"] += 4.0
                        else:
                            self.room.door.lock()
                            door_rewards["seeker"] += 5.0
                    else:
                        self.room.door.lock()
                elif action == "unlock":
                    self.room.door.unlock()

        # Process movement actions for each agent
        for agent_key in ["seeker", "hider"]:
            action = actions.get(agent_key)
            if agent_key == "hider" and isinstance(action, int) and action >= 4:
                continue  # Door action already processed
            if agent_key == "seeker" and not self.seeker_active:
                continue  # Seeker inactive: skip movement
            if isinstance(action, int) and action in self.moves:
                dx, dy = self.moves[action]
                agent = self.seeker if agent_key == "seeker" else self.hider
                x, y, d = agent.get_state()
                if self.is_valid_move(x, y, dx, dy):
                    new_x, new_y = x + dx, y + dy
                    agent.update_state(x=new_x, y=new_y, direction=action)
                else:
                    log_debug(f"{agent_key.capitalize()} move blocked. Staying at ({x}, {y})")  # Uncomment for debugging
            else:
                log_debug(f"{agent_key.capitalize()} action ({action}) is not a movement command.")  # Uncomment for debugging

        # After movement, if the seeker is active, check vision:
        if self.seeker_active:
            visible_seeker = set(self.compute_visible_cells(self.seeker.get_state()))
            hider_cell = self.hider.get_state()[:2]
            if hider_cell in visible_seeker:
                # Seeker sees hider: reward seeker +10, hider -10, terminate episode.
                door_rewards["seeker"] += 10.0
                door_rewards["hider"] += -10.0
                log_debug("Seeker sees the hider! Terminating episode.")  # Uncomment for debugging
                done = True
            else:
                door_rewards["hider"] += 1.0
                done = (self.step_count >= self.max_steps)
        else:
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
        return obs, done, door_rewards

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

            # For seeker, if inactive, choose a dummy action.
            seeker_action = np.random.choice([0, 1, 2, 3]) if env.seeker_active else 0

            actions = {
                "seeker": seeker_action,
                "hider": hider_action
            }
            observation, done, door_rewards = env.step(actions)
            env.render()
            log_debug(f"Door rewards: {door_rewards}")  # Uncomment for debugging
            if done:
                log_info("Episode finished. Resetting environment...")  # Uncomment for debugging
                env.reset()
            plt.pause(0.5)
    except KeyboardInterrupt:
        log_info("Exiting visualization loop.")  # Uncomment for debugging
        plt.close()

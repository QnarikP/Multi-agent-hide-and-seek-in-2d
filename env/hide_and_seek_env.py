"""
env/hide_and_seek_env.py

This module defines the HideAndSeekEnv environment for a multi-agent hide and seek project.
It leverages the Room class (from env.room) for room layout and door properties,
and uses agent classes from the agents package (Seeker and Hider) for managing agent state.
The grid is 10x10 and the room is a 4x4 area at the bottom right corner.
Debug and informational output is standardized via the custom logger.

Reward system:
1. Door Action Rewards:
   - Hider Actions:
       * "toggle_door":
            - If the hider toggles an open door (thus closing it): +2 reward.
            - If the door is already closed, toggling opens it: -1 reward.
       * "lock":
            - If the hider locks the door when it is closed and the hider is inside the room: +4 reward.
            - If the hider locks the door while it is open: -5 reward.
       * "unlock":
            - If the hider is inside the room and the door is closed and locked, unlocking it: -5 reward.
   - Seeker Actions:
       * If the seeker locks an open door, it receives +7 reward.
2. Vision-Based Rewards (when the seeker is active):
   - In every step that the active seeker sees the hider:
         * Seeker gets +10 reward.
         * Hider gets -10 reward.
3. In-Room Reward for the Hider:
   - Every step, if the hider is inside the room: +1 reward.

Additionally, the hider is spawned at reset while the seeker is spawned after 10 steps
(to give the hider time to hide). When the seeker is inactive, its state and FOV are not rendered.
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
        # log_debug(f"Using room at {self.room.top_left} with door at {self.room.door.position}")

        # Define observation space: each agent's state as (x, y, direction)
        self.observation_space = spaces.Dict({
            "seeker": spaces.Box(low=0, high=self.grid_size - 1, shape=(3,), dtype=np.int32),
            "hider": spaces.Box(low=0, high=self.grid_size - 1, shape=(3,), dtype=np.int32)
        })

        # Both agents: 0-3 for movement; 4 -> "toggle_door", 5 -> "lock", 6 -> "unlock"
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
        # Seeker is not spawned yet.
        self.seeker = None

        # log_debug(f"Hider initial state: {self.hider.get_state()}")
        return {
            "seeker": {
                "state": (-1, -1, -1),  # Dummy state indicating inactive seeker
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
        door_rewards = {"seeker": 0.0, "hider": 0.0}

        # Spawn seeker after 10 steps if not active
        if not self.seeker_active and self.step_count >= 10:
            self.spawn_seeker()
            self.seeker_active = True

        # Process door actions for agents on the door cell
        for agent_key in ["seeker", "hider"]:
            action = actions.get(agent_key)
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
                            door_rewards["hider"] += -1.0
                    else:
                        self.room.door.toggle()
                elif action == "lock":
                    if agent_key == "hider":
                        if not self.room.door.is_open:
                            self.room.door.lock()
                            if self.room.door.is_locked and self.room.is_inside((x, y)):
                                door_rewards["hider"] += 4.0
                        else:
                            self.room.door.lock()
                            door_rewards["hider"] += -5.0
                    else:
                        if self.room.door.is_open:
                            self.room.door.lock()
                            door_rewards["seeker"] += 7.0
                        else:
                            self.room.door.lock()
                elif action == "unlock":
                    if agent_key == "hider":
                        if self.room.is_inside((x, y)) and (not self.room.door.is_open) and self.room.door.is_locked:
                            self.room.door.unlock()
                            door_rewards["hider"] += -5.0
                        else:
                            self.room.door.unlock()
                    else:
                        self.room.door.unlock()

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
                x, y, d = agent.get_state()
                if self.is_valid_move(x, y, dx, dy):
                    new_x, new_y = x + dx, y + dy
                    agent.update_state(x=new_x, y=new_y, direction=action)
            #     else:
            #         log_debug(f"{agent_key.capitalize()} move blocked. Staying at ({x}, {y})")
            # else:
            #     log_debug(f"{agent_key.capitalize()} action ({action}) is not a movement command.")

        # Vision-based reward: if seeker is active, check visible cells.
        if self.seeker_active:
            visible_seeker = set(self.compute_visible_cells(self.seeker.get_state()))
            hider_cell = self.hider.get_state()[:2]
            if hider_cell in visible_seeker:
                door_rewards["seeker"] += 10.0
                door_rewards["hider"] += -10.0

        # Extra reward: if the hider is inside the room, add +1 reward.
        if self.room.is_inside(self.hider.get_state()[:2]):
            door_rewards["hider"] += 1.0

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
    log_info("Environment visualized. Starting random actions loop...")
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
            observation, done, door_rewards = env.step(actions)
            env.render()
            # log_debug(f"Door rewards: {door_rewards}")
            if done:
                log_info("Episode finished. Resetting environment...")
                env.reset()
            plt.pause(0.5)
    except KeyboardInterrupt:
        log_info("Exiting visualization loop.")
        plt.close()
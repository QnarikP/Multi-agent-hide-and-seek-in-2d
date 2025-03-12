"""
training/train_rl.py

This module implements the training loop for the multi-agent hide and seek project using DQN.
We create two independent DQN agents—one for the seeker and one for the hider.
At each step:
  - The environment is reset.
  - Each agent selects an action using an epsilon-greedy policy.
  - The environment processes the actions and returns the next state (and door rewards).
  - A reward function is applied:
        * If the seeker finds the hider (occupies the same cell), the seeker gets +1 and the hider gets -1.
        * Additionally, if the hider is in the room and closes the door (via "toggle_door") it gets +2 reward,
          and if it locks the door it gets +4 reward.
  - The agents store experiences and perform training steps.
  - Target networks are updated periodically.

This version ensures that all computations are performed on the GPU (if available) and uses our visualization routines.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from env.hide_and_seek_env import HideAndSeekEnv
from training.rl_agent import DQNAgent
from utils.logger import log_info
from utils.visualization import visualize_all_metrics  # Assume this function is defined in utils/visualization.py

# Hyperparameters for training
num_episodes = 10000
max_steps_per_episode = 100
target_update_frequency = 10  # Update target networks every 10 episodes

# Initialize the environment
env = HideAndSeekEnv()
state_dim = 3    # State: (x, y, direction)
seeker_action_dim = 4  # Seeker: movement actions (0:Up,1:Right,2:Down,3:Left)
hider_action_dim = 7   # Hider: 0-3 for movement; 4:"toggle_door", 5:"lock", 6:"unlock"

# Set device to GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_info(f"Training on device: {device}")

# Initialize DQN agents for seeker and hider (pass the appropriate action dimensions)
seeker_agent = DQNAgent(state_dim, seeker_action_dim, device=device)
hider_agent = DQNAgent(state_dim, hider_action_dim, device=device)

# Metrics to store per episode
rewards_seeker_list = []
rewards_hider_list = []
penalties_seeker_list = []
penalties_hider_list = []
invalid_moves_seeker_list = []  # Placeholder – update if env provides this info.
invalid_moves_hider_list = []   # Placeholder – update if env provides this info.

# Mapping for hider door actions (consistent with env)
hider_door_mapping = {4: "toggle_door", 5: "lock", 6: "unlock"}

# Training loop over episodes
for episode in range(num_episodes):
    state = env.reset()
    # Convert agent states to float tensors and move to GPU
    seeker_state = torch.tensor(state['seeker']['state'], dtype=torch.float32).to(device)
    hider_state = torch.tensor(state['hider']['state'], dtype=torch.float32).to(device)
    log_info(f"Episode {episode + 1}/{num_episodes} started. "
             f"Initial seeker state: {seeker_state}, hider state: {hider_state}")

    total_reward_seeker = 0
    total_reward_hider = 0
    penalty_seeker = 0
    penalty_hider = 0
    invalid_moves_seeker = 0  # Placeholder counter
    invalid_moves_hider = 0   # Placeholder counter

    for step in range(max_steps_per_episode):
        # Seeker selects an action (always integer 0-3)
        action_seeker = seeker_agent.select_action(seeker_state.cpu().numpy())
        # Hider selects an action (integer 0-6)
        hider_action_int = hider_agent.select_action(hider_state.cpu().numpy())
        # Map hider action: if < 4, it's a movement; if >=4, map to door action string.
        if hider_action_int < 4:
            action_hider = hider_action_int
        else:
            action_hider = hider_door_mapping[hider_action_int]
        actions = {"seeker": action_seeker, "hider": action_hider}

        # Apply actions in the environment and receive next state, done flag, and door rewards
        next_state, done, door_rewards = env.step(actions)
        next_seeker_state = torch.tensor(next_state['seeker']['state'], dtype=torch.float32).to(device)
        next_hider_state = torch.tensor(next_state['hider']['state'], dtype=torch.float32).to(device)

        # Base reward function:
        # If the seeker and hider are in the same cell, seeker gets +1 and hider gets -1.
        if seeker_state[0].item() == hider_state[0].item() and seeker_state[1].item() == hider_state[1].item():
            reward_seeker = 1.0
            reward_hider = -1.0
            penalty_hider += 1  # Count penalty for hider being caught
        else:
            reward_seeker = 0.0
            reward_hider = 0.0

        # Add door rewards (extra rewards for the hider's door actions)
        reward_hider += door_rewards.get("hider", 0.0)

        total_reward_seeker += reward_seeker
        total_reward_hider += reward_hider

        # Store experiences in each agent's replay buffer
        seeker_agent.store_experience(seeker_state, action_seeker, reward_seeker, next_seeker_state, done)
        hider_agent.store_experience(hider_state, hider_action_int, reward_hider, next_hider_state, done)

        # Update current state to next state
        seeker_state = next_seeker_state
        hider_state = next_hider_state

        # Perform a training step (if enough experiences are available)
        seeker_agent.train_step()
        hider_agent.train_step()

        # Visualize the environment every 10 steps
        # if step % 10 == 0:
        #     env.render()

        if done:
            break

    log_info(
        f"Episode {episode + 1} finished. Total reward - Seeker: {total_reward_seeker}, Hider: {total_reward_hider}"
    )

    # Append metrics for this episode
    rewards_seeker_list.append(total_reward_seeker)
    rewards_hider_list.append(total_reward_hider)
    penalties_seeker_list.append(penalty_seeker)
    penalties_hider_list.append(penalty_hider)
    invalid_moves_seeker_list.append(invalid_moves_seeker)
    invalid_moves_hider_list.append(invalid_moves_hider)

    # Periodically update the target networks for stable learning
    if (episode + 1) % target_update_frequency == 0:
        seeker_agent.update_target_network()
        hider_agent.update_target_network()

    if episode % 100 == 0:
        # Specify filenames where the models are saved
        seeker_model_filename = f'seeker_dqn_model_{episode}.pth'
        hider_model_filename = f'hider_dqn_model_{episode}.pth'

        # Save models after training is complete
        seeker_agent.save_model(seeker_model_filename)
        hider_agent.save_model(hider_model_filename)


# Final rendering and show the plot window
env.render()
plt.show()

# Save plots of the collected metrics
metrics = {
    'rewards_seeker': rewards_seeker_list,
    'rewards_hider': rewards_hider_list,
    'penalties_seeker': penalties_seeker_list,
    'penalties_hider': penalties_hider_list,
    'invalid_moves_seeker': invalid_moves_seeker_list,
    'invalid_moves_hider': invalid_moves_hider_list,
}
visualize_all_metrics(metrics, filename_prefix="training_metrics")

# Specify filenames where the models are saved
seeker_model_filename = f'last_seeker_dqn_model_{num_episodes}.pth'
hider_model_filename = f'last_hider_dqn_model_{num_episodes}.pth'

# Save models after training is complete
seeker_agent.save_model(seeker_model_filename)
hider_agent.save_model(hider_model_filename)

log_info("Models saved successfully.")
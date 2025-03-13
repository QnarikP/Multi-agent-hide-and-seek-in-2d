"""
training/train_rl.py

This module implements the training loop for the multi-agent hide and seek project using DQN.
We create two independent DQN agents—one for the seeker and one for the hider.
At each step:
  - The environment is reset.
  - Each agent selects an action using an epsilon-greedy policy.
  - The environment processes the actions and returns the next state along with door/vision rewards.
  - Reward rules:
        * If the seeker sees the hider (once active), the seeker gets +10 and the hider gets -10.
        * Otherwise, if the hider is inside the room, it gains +1 reward.
        * Door actions provide additional rewards:
             - Hider "toggle_door": if door open then +2, if door closed then -1.
             - Hider "lock": if door closed and inside room: +4, if door open: -5.
             - Hider "unlock": if inside room and door closed+locked: -5.
             - Seeker "lock" on open door: +7.
  - The agents store experiences and perform training steps.
  - Target networks are updated periodically.

This version ensures GPU usage (if available) and uses our visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from env.hide_and_seek_env import HideAndSeekEnv
from training.rl_agent import DQNAgent
from utils.logger import log_info
from utils.visualization import visualize_all_metrics

# Hyperparameters
num_episodes = 1e6
max_steps_per_episode = 100
target_update_frequency = 10

env = HideAndSeekEnv()
state_dim = 3
agent_action_dim = 7  # Both agents have 7 actions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_info(f"Training on device: {device}")

seeker_agent = DQNAgent(state_dim, agent_action_dim, device=device)
hider_agent = DQNAgent(state_dim, agent_action_dim, device=device)

# Metrics
rewards_seeker_list = []
rewards_hider_list = []
penalties_seeker_list = []
penalties_hider_list = []
invalid_moves_seeker_list = []
invalid_moves_hider_list = []

door_mapping = {4: "toggle_door", 5: "lock", 6: "unlock"}
best_reward_seeker = -np.inf
best_reward_hider = -np.inf

for episode in range(num_episodes):
    state = env.reset()
    seeker_state = torch.tensor(state['seeker']['state'], dtype=torch.float32).to(device)
    hider_state = torch.tensor(state['hider']['state'], dtype=torch.float32).to(device)
    log_info(f"Episode {episode+1}/{num_episodes} started. Initial seeker state: {seeker_state}, hider state: {hider_state}")

    total_reward_seeker = 0
    total_reward_hider = 0
    penalty_seeker = 0
    penalty_hider = 0
    invalid_moves_seeker = 0
    invalid_moves_hider = 0

    for step in range(max_steps_per_episode):
        if env.seeker_active:
            action_seeker = seeker_agent.select_action(seeker_state.cpu().numpy())
        else:
            action_seeker = 0
        hider_action_int = hider_agent.select_action(hider_state.cpu().numpy())
        if hider_action_int < 4:
            action_hider = hider_action_int
        else:
            action_hider = door_mapping[hider_action_int]
        actions = {"seeker": action_seeker, "hider": action_hider}

        next_state, done, door_rewards = env.step(actions)
        next_seeker_state = torch.tensor(next_state['seeker']['state'], dtype=torch.float32).to(device)
        next_hider_state = torch.tensor(next_state['hider']['state'], dtype=torch.float32).to(device)

        reward_seeker = door_rewards.get("seeker", 0.0)
        reward_hider = door_rewards.get("hider", 0.0)
        total_reward_seeker += reward_seeker
        total_reward_hider += reward_hider

        seeker_agent.store_experience(seeker_state, action_seeker, reward_seeker, next_seeker_state, done)
        hider_agent.store_experience(hider_state, hider_action_int, reward_hider, next_hider_state, done)

        seeker_state = next_seeker_state
        hider_state = next_hider_state

        seeker_agent.train_step()
        hider_agent.train_step()

        # if step % 10 == 0:
        #     env.render()

        if done:
            break

    log_info(f"Episode {episode+1} finished. Total reward - Seeker: {total_reward_seeker}, Hider: {total_reward_hider}")
    rewards_seeker_list.append(total_reward_seeker)
    rewards_hider_list.append(total_reward_hider)
    penalties_seeker_list.append(penalty_seeker)
    penalties_hider_list.append(penalty_hider)
    invalid_moves_seeker_list.append(invalid_moves_seeker)
    invalid_moves_hider_list.append(invalid_moves_hider)

    if (episode+1) % target_update_frequency == 0:
        seeker_agent.update_target_network()
        hider_agent.update_target_network()

    if total_reward_seeker > best_reward_seeker:
        best_reward_seeker = total_reward_seeker
        seeker_agent.save_model("best_seeker_dqn_model.pth")
        log_info(f"New best seeker reward: {best_reward_seeker} - model saved.")

    if total_reward_hider > best_reward_hider:
        best_reward_hider = total_reward_hider
        hider_agent.save_model("best_hider_dqn_model.pth")
        log_info(f"New best hider reward: {best_reward_hider} - model saved.")

# env.render()
# plt.show()

metrics = {
    'rewards_seeker': rewards_seeker_list,
    'rewards_hider': rewards_hider_list,
    'penalties_seeker': penalties_seeker_list,
    'penalties_hider': penalties_hider_list,
    'invalid_moves_seeker': invalid_moves_seeker_list,
    'invalid_moves_hider': invalid_moves_hider_list,
}
visualize_all_metrics(metrics, filename_prefix="training_metrics")

seeker_model_filename = 'seeker_dqn_model.pth'
hider_model_filename = 'hider_dqn_model.pth'

seeker_agent.save_model(seeker_model_filename)
hider_agent.save_model(hider_model_filename)

log_info("Models saved successfully.")
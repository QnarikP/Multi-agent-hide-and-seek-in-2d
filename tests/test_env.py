import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
from env.hide_and_seek_env import HideAndSeekEnv
from training.rl_agent import DQNAgent
from utils.visualization import render_environment
from utils.logger import log_info

# Test parameters
num_test_episodes = 10
max_steps_per_episode = 50
output_video_path = "test_video.mp4"
output_gif_path = "test_animation.gif"

env = HideAndSeekEnv()
state_dim = 3
agent_action_dim = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_info(f"Testing on device: {device}")

seeker_agent = DQNAgent(state_dim, agent_action_dim, device=device)
hider_agent = DQNAgent(state_dim, agent_action_dim, device=device)

seeker_agent.load_model(r'C:\Users\User\Projects\NPUA\multi_agent_hide_and_seek\training\models\seeker_dqn_model.pth')
hider_agent.load_model(r'C:\Users\User\Projects\NPUA\multi_agent_hide_and_seek\training\models\best_hider_dqn_model.pth')

seeker_agent.q_network.to(device)
hider_agent.q_network.to(device)

fig, ax = plt.subplots(figsize=(8, 8))
frames = []

for episode in range(num_test_episodes):
    state = env.reset()
    seeker_state = torch.tensor(state['seeker']['state'], dtype=torch.float32).to(device)
    hider_state = torch.tensor(state['hider']['state'], dtype=torch.float32).to(device)

    total_reward_seeker = 0
    total_reward_hider = 0

    log_info(f"Test Episode {episode + 1} started.")

    for step in range(max_steps_per_episode):
        if env.seeker_active:
            action_seeker = seeker_agent.select_action(seeker_state.cpu().numpy())
        else:
            action_seeker = 0
        hider_action_int = hider_agent.select_action(hider_state.cpu().numpy())
        if hider_action_int < 4:
            action_hider = hider_action_int
        else:
            action_hider = {4: "toggle_door", 5: "lock", 6: "unlock"}[hider_action_int]
        actions = {"seeker": action_seeker, "hider": action_hider}

        next_state, done, door_rewards = env.step(actions)
        render_environment(ax, env)
        frame = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        width, height = fig.canvas.get_width_height()
        frame = frame.reshape((height, width, 4))
        frame = frame[..., 1::-1]
        frames.append(frame)

        next_seeker_state = torch.tensor(next_state['seeker']['state'], dtype=torch.float32).to(device)
        next_hider_state = torch.tensor(next_state['hider']['state'], dtype=torch.float32).to(device)

        reward_seeker = door_rewards.get("seeker", 0.0)
        reward_hider = door_rewards.get("hider", 0.0)

        total_reward_seeker += reward_seeker
        total_reward_hider += reward_hider

        seeker_state = next_seeker_state
        hider_state = next_hider_state

        if done:
            break

    log_info(f"Test Episode {episode + 1} finished. Total reward - Seeker: {total_reward_seeker}, Hider: {total_reward_hider}")

imageio.mimsave(output_video_path, frames, fps=10)
log_info("Saving video...")
imageio.mimsave(output_gif_path, frames, duration=0.5)
log_info("Saving GIF...")
plt.close()
log_info("Testing complete. Video and GIF saved.")

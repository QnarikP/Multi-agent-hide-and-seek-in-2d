import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
from env.hide_and_seek_env import HideAndSeekEnv
from training.rl_agent import DQNAgent
from utils.visualization import render_environment
from utils.logger import log_info

# Define test parameters
num_test_episodes = 50  # Number of episodes to visualize
max_steps_per_episode = 50  # Max steps per episode for visualization
output_video_path = "test_video.mp4"  # Output video file
output_gif_path = "test_animation.gif"  # Output GIF file

# Initialize the environment
env = HideAndSeekEnv()
state_dim = 3  # State: (x, y, direction)
# Seeker uses 4 actions; hider uses 7 (0-3 movement, 4-6 door actions)
seeker_action_dim = 4
hider_action_dim = 7

# Load trained models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_info(f"Testing on device: {device}")

seeker_agent = DQNAgent(state_dim, seeker_action_dim, device=device)
hider_agent = DQNAgent(state_dim, hider_action_dim, device=device)

# Load pre-trained models (adjust paths as needed)
seeker_agent.load_model(r'C:\Users\User\Projects\NPUA\multi_agent_hide_and_seek\training\models\seeker_dqn_model.pth')
hider_agent.load_model(r'C:\Users\User\Projects\NPUA\multi_agent_hide_and_seek\training\models\hider_dqn_model.pth')

seeker_agent.q_network.to(device)
hider_agent.q_network.to(device)

# Initialize figure for plotting
fig, ax = plt.subplots(figsize=(8, 8))

# Store frames for video/GIF
frames = []

# Run test episode(s)
for episode in range(num_test_episodes):
    state = env.reset()
    # Get hider state; for seeker, initial state is dummy (-1,-1,-1)
    seeker_state = torch.tensor(state['seeker']['state'], dtype=torch.float32).to(device)
    hider_state = torch.tensor(state['hider']['state'], dtype=torch.float32).to(device)

    total_reward_seeker = 0
    total_reward_hider = 0

    log_info(f"Test Episode {episode + 1} started.")

    for step in range(max_steps_per_episode):
        # For the seeker: if inactive (dummy state), choose default action 0.
        if env.seeker_active:
            action_seeker = seeker_agent.select_action(seeker_state.cpu().numpy())
        else:
            action_seeker = 0  # Default movement action when inactive

        # Hider selects action (0-6)
        hider_action_int = hider_agent.select_action(hider_state.cpu().numpy())
        # Map door actions (>=4) to strings
        if hider_action_int < 4:
            action_hider = hider_action_int
        else:
            action_hider = {4: "toggle_door", 5: "lock", 6: "unlock"}[hider_action_int]
        actions = {"seeker": action_seeker, "hider": action_hider}

        # Apply actions in the environment
        next_state, done, door_rewards = env.step(actions)

        # Capture the current environment as a frame
        render_environment(ax, env)
        frame = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        width, height = fig.canvas.get_width_height()
        frame = frame.reshape((height, width, 4))
        frame = frame[..., 1::-1]  # Convert ARGB to RGB
        frames.append(frame)

        # Get next states (for hider always; for seeker if active)
        next_seeker_state = torch.tensor(next_state['seeker']['state'], dtype=torch.float32).to(device)
        next_hider_state = torch.tensor(next_state['hider']['state'], dtype=torch.float32).to(device)

        # Reward function: if seeker and hider are in the same cell, seeker +1, hider -1.
        if (seeker_state[0].item() == hider_state[0].item() and
            seeker_state[1].item() == hider_state[1].item()):
            reward_seeker = 1.0
            reward_hider = -1.0
        else:
            reward_seeker = 0.0
            reward_hider = 0.0

        total_reward_seeker += reward_seeker
        total_reward_hider += reward_hider

        seeker_state = next_seeker_state
        hider_state = next_hider_state

        if done:
            break

    log_info(f"Test Episode {episode + 1} finished. Total reward - Seeker: {total_reward_seeker}, Hider: {total_reward_hider}")

# Save video and GIF
log_info("Saving video...")
import imageio
imageio.mimsave(output_video_path, frames, fps=10)
log_info("Saving GIF...")
imageio.mimsave(output_gif_path, frames, duration=0.5)

plt.close()
log_info("Testing complete. Video and GIF saved.")

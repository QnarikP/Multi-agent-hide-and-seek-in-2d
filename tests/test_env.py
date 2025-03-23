import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
from env.hide_and_seek_env import HideAndSeekEnv
from training.rl_agent import DQNAgent
from utils.visualization import render_environment
from utils.logger import log_info

# Test parameters
num_test_episodes = 10  # Number of episodes to visualize
max_steps_per_episode = 50  # Max steps per episode for visualization
output_video_path = "test_video.mp4"  # Output video file
output_gif_path = "test_animation.gif"  # Output GIF file

# Initialize environment
env = HideAndSeekEnv()
state_dim = 3  # (x, y, direction)
agent_action_dim = 7  # Both agents have 7 actions

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_info(f"Testing on device: {device}")

# Initialize DQN agents for testing (without any further training)
seeker_agent = DQNAgent(state_dim, agent_action_dim, device=device)
hider_agent = DQNAgent(state_dim, agent_action_dim, device=device)

# Load pre-trained models (ensure these are the final weights you want to test)
seeker_agent.load_model(r'C:\Users\User\Projects\NPUA\multi_agent_hide_and_seek\training\models\seeker_dqn_model.pth')
hider_agent.load_model(r'C:\Users\User\Projects\NPUA\multi_agent_hide_and_seek\training\models\hider_dqn_model.pth')

# Move networks to device
seeker_agent.q_network.to(device)
hider_agent.q_network.to(device)

# Do not update target networks or call train_step() in test mode.
# In test mode, we use a greedy policy (i.e. ε = 0) and do not perform any learning.

fig, ax = plt.subplots(figsize=(8, 8))
frames = []

for episode in range(num_test_episodes):
    state = env.reset()
    # Convert states; note: seeker state may be dummy until activated.
    seeker_state = torch.tensor(state['seeker']['state'], dtype=torch.float32).to(device)
    hider_state = torch.tensor(state['hider']['state'], dtype=torch.float32).to(device)

    total_reward_seeker = 0
    total_reward_hider = 0

    log_info(f"Test Episode {episode + 1} started.")

    for step in range(max_steps_per_episode):
        # Select actions using a greedy policy (ε=0)
        if env.seeker_active:
            action_seeker = seeker_agent.select_action(seeker_state.cpu().numpy())
        else:
            action_seeker = 0  # Default action if seeker not yet active

        hider_action_int = hider_agent.select_action(hider_state.cpu().numpy())
        # Map hider actions: if < 4, it's a movement; otherwise, map to door action string.
        if hider_action_int < 4:
            action_hider = hider_action_int
        else:
            action_hider = {4: "toggle_door", 5: "lock", 6: "unlock"}[hider_action_int]

        actions = {"seeker": action_seeker, "hider": action_hider}

        next_state, done, door_rewards = env.step(actions)

        # Render the environment and capture an RGB frame.
        render_environment(ax, env)
        plt.pause(0.5)
        frame = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        width, height = fig.canvas.get_width_height()
        frame = frame.reshape((height, width, 4))
        # Convert ARGB to RGB by dropping the alpha channel.
        frame = frame[:, :, 1:4]
        frames.append(frame)

        next_seeker_state = torch.tensor(next_state['seeker']['state'], dtype=torch.float32).to(device)
        next_hider_state = torch.tensor(next_state['hider']['state'], dtype=torch.float32).to(device)

        # Use door_rewards as the step reward.
        reward_seeker = door_rewards.get("seeker", 0.0)
        reward_hider = door_rewards.get("hider", 0.0)
        total_reward_seeker += reward_seeker
        total_reward_hider += reward_hider

        seeker_state = next_seeker_state
        hider_state = next_hider_state

        if done:
            break

    log_info(f"Test Episode {episode + 1} finished. Total reward - Seeker: {total_reward_seeker}, Hider: {total_reward_hider}")

# Save frames as a video (RGB)
log_info("Saving video...")
imageio.mimsave(output_video_path, frames, fps=10)

# Save frames as a GIF
log_info("Saving GIF...")
imageio.mimsave(output_gif_path, frames, duration=0.5)

plt.close()
log_info("Testing complete. Video and GIF saved.")
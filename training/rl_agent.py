"""
training/rl_agent.py

This module implements a Deep Q-Network (DQN) agent for a grid-based, discrete action space.
We use independent DQN agents—one for each agent (seeker and hider).

Key components:
- QNetwork: A simple feed-forward neural network that maps state vectors to Q-values for each action.
- DQNAgent: Implements the DQN algorithm, including:
    - Epsilon-greedy action selection.
    - Experience replay: storing and sampling experiences.
    - Training step: computing loss between predicted Q-values and target Q-values (using the target network).
    - Target network update for stable learning.

Hyperparameters (e.g., learning rate, gamma, epsilon decay) are defined and can be adjusted.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from utils.logger import log_debug, log_info
import os

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Initialize the QNetwork.

        Args:
            state_dim (int): Dimension of the state vector.
            action_dim (int): Number of discrete actions.
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_dim)

    def forward(self, state):
        """
        Forward pass of the network.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Q-values for each action.
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_values = self.out(x)
        return q_values

class DQNAgent:
    """
    DQNAgent implements a Deep Q-Learning agent.

    It maintains:
      - A Q-network (for estimating Q-values).
      - A target network (for stable target Q-value estimation).
      - An experience replay buffer.
      - An epsilon-greedy policy for action selection.
    """
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, buffer_size=10000, batch_size=64,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, model_save_path='models', device=None):
        """
        Initialize the DQNAgent.

        Args:
            state_dim (int): Dimension of state representation.
            action_dim (int): Number of discrete actions.
            lr (float): Learning rate.
            gamma (float): Discount factor.
            buffer_size (int): Maximum size of the replay buffer.
            batch_size (int): Size of minibatches for training.
            epsilon_start (float): Initial exploration rate.
            epsilon_end (float): Minimum exploration rate.
            epsilon_decay (float): Decay factor for epsilon per training step.
            model_save_path (str): Directory path for saving models.
            device (torch.device): Device on which computations are performed.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.model_save_path = model_save_path
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Experience replay buffer: stores tuples (state, action, reward, next_state, done)
        self.replay_buffer = deque(maxlen=buffer_size)

        # Ensure the model save path exists
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

    def update_target_network(self):
        """
        Update the target network by copying parameters from the Q-network.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
        log_info("Target network updated.")

    def select_action(self, state):
        """
        Select an action using an epsilon-greedy policy.

        Args:
            state (np.array): Current state.

        Returns:
            int: Chosen action.
        """
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_dim)
            # log_debug(f"Selecting random action: {action}")
            return action
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # shape: (1, state_dim)
            q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
            # log_debug(f"Selecting greedy action: {action}")
            return action

    def store_experience(self, state, action, reward, next_state, done):
        """
        Store an experience tuple in the replay buffer.
        """
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self):
        """
        Perform a training step by sampling a minibatch from the replay buffer and updating the Q-network.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Instead of converting using np.array(), we stack the tensors
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Compute predicted Q-values for current states
        q_values = self.q_network(states).gather(1, actions)

        # Compute target Q-values using the target network
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # log_debug(f"Training step loss: {loss.item()}")

        # Decay epsilon after each training step
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save_model(self, filename="dqn_model.pth"):
        """
        Save the model weights to a file.

        Args:
            filename (str): Name of the file to save the model as.
        """
        save_path = os.path.join(self.model_save_path, filename)
        torch.save(self.q_network.state_dict(), save_path)
        log_info(f"Model saved at {save_path}")

    def load_model(self, filename="dqn_model.pth"):
        """
        Load the model weights from a file.

        Args:
            filename (str): Name of the file to load the model from.
        """
        load_path = os.path.join(self.model_save_path, filename)
        if os.path.exists(load_path):
            self.q_network.load_state_dict(torch.load(load_path))
            self.target_network.load_state_dict(torch.load(load_path))
            log_info(f"Model loaded from {load_path}")
        else:
            log_info(f"Model file {filename} not found. Starting from scratch.")

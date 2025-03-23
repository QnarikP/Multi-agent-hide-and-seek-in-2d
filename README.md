# Multi-Agent Hide and Seek

This repository contains a multi-agent hide-and-seek simulation where two agents (a *Seeker* and a *Hider*) interact in a 2D grid environment. The environment features a room with a door that can be opened, closed, locked, or unlocked, allowing for interesting emergent strategies between the agents.

The project implements:
- A custom environment (`HideAndSeekEnv`) using **OpenAI Gym**-like design patterns.
- DQN-based Reinforcement Learning agents (one for the *Seeker* and one for the *Hider*).
- Training scripts and utilities for logging and visualization.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Key Components](#key-components)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Training Process](#training-process)
6. [Testing and Visualization](#testing-and-visualization)
7. [Customization](#customization)

---

## Project Structure

```
multi_agent_hide_and_seek/
├── agents
│   ├── __init__.py
│   ├── hider.py
│   └── seeker.py
├── env
│   ├── __init__.py
│   └── hide_and_seek_env.py
├── models
│   └── ... (trained model files saved here)
├── tests
│   ├── test_animation.gif
│   ├── test_env.py
│   └── test_video.mp4
├── training
│   ├── __init__.py
│   ├── rl_agent.py
│   ├── train_rl.py
│   ├── training_metrics_invalid_moves.png
│   ├── training_metrics_penalties.png
│   └── training_metrics_rewards.png
├── utils
│   ├── __init__.py
│   ├── logger.py
│   └── visualization.py
├── .gitignore
├── example.mp4
├── README.md
└── requirements.txt
```

### Folders Overview
- **agents/**  
  Contains classes for the individual agents:
  - `hider.py`: Defines the Hider agent’s internal logic.
  - `seeker.py`: Defines the Seeker agent’s internal logic.
  
- **env/**  
  Contains the environment code:
  - `hide_and_seek_env.py`: The main environment class `HideAndSeekEnv`, implementing the grid, door mechanics, and agent interactions.

- **models/**  
  Stores trained model weights (e.g., `.pth` files) for the DQN agents.

- **tests/**  
  Contains test scripts and example media:
  - `test_env.py`: A quick script to run or visualize the environment.
  - `test_video.mp4` / `test_animation.gif`: Example outputs demonstrating the environment or agent behavior.

- **training/**  
  Holds training logic and RL agent code:
  - `rl_agent.py`: Contains the `DQNAgent` class and neural network architecture.
  - `train_rl.py`: Main script to train the *Seeker* and *Hider* agents.
  - Various PNG files showing reward, penalty, and invalid move metrics over training.

- **utils/**  
  Utility functions:
  - `logger.py`: Simple logging utility.
  - `visualization.py`: Visualization functions for metrics and environment rendering (some of which may have been moved into the environment).

- **example.mp4**  
  A short demonstration of the environment or trained agents.

- **requirements.txt**  
  Lists the Python dependencies required to run the project.

---

## Key Components

1. **HideAndSeekEnv**  
   - A 2D grid environment with a room, walls, and a door.
   - Two agents: *Seeker* (tries to find the Hider) and *Hider* (tries to hide inside the room).
   - The door can be toggled, locked, or unlocked by either agent.

2. **DQN Agents**  
   - Each agent uses a **Deep Q-Network** (DQN) with independent replay buffers and target networks.
   - The agents have discrete actions: move up/down/left/right, or perform door-related actions (toggle, lock, unlock).

3. **Reward Structure**  
   - *Seeker* gains +1 if it sees the Hider; Hider gets -1 in that scenario.
   - *Hider* gains +1 if it is inside the room.
   - (Optional) Additional penalties or invalid move counts can be logged.

4. **Training**  
   - Experience replay and target network updates for stable DQN training.
   - Epsilon-greedy policy for exploration.

5. **Visualization**  
   - The environment can be rendered using `matplotlib` to show the grid, agents, and their fields of view.
   - Training metrics (rewards, penalties, invalid moves) can be plotted and saved as PNG files.

---

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/QnarikP/Multi-agent-hide-and-seek-in-2d
   cd Multi-agent-hide-and-seek-in-2d
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   This will install packages like `numpy`, `torch`, `matplotlib`, `gym`, `imageio`, etc.

3. **Optional**: Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

---

## Usage

### Quick Start

- **Train** the DQN agents:
  ```bash
  python training/train_rl.py
  ```
  This will run for a specified number of episodes, logging progress and saving trained model weights into the `models/` folder.

- **Test** the environment (no training, just random or test policy):
  ```bash
  python tests/test_env.py
  ```
  You should see a GUI window (matplotlib) rendering the environment step-by-step.

### Environment Behavior
- The environment runs in episodes up to `max_steps_per_episode`.
- The *Hider* is spawned immediately. The *Seeker* typically spawns after some delay (e.g., 10 steps) to let the Hider move.
- The *Hider* tries to get inside the room. The *Seeker* tries to see the Hider.

---

## Training Process

- **train_rl.py**:  
  1. Initializes the environment and agents.  
  2. Loops through episodes:
     - Resets the environment.
     - Runs a maximum of `max_steps_per_episode`.
     - Agents pick actions via epsilon-greedy.
     - Receives rewards and next states from the environment.
     - Stores transitions in replay buffers and performs DQN updates.
  3. Periodically updates the target networks.
  4. Optionally renders the environment at certain steps.
  5. Logs metrics (rewards, invalid moves, penalties) and saves them.

- **rl_agent.py**:  
  1. `QNetwork`: A simple feed-forward neural network for Q-value approximation.  
  2. `DQNAgent`: Manages the Q-network, target network, replay buffer, and epsilon-greedy policy.

---

## Testing and Visualization

- **Environment Rendering**:  
  By default, `train_rl.py` may render the environment every few steps. You can comment/uncomment lines to adjust the frequency of rendering.

- **Metrics Plots**:  
  After training, various PNG plots (like `training_metrics_rewards.png`) will appear in the `training/` folder. These plots help visualize agent performance over time.

- **Animations and Videos**:  
  The `tests/test_animation.gif` or `tests/test_video.mp4` show example runs of the environment.  
  You can also record episodes by capturing frames in your own scripts (e.g., using `imageio.mimsave`).

---

## Customization

- **Adjust Hyperparameters** in `train_rl.py`:
  - `num_episodes`, `max_steps_per_episode`, `target_update_frequency`, etc.
- **Change DQN architecture** in `rl_agent.py` (e.g., more layers, different activations).
- **Modify Reward Function** in `hide_and_seek_env.py` to encourage or discourage certain behaviors.
- **Room Layout** can be customized by changing the size or location of the room, as well as door placement.

---

**Enjoy experimenting with multi-agent hide and seek!** If you encounter any issues or have ideas for improvement, feel free to open an issue or submit a pull request.
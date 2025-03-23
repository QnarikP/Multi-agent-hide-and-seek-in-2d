"""
utils/visualization.py

This module provides helper functions for rendering and plotting the grid, room, agents, and debug overlays.
It uses matplotlib to:
  - Draw grid lines.
  - Plot the room (interior, wall cells, and door with its current state).
  - Render agents along with their fields of view.
  - Overlay additional debugging information.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_grid(ax, grid_size):
    """
    Draw grid lines on the provided matplotlib axis.

    Args:
        ax (matplotlib.axes.Axes): The axis to draw on.
        grid_size (int): Size of the grid (assumes square grid).
    """
    for x in range(grid_size + 1):
        ax.plot([x, x], [0, grid_size], color='lightgray', lw=0.5)
    for y in range(grid_size + 1):
        ax.plot([0, grid_size], [y, y], color='lightgray', lw=0.5)


def draw_room(ax, room):
    """
    Draw the room interior, walls, and door cell.

    Args:
        ax (matplotlib.axes.Axes): The axis to draw on.
        room: A Room object (from env.room) with methods get_all_cells() and attributes wall_cells and door.
    """
    # Draw room interior
    room_cells = room.get_all_cells()
    for cell in room_cells:
        rect = patches.Rectangle(cell, 1, 1, facecolor='lightyellow', edgecolor='black', lw=0.5)
        ax.add_patch(rect)
    for wall in room.wall_cells:
        rect = patches.Rectangle(wall, 1, 1, facecolor='gray', edgecolor='black')
        ax.add_patch(rect)
    door_color = 'green' if room.door.is_open and not room.door.is_locked else 'red'
    door_rect = patches.Rectangle(room.door.position, 1, 1, facecolor=door_color, edgecolor='black')
    ax.add_patch(door_rect)
    ax.text(room.door.position[0] + 0.2, room.door.position[1] + 0.5, "Door", color='white', fontsize=8)



def draw_agent(ax, agent, moves, label, color):
    """
    Draw an agent on the grid.

    Args:
        ax (matplotlib.axes.Axes): The axis to draw on.
        agent: An agent object with a get_state() method returning (x, y, direction).
        moves (dict): Dictionary mapping direction indices to (dx, dy) deltas.
        label (str): Label to display near the agent.
        color (str): Color used for drawing the agent.
    """
    x, y, d = agent.get_state()
    agent_circle = patches.Circle((x + 0.5, y + 0.5), 0.3, color=color)
    ax.add_patch(agent_circle)
    dx, dy = moves.get(d, (0, 0))
    ax.arrow(x + 0.5, y + 0.5, dx * 0.3, dy * 0.3, head_width=0.1, head_length=0.1, fc=color, ec=color)
    ax.text(x + 0.1, y + 0.2, label, color='black', fontsize=8)


def draw_agent_fov(ax, agent, fov_offsets, color='cyan', alpha=0.3):
    """
    Draw the field of view (FOV) overlay for an agent.

    Args:
        ax (matplotlib.axes.Axes): The axis to draw on.
        agent: An agent object with get_state() returning (x, y, direction).
        fov_offsets (dict): A dictionary mapping directions (0,1,2,3) to lists of (dx, dy) offsets for FOV cells.
        color (str): The color for the FOV overlay.
        alpha (float): The transparency level of the FOV overlay.
    """
    x, y, d = agent.get_state()
    offsets = fov_offsets.get(d, [])
    for dx, dy in offsets:
        cell = (x + dx, y + dy)
        rect = patches.Rectangle(cell, 1, 1, facecolor=color, edgecolor=color, alpha=alpha)
        ax.add_patch(rect)


def get_default_fov_offsets():
    """
    Returns a default dictionary of field-of-view (FOV) offsets for each direction.

    The offsets approximate a wedge-shaped FOV.

    Returns:
        dict: Mapping from direction to list of (dx, dy) tuples.
    """
    return {
        0: [(-1, -1), (0, -1), (1, -1), (-1, -2), (0, -2), (1, -2)],
        1: [(1, -1), (1, 0), (1, 1), (2, -1), (2, 0), (2, 1)],
        2: [(-1, 1), (0, 1), (1, 1), (-1, 2), (0, 2), (1, 2)],
        3: [(-1, -1), (-1, 0), (-1, 1), (-2, -1), (-2, 0), (-2, 1)]
    }


def render_environment(ax, env):
    """
    Render the environment on the provided axis.
    If the seeker is inactive, its agent and visible overlay are not drawn.
    Visible cells overlay:
      - Light red: only visible to seeker.
      - Light blue: only visible to hider.
      - Light purple: visible to both.
    """
    ax.clear()
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title("Hide and Seek Environment")

    draw_grid(ax, env.grid_size)
    draw_room(ax, env.room)

    # Draw hider always.
    draw_agent(ax, env.hider, env.moves, "Hider", "blue")
    visible_hider = set(env.compute_visible_cells(env.hider.get_state()))

    # Only draw seeker if active.
    if env.seeker_active:
        draw_agent(ax, env.seeker, env.moves, "Seeker", "red")
        visible_seeker = set(env.compute_visible_cells(env.seeker.get_state()))
    else:
        visible_seeker = set()

    # Determine cells visible exclusively by one agent or by both.
    only_seeker = visible_seeker - visible_hider
    only_hider = visible_hider - visible_seeker
    both = visible_seeker & visible_hider

    for cell in only_seeker:
        rect = patches.Rectangle(cell, 1, 1, facecolor='lightcoral', alpha=0.4)
        ax.add_patch(rect)
    for cell in only_hider:
        rect = patches.Rectangle(cell, 1, 1, facecolor='lightblue', alpha=0.4)
        ax.add_patch(rect)
    for cell in both:
        rect = patches.Rectangle(cell, 1, 1, facecolor='plum', alpha=0.4)
        ax.add_patch(rect)

    # Draw FOV overlays (only draw seeker FOV if active)
    fov_offsets = get_default_fov_offsets()
    if env.seeker_active:
        draw_agent_fov(ax, env.seeker, fov_offsets, color='pink', alpha=0.3)
    draw_agent_fov(ax, env.hider, fov_offsets, color='lightblue', alpha=0.3)
    ax.figure.canvas.draw()


def plot_rewards(rewards_seeker, rewards_hider, filename=None):
    episodes = range(1, len(rewards_seeker) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards_seeker, label='Seeker Rewards')
    plt.plot(episodes, rewards_hider, label='Hider Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards per Episode')
    plt.legend()
    if filename:
        plt.savefig(filename)
    plt.show()

def plot_invalid_moves(invalid_moves_seeker, invalid_moves_hider, filename=None):
    episodes = range(1, len(invalid_moves_seeker) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, invalid_moves_seeker, label='Seeker Invalid Moves')
    plt.plot(episodes, invalid_moves_hider, label='Hider Invalid Moves')
    plt.xlabel('Episode')
    plt.ylabel('Count of Invalid Moves')
    plt.title('Invalid Moves per Episode')
    plt.legend()
    if filename:
        plt.savefig(filename)
    plt.show()

def plot_penalties(penalties_seeker, penalties_hider, filename=None):
    episodes = range(1, len(penalties_seeker) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, penalties_seeker, label='Seeker Penalties')
    plt.plot(episodes, penalties_hider, label='Hider Penalties')
    plt.xlabel('Episode')
    plt.ylabel('Penalty Count')
    plt.title('Penalties per Episode')
    plt.legend()
    if filename:
        plt.savefig(filename)
    plt.show()

def visualize_all_metrics(metrics, filename_prefix=None):
    """
    Visualize all provided metrics.

    Parameters:
      metrics: dict with keys 'rewards_seeker', 'rewards_hider',
               'invalid_moves_seeker', 'invalid_moves_hider',
               'penalties_seeker', 'penalties_hider'
      filename_prefix: optional prefix for saving the plots to files.
    """
    plot_rewards(
        metrics['rewards_seeker'], metrics['rewards_hider'],
        filename=f"{filename_prefix}_rewards.png" if filename_prefix else None
    )
    plot_invalid_moves(
        metrics['invalid_moves_seeker'], metrics['invalid_moves_hider'],
        filename=f"{filename_prefix}_invalid_moves.png" if filename_prefix else None
    )
    plot_penalties(
        metrics['penalties_seeker'], metrics['penalties_hider'],
        filename=f"{filename_prefix}_penalties.png" if filename_prefix else None
    )
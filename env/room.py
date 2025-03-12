"""
env/room.py

This module encapsulates the room layout and door properties for the hide and seek environment.
It includes two classes:
    - Door: Manages the door's position, state (open/closed), and lock status.
    - Room: Defines the room boundaries, computes wall cells (excluding the door cell),
            and provides helper methods to check if a given cell is part of the room, a wall, or the door.

The room is defined as a 4x4 area located at the bottom right corner of a 10x10 grid.
The door is located on the left wall, at the center cell.
"""

from utils.logger import log_debug

class Door:
    """
    Represents a door with a position and state.
    """
    def __init__(self, position):
        self.position = position
        self.is_open = True
        self.is_locked = False
        # log_debug(f"Door created at {self.position}. Initial state: open={self.is_open}, locked={self.is_locked}")

    def toggle(self):
        """
        Toggle the door's state between open and closed.
        """
        self.is_open = not self.is_open
        # log_debug(f"Door at {self.position} toggled to {'open' if self.is_open else 'closed'}.")

    def lock(self):
        """
        Lock the door.
        """
        if not self.is_locked:
            self.is_locked = True
            # log_debug(f"Door at {self.position} locked.")
        # else:
            # log_debug(f"Door at {self.position} is already locked.")

    def unlock(self):
        """
        Unlock the door.
        """
        if self.is_locked:
            self.is_locked = False
            # log_debug(f"Door at {self.position} unlocked.")
        # else:
            # log_debug(f"Door at {self.position} is already unlocked.")


class Room:
    """
    Represents a room within the grid.
    """
    def __init__(self, top_left, width, height, door_side="left"):
        """
        Initialize the room.

        Args:
            top_left (tuple): (x, y) coordinate for the top-left corner.
            width (int): Width of the room.
            height (int): Height of the room.
            door_side (str): Side of the room where the door is located ("left", "right", "top", "bottom").
        """
        self.top_left = top_left
        self.width = width
        self.height = height

        # Determine door position based on door_side
        if door_side == "left":
            door_x = top_left[0]
            door_y = top_left[1] + height // 2
        elif door_side == "right":
            door_x = top_left[0] + width - 1
            door_y = top_left[1] + height // 2
        elif door_side == "top":
            door_x = top_left[0] + width // 2
            door_y = top_left[1]
        elif door_side == "bottom":
            door_x = top_left[0] + width // 2
            door_y = top_left[1] + height - 1
        else:
            raise ValueError("Invalid door_side. Choose from 'left', 'right', 'top', 'bottom'.")

        self.door = Door((door_x, door_y))

        # Compute wall cells: all cells on the perimeter except the door cell.
        self.wall_cells = set()
        for x in range(top_left[0], top_left[0] + width):
            self.wall_cells.add((x, top_left[1]))                   # Top wall
            self.wall_cells.add((x, top_left[1] + height - 1))        # Bottom wall
        for y in range(top_left[1], top_left[1] + height):
            self.wall_cells.add((top_left[0], y))                    # Left wall
            self.wall_cells.add((top_left[0] + width - 1, y))          # Right wall

        # Remove the door cell from the wall cells if present
        if self.door.position in self.wall_cells:
            self.wall_cells.remove(self.door.position)

        # log_debug(f"Room created at {self.top_left} with size {width}x{height}.")
        # log_debug(f"Room wall cells: {self.wall_cells}")
        # log_debug(f"Door position: {self.door.position}")

    def is_inside(self, cell):
        """
        Check if a cell is inside the room boundaries.
        """
        x, y = cell
        return (self.top_left[0] <= x < self.top_left[0] + self.width and
                self.top_left[1] <= y < self.top_left[1] + self.height)

    def is_wall(self, cell):
        """
        Check if a cell is a wall cell of the room.
        """
        return cell in self.wall_cells

    def is_door(self, cell):
        """
        Check if a cell is the door cell.
        """
        return cell == self.door.position

    def blocks_vision(self, cell):
        """
        Determine if a cell blocks vision.
        A cell blocks vision if it is a wall cell, or if it is the door cell and the door is closed.
        """
        if cell in self.wall_cells:
            return True
        if self.is_door(cell) and not self.door.is_open:
            return True
        return False

    def get_all_cells(self):
        """
        Get all cells that belong to the room (both interior and walls).
        """
        cells = set()
        for x in range(self.top_left[0], self.top_left[0] + self.width):
            for y in range(self.top_left[1], self.top_left[1] + self.height):
                cells.add((x, y))
        return cells


if __name__ == "__main__":
    # Visualization for the Room module
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    grid_size = 10
    # Create a 4x4 room at the bottom right corner (top_left=(6,6) because 10-4=6)
    room = Room(top_left=(6, 6), width=4, height=4, door_side="left")

    fig, ax = plt.subplots()
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title("Room Visualization")

    # Draw grid lines for clarity
    for x in range(grid_size + 1):
        ax.plot([x, x], [0, grid_size], color='lightgray', lw=0.5)
    for y in range(grid_size + 1):
        ax.plot([0, grid_size], [y, y], color='lightgray', lw=0.5)

    # Draw room cells
    room_cells = room.get_all_cells()
    for cell in room_cells:
        rect = patches.Rectangle(cell, 1, 1, facecolor='lightyellow', edgecolor='black', lw=0.5)
        ax.add_patch(rect)

    # Highlight wall cells in gray
    for wall in room.wall_cells:
        rect = patches.Rectangle(wall, 1, 1, facecolor='gray', edgecolor='black')
        ax.add_patch(rect)

    # Highlight the door cell with color based on its state
    door_color = 'green' if room.door.is_open and not room.door.is_locked else 'red'
    door_rect = patches.Rectangle(room.door.position, 1, 1, facecolor=door_color, edgecolor='black')
    ax.add_patch(door_rect)
    ax.text(room.door.position[0] + 0.2, room.door.position[1] + 0.5, "Door", color='white', fontsize=8)

    plt.show()

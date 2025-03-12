def compute_visible_cells(self, state, max_distance=10):
    """
    Compute the visible cells for an agent using a row-based field of view.
    The agent sees cells in a growing triangular pattern, ensuring visibility
    does not pass through walls or include interior room cells.

    Args:
        state (tuple): (x, y, d) representing the agent's state.
        max_distance (int): Maximum number of rows the agent can see.

    Returns:
        list: A list of (x, y) tuples that are visible.
    """
    x, y, d = state
    visible = set()

    # Iterate through rows increasing in width
    for i in range(max_distance):
        row_width = 2 * i + 1  # Number of cells in the current row
        start_x = x - i if d in {0, 2} else x  # Shift left/right for up/down
        start_y = y - i if d in {1, 3} else y  # Shift up/down for left/right

        for j in range(row_width):
            cell_x = start_x + (j if d in {0, 2} else 0)  # Horizontal spread for up/down
            cell_y = start_y + (j if d in {1, 3} else 0)  # Vertical spread for left/right

            cell = (cell_x, cell_y)

            # Ensure cell is within bounds
            if not (0 <= cell_x < self.grid_size and 0 <= cell_y < self.grid_size):
                continue

            # Stop if encountering a blocking wall
            if self.room.blocks_vision(cell):
                break

            visible.add(cell)

    return list(visible)
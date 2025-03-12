"""
agents/seeker.py

This module defines the Seeker agent for the multi-agent hide and seek project.
The seeker is responsible for finding the hider.
It stores the agent's state (position and direction) and provides methods to update the state
and process actions. Debug output is standardized using the custom logger.
"""

from utils.logger import log_debug, log_info

class Seeker:
    def __init__(self, x, y, direction):
        """
        Initialize the seeker with a starting position and direction.
        """
        self.x = x
        self.y = y
        self.direction = direction
        # log_debug(f"Seeker created at ({self.x}, {self.y}) facing direction {self.direction}")

    def update_state(self, x=None, y=None, direction=None):
        """
        Update the seeker's state.
        """
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if direction is not None:
            self.direction = direction
        # log_debug(f"Seeker updated state: ({self.x}, {self.y}), direction: {self.direction}")

    def process_action(self, action, moves):
        """
        Process a movement action.
        For movement actions (0-3), update the position based on the provided movement mapping.
        """
        if isinstance(action, int) and action in moves:
            dx, dy = moves[action]
            new_x = self.x + dx
            new_y = self.y + dy
            self.update_state(x=new_x, y=new_y, direction=action)
            # log_debug(f"Seeker processed movement action {action}: moved to ({new_x}, {new_y})")
            return (new_x, new_y, action)
        else:
            # log_debug(f"Seeker received non-movement action: {action}")
            return (self.x, self.y, self.direction)

    def get_state(self):
        """
        Return the current state as a tuple (x, y, direction).
        """
        return (self.x, self.y, self.direction)


if __name__ == "__main__":
    seeker = Seeker(2, 3, 1)
    log_info(f"Initial Seeker state: {seeker.get_state()}")
    seeker.process_action(2, {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)})
    log_info(f"Updated Seeker state: {seeker.get_state()}")

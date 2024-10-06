import numpy as np
from gym import spaces
import copy
import random

class PlayerDiscrete:
    def __init__(self, initial_position, name):
        """
        Initializes a discrete player object.

        Args:
            initial_position (tuple): The initial position of the player.
            name (str): The name of the player.
        """
        self.position = np.array(initial_position, dtype=np.float32)
        initial_direction = (0, 1)
        self.direction = np.array(initial_direction, dtype=np.float32)
        self.angle_change = np.pi / 4
        self.name = name

    def move(self, action, environment):
        """
        Moves the player based on the given action.

        Args:
            action (tuple): The action to be performed by the player.
            environment (Environment): The environment in which the player is moving.
        """
        # Update the direction based on the turning action
        if action[1] == 0:
            change_factor = 0
        elif action[1] == 1:
            change_factor = 1
        else:
            change_factor = -1

        angle_change = change_factor * self.angle_change

        current_angle = np.arctan2(self.direction[1], self.direction[0])

        new_angle = current_angle + angle_change
        
        self.direction = np.array([np.cos(new_angle), np.sin(new_angle)])
        self.direction /= np.linalg.norm(self.direction)
        
        if action[0] == 0:
            step_size = 0
        elif action[0] == 1:
            step_size = 1
        else:
            step_size = -1

        new_position = self.position + self.direction * step_size

        if new_position[0] < 0 or new_position[0] >= environment.width:
            new_position[0] = np.clip(new_position[0], 0, environment.width - 1)
        if new_position[1] < 0 or new_position[1] >= environment.height:
            new_position[1] = np.clip(new_position[1], 0, environment.height - 1)

        # Update position if no collision
        self.position = new_position

    def get_position(self):
        """
        Returns the current position of the player.

        Returns:
            numpy.ndarray: The current position of the player.
        """
        return self.position

    def get_direction(self):
        """
        Returns the current direction of the player.

        Returns:
            numpy.ndarray: The current direction of the player.
        """
        return self.direction

class Target:
    def __init__(self, position, timesteps_left=20):
        """
        Initializes a Target object.

        Args:
            position (list or tuple): The initial position of the target.
            timesteps_left (int, optional): The number of timesteps the target will remain active. Defaults to 20.
        """
        self.position = np.array(position, dtype=np.float32)
        self.timesteps_left = timesteps_left

    def update(self):
        """
        Updates the target's position based on the current direction.
        """
        self.timesteps_left = self.timesteps_left - 1

    def get_position(self):
        """
        Returns the current position of the target.

        Returns:
            numpy.ndarray: The current position of the target.
        """
        return self.position

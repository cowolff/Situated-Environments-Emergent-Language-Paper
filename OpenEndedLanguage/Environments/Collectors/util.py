import random
import math
import numpy as np

def position_relative(position, goal_position):
    
    x = goal_position[0] - position[0]
    y = goal_position[1] - position[1]
    
    return np.array([x, y])

def get_direction_relative(direction, team=1.0):
    # direction is a tuple (dx, dy)
    # team is either -1 or 1

    if team not in [-1, 1]:
        raise ValueError("team must be either -1 or 1")

    return (direction[0] * team, direction[1] * team)

def generate_random_coordinates(x_range, y_range):
    coord1 = (random.uniform(x_range[0], x_range[1]), random.uniform(y_range[0], y_range[1]))
    return np.array(coord1, dtype=np.float32)

def generate_random_coordinates_int(x_range, y_range):
    coord1 = (random.randint(x_range[0], x_range[1]), random.randint(y_range[0], y_range[1]))
    return np.array(coord1, dtype=np.float32)
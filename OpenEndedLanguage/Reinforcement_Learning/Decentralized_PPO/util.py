import numpy as np
import torch

def flatten_list(data):
    flat_list = []
    for item in data:
        flat_list.extend(item.values())
    return flat_list

def reverse_flatten_list_with_agent_list(flat_list, agent_names):
    reversed_data = []
    for i in range(0, len(flat_list), len(agent_names)):
        # Creating a dictionary for each set of lists corresponding to the agent names
        entry = {agent_names[j]: flat_list[i + j] for j in range(len(agent_names))}
        reversed_data.append(entry)
    return reversed_data

def normalize_batch_observations(batch_observations, min_values, max_values):
    """ Normalize a batch of observations to the range [-1, 1] """
    # Ensure min_values and max_values are arrays and have the same shape as observations
    min_values = np.array(min_values, dtype=np.float32)
    max_values = np.array(max_values, dtype=np.float32)

    # Calculate the range, avoiding division by zero
    range_values = np.where((max_values > min_values), max_values - min_values, 1)

    # Normalize each observation in the batch to the range [0, 1]
    normalized_batch_0_1 = (batch_observations - min_values) / range_values

    # Scale the normalized values from [0, 1] to [-1, 1]
    normalized_batch = normalized_batch_0_1 * 2 - 1
    
    return normalized_batch

def normalize_batch_torch(batch_observations, min_values, max_values):
    """ Normalize a batch of observations to the range [-1, 1] """
    # Ensure min_values and max_values are arrays and have the same shape as observations
    min_values = torch.tensor(min_values, dtype=torch.float32)
    max_values = torch.tensor(max_values, dtype=torch.float32)

    # Calculate the range, avoiding division by zero
    range_values = torch.where((max_values > min_values), max_values - min_values, 1)

    # Normalize each observation in the batch to the range [0, 1]
    normalized_batch_0_1 = (batch_observations - min_values) / range_values

    # Scale the normalized values from [0, 1] to [-1, 1]
    normalized_batch = normalized_batch_0_1 * 2 - 1
    
    return normalized_batch

def concatenate_agent_observations(agent_dict):
    """
    Concatenates arrays from a dictionary of agents into a single numpy array.

    Parameters:
    agent_dict (dict): A dictionary where keys are agent names and values are numpy arrays of shape (12, 4).

    Returns:
    numpy.ndarray: A numpy array.
    """
    arrays = list(agent_dict.values())  # Extract arrays from the dictionary
    concatenated_array = np.concatenate(arrays, axis=0)  # Concatenate along the first axis
    return concatenated_array

def concatenate_torch(agent_dict):
    """
    Concatenates arrays from a dictionary of agents into a single numpy array.

    Parameters:
    agent_dict (dict): A dictionary where keys are agent names and values are numpy arrays of shape (12, 4).

    Returns:
    numpy.ndarray: A numpy array.
    """
    arrays = list(agent_dict.values())  # Extract arrays from the dictionary
    concatenated_array = torch.concatenate(arrays, axis=0)  # Concatenate along the first axis
    return concatenated_array

def split_agent_actions(concatenated_array, original_keys):
    """
    Splits a concatenated numpy array back into a dictionary of smaller arrays.

    Parameters:
    concatenated_array (numpy.ndarray): The concatenated array of shape (N, 4).
    original_keys (list): The list of original dictionary keys.

    Returns:
    dict: A dictionary where keys are the original keys and values are numpy arrays of shape (M, 4).
    """
    num_keys = len(original_keys)
    if concatenated_array.shape[0] % num_keys != 0:
        raise ValueError("The concatenated array cannot be evenly split into the number of original keys.")

    split_arrays = np.array_split(concatenated_array, num_keys)
    
    return dict(zip(original_keys, split_arrays))

def split_agent_torch(concatenated_array, original_keys):
    """
    Splits a concatenated numpy array back into a dictionary of smaller arrays.

    Parameters:
    concatenated_array (numpy.ndarray): The concatenated array of shape (N, 4).
    original_keys (list): The list of original dictionary keys.

    Returns:
    dict: A dictionary where keys are the original keys and values are numpy arrays of shape (M, 4).
    """
    num_keys = len(original_keys)
    if concatenated_array.shape[0] % num_keys != 0:
        raise ValueError("The concatenated array cannot be evenly split into the number of original keys.")

    split_arrays = torch.tensor_split(concatenated_array, num_keys)
    
    return dict(zip(original_keys, split_arrays))

def split_network_output_by_team(network_output, original_keys, teams):
    split_outputs = split_agent_torch(network_output, original_keys)
    team_outputs = {}
    for i, team in enumerate(teams):
        team_outputs[i] = torch.cat([split_outputs[agent] for agent in teams[team]], dim=0)
    return team_outputs

def split_env_output_by_team(env_output, original_keys, teams):
    team_outputs = {}
    for i, team in enumerate(teams):
        team_outputs[team] = {agent: torch.Tensor(env_output[agent]) for agent in original_keys}
    return team_outputs
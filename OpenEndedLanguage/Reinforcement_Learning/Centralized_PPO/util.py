import numpy as np

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
from pettingzoo.utils import ParallelEnv
from gym import spaces
import numpy as np
import time
import copy
import pygame
import sys
import random

try:
    from game import PlayerDiscrete, Target
    from util import position_relative, get_direction_relative, generate_random_coordinates_int
except ImportError:
    from ThesisPackage.Environments.collectors.game import PlayerDiscrete, Target
    from ThesisPackage.Environments.collectors.util import position_relative, get_direction_relative, generate_random_coordinates_int

class Collectors(ParallelEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, width=7, height=5, sequence_length=1, vocab_size=1, num_targets=3, timestep_countdown=20, spawn_frequency = 10, frequency_range=7, max_timesteps=1024):
        super().__init__()
        self.width = width
        self.height = height
        # randomly initialize the players positions
        player_pos = generate_random_coordinates_int((1, self.width - 1), (1, self.height - 1))
        self.players = {
            "player_1": PlayerDiscrete(copy.deepcopy(player_pos), "player_1"),
            "player_2": PlayerDiscrete(copy.deepcopy(player_pos), "player_2"),
        }
        self.agents = list(self.players.keys())
        self.possible_agents = self.agents[:]
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size

        self.spawn_frequency = spawn_frequency
        self.frequency_range = frequency_range
        self.next_spawn = spawn_frequency + random.randint(-frequency_range, frequency_range)

        self.num_targets = num_targets
        self.timestep_countdown = timestep_countdown
        self.targets = []
        
        self._action_space = {agent: spaces.MultiDiscrete([3, 3] + [self.vocab_size for _ in range(self.sequence_length)]) for agent in self.agents}

        self._observation_spaces = {agent: self.__get_observation_space(agent) for agent in self.agents}
        self._critic_observation_space = self.__get_observation_space("critic")
        self.max_timesteps = max_timesteps
        self.episode_rewards = 0
        self.utterances = {agent: [0 for _ in range(self.sequence_length * self.vocab_size)] for agent in self.agents}
        self.reset()



    def __get_observation_space_critic(self):
        """
        Returns the observation space for the critic in the collector environment.

        The observation space is a Box space defined by the concatenation of high and low arrays.
        The high array contains the maximum values for each observation dimension,
        while the low array contains the minimum values for each observation dimension.

        Returns:
            spaces.Box: The observation space for the critic.
        """
        observation_high = np.array([])
        observation_low = np.array([])

        for agent in self.agents:
            observation_high = np.concatenate((observation_high, np.array([self.width - 1, self.height - 1, 1, 1])))
            observation_low = np.concatenate((observation_low, np.array([-self.width, -self.height, -1, -1])))

            observation_high = np.concatenate((observation_high, np.array([1 for _ in range(self.sequence_length * self.vocab_size)])))
            observation_low = np.concatenate((observation_low, np.array([0 for _ in range(self.sequence_length * self.vocab_size)])))

        for i in range(self.num_targets):
            observation_high = np.concatenate((observation_high, np.array([self.width - 1, self.height - 1, self.timestep_countdown])))
            observation_low = np.concatenate((observation_low, np.array([-self.width, -self.height, 0])))

        return spaces.Box(low=observation_low, high=observation_high, dtype=np.float32)
    


    
    def __get_observation_space_player(self):
        """
        Returns the observation space for the player.

        The observation space includes information about the player's position, targets, and language.

        Returns:
            spaces.Box: The observation space for the player.
        """
        observation_high = np.array([self.width - 1, self.height - 1, 1, 1])
        observation_low = np.array([-self.width, -self.height, -1, -1])

        for i in range(self.num_targets):
            observation_high = np.concatenate((observation_high, np.array([self.width, self.height, self.timestep_countdown])))
            observation_low = np.concatenate((observation_low, np.array([-self.width, -self.height, 0])))

        language_high = np.array([1 for _ in range(self.sequence_length * self.vocab_size)])
        language_low = np.array([0 for _ in range(self.sequence_length * self.vocab_size)])

        observation_high = np.concatenate((observation_high, language_high))
        observation_low = np.concatenate((observation_low, language_low))

        return spaces.Box(low=observation_low, high=observation_high, dtype=np.float32)
    


    def __get_observation_space(self, agent):
        if agent == "critic":
            return self.__get_observation_space_critic()
        else:
            return self.__get_observation_space_player()
        


    def reset(self, seed=1, options=None):
        """
        Resets the environment to its initial state.

        Parameters:
        - seed (int): The seed for random number generation. Default is 1.
        - options (dict): Additional options for resetting the environment. Default is None.

        Returns:
        - observation (dict): The initial observation for each agent.
        - infos (dict): Additional information about the environment state for each agent.
        """
        player_pos = generate_random_coordinates_int((1, self.width - 1), (1, self.height - 1))

        self.players['player_1'].position = copy.deepcopy(player_pos)
        self.players['player_1'].direction = np.array([0, 1], dtype=np.float32)
        self.players['player_2'].position = copy.deepcopy(player_pos)
        self.players['player_2'].direction = np.array([0, 1], dtype=np.float32)

        self.targets = []
        self.targets.append(Target(generate_random_coordinates_int((1, self.width - 1), (1, self.height - 1)), self.timestep_countdown))
        self.next_spawn = self.spawn_frequency + random.randint(-self.frequency_range, self.frequency_range)

        self.timestep = 0
        self.episode_rewards = 0
        self.utterances = {agent: [0 for _ in range(self.sequence_length * self.vocab_size)] for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return {agent: self.observe(agent) for agent in self.agents}, infos
    
    

    def step(self, actions):
        """
        Executes a single step in the environment.

        Args:
            actions (dict): A dictionary containing the actions for each agent.

        Returns:
            tuple: A tuple containing the observations, rewards, termination status, truncation status, and additional information for each agent.
        """
        rewards = {agent: 0 for agent in self.agents}
        terminated = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        self.timestep += 1

        # Move the players
        for agent, action in actions.items():
            self.players[agent].move(action[:2], self)

        self.next_spawn -= 1
        if len(self.targets) < self.num_targets:
            if self.next_spawn <= 0:
                self.targets.append(Target(generate_random_coordinates_int((1, self.width - 1), (1, self.height - 1)), self.timestep_countdown))
                self.next_spawn = self.spawn_frequency + random.randint(-self.frequency_range, self.frequency_range)

        length = len(self.targets)
        removes = []
        for target in range(length):
            self.targets[target].update()
            if self.targets[target].timesteps_left <= 0:
                terminated = {agent: True for agent in self.agents}
                rewards = {agent: -1 for agent in self.agents}
                break
            for agent in self.agents:
                # if distance smaller than 0.5
                if np.linalg.norm(self.players[agent].get_position() - self.targets[target].get_position()) <= 1.5:
                    removes.append(self.targets[target])
        removes = list(set(removes))

        for target in removes:
            self.targets.remove(target)
            rewards = {agent: 1 for agent in self.agents}
        
        if self.sequence_length > 0:
            for agent, action in actions.items():
                one_hot_utterance = np.eye(self.vocab_size)[actions[agent][2:]]
                one_hot_utterance = one_hot_utterance.flatten()
                self.utterances[agent] = one_hot_utterance
        
        self.episode_rewards += sum(rewards.values())

        if self.max_timesteps < self.timestep:
            truncated = {agent: True for agent in self.agents}
        else:
            truncated = {agent: False for agent in self.agents}

        return {agent: self.observe(agent) for agent in self.agents}, rewards, terminated, truncated, infos
    
    

    def observe(self, agent):
        """
        Generates an observation for the given agent.

        Parameters:
        agent (int): The index of the agent.

        Returns:
        np.ndarray: The observation array.
        """
        other_agents = [a for a in self.agents if a != agent][0]
        
        player = self.players[agent]

        agent_coord = position_relative(player.get_position(), (self.width // 2, self.height // 2))
        agent_direct = self.players[agent].get_direction()

        obs = np.concatenate((agent_coord, agent_direct), dtype=np.float32)
        
        for i in range(self.num_targets):
            if i < len(self.targets):
                target_data = np.concatenate((position_relative(self.targets[i].get_position(), player.get_position()), np.array([self.targets[i].timesteps_left])))
            else:
                target_data = np.array([0, 0, 0])
            obs = np.concatenate((obs, target_data), dtype=np.float32)

        other_utterance = self.utterances[other_agents]
        obs = np.concatenate((obs, other_utterance), dtype=np.float32)
        
        return obs
    

    def state(self) -> np.ndarray:
        """
        Returns the state of the environment as a numpy array.

        Returns:
            np.ndarray: The state of the environment.
        """
        
        obs = np.array([])
        for player in self.players:
            player = self.players[player]

            agent_coord = position_relative(player.get_position(), (self.width // 2, self.height // 2))
            agent_direct = player.get_direction()

            utterance = np.array(self.utterances[player.name], dtype=np.float32)

            obs = np.concatenate((obs, agent_coord, agent_direct, utterance), dtype=np.float32)

        for i in range(self.num_targets):
            if i < len(self.targets):
                target_data = np.concatenate((position_relative(self.targets[i].get_position(), player.get_position()), np.array([self.targets[i].timesteps_left])))
            else:
                target_data = np.array([0, 0, 0])
            obs = np.concatenate((obs, target_data), dtype=np.float32)

        return {agent: np.array(obs.flatten()) for agent in self.agents}
    

    
    def observation_space(self, agent):
        if agent == "critic":
            return self._critic_observation_space
        else:
            return self._observation_spaces[agent]
        

    
    def action_space(self, agent):
        return self._action_space[agent]
    
    

    def render(self, mode='human'):
        if not hasattr(self, 'screen'):
            pygame.init()
            self.window_size = 600  # Set the size of the window
            self.cell_size = self.window_size // max(self.width, self.height)
            self.screen = pygame.display.set_mode((self.width * self.cell_size, self.height * self.cell_size))
            pygame.display.set_caption('Collectors Environment')
            self.clock = pygame.time.Clock()

        # Fill the screen with green
        self.screen.fill((0, 150, 0))

        for target in self.targets:
            # Draw the ball
            target_pos = target.get_position()
            pygame.draw.circle(self.screen, (160, 160, 160), (target_pos[0] * self.cell_size + self.cell_size // 2, 
                                target_pos[1] * self.cell_size + self.cell_size // 2), self.cell_size // 2)

        # Draw the players
        for agent, player in self.players.items():
            color = (255, 0, 0)
            pos = player.get_position()
            direction = player.get_direction()
            pygame.draw.rect(self.screen, color, (pos[0] * self.cell_size, 
                                                pos[1] * self.cell_size, 
                                                self.cell_size, self.cell_size))
            pygame.draw.rect(self.screen, color, ((pos[0] + direction[0]) * self.cell_size,
                                                (pos[1] + direction[1]) * self.cell_size, 
                                                self.cell_size * 0.5, self.cell_size * 0.5))

        pygame.display.flip()  # Update the full display Surface to the screen

        # Handle quitting from the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


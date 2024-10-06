import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from gym.spaces import Box, Discrete, MultiDiscrete, Tuple

class Agent(nn.Module):
    def __init__(self, env):
        super(Agent, self).__init__()

        self.action_space = env.action_space(env.agents[0])
        if type(self.action_space) == Box:
            self.n = self.action_space.shape[0]
        elif type(self.action_space) == Discrete:
            self.n = self.action_space.n
        elif type(self.action_space) == MultiDiscrete:
            self.n = self.action_space.nvec.sum()
        elif type(self.action_space) == Tuple:
            self.n = 0
            for space in self.action_space:
                if type(space) == Box:
                    self.n += space.shape[0]
                elif type(space) == Discrete:
                    self.n += space.n
                elif type(space) == MultiDiscrete:
                    self.n += space.nvec.sum()
       
        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(np.array(env.observation_space(env.agents[0]).shape).prod(), 128)),
            nn.Tanh(),
            self.layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            self.layer_init(nn.Linear(np.array(env.observation_space(env.agents[0]).shape).prod(), 128)),
            nn.Tanh(),
            self.layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, self.n), std=0.01),
        )
        if isinstance(self.action_space, Box):
            self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(self.action_space.shape)))
        if isinstance(self.action_space, Tuple):
            number_of_continuous_actions = 0
            for space in self.action_space:
                if isinstance(space, Box):
                    number_of_continuous_actions += space.shape[0]
            self.actor_logstd = nn.Parameter(torch.zeros(1, number_of_continuous_actions))

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        if isinstance(self.action_space, MultiDiscrete):
            split_logits = torch.split(logits, self.action_space.nvec.tolist(), dim=1)
            multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
            if action is None:
                action = torch.stack([categorical.sample() for categorical in multi_categoricals])
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            return action.T, logprob.sum(0), entropy.sum(0), self.critic(x)
        elif isinstance(self.action_space, Discrete):
            categorical = Categorical(logits=logits)
            if action is None:
                action = categorical.sample()
            logprob = categorical.log_prob(action)
            entropy = categorical.entropy()
            return action, logprob, entropy, self.critic(x)
        elif isinstance(self.action_space, Box):
            action_mean = self.actor(x)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
        elif isinstance(self.action_space, Tuple):
            actions = []
            logprobs = []
            entropies = []
            offset = 0
            logstd_offset = 0
            action_offset = 0
            for space in self.action_space:
                if isinstance(space, Discrete):
                    categorical = Categorical(logits=logits[:, offset:offset + space.n])
                    if action is None:
                        cur_action = categorical.sample()
                    else:
                        cur_action = action[:, action_offset:action_offset + space.n]
                        action_offset += space.n
                    actions.append(cur_action)
                    logprobs.append(categorical.log_prob(cur_action))
                    entropies.append(categorical.entropy())
                    offset += space.n
                elif isinstance(space, Box):
                    action_mean = logits[:, offset:offset + space.shape[0]]
                    action_logstd = self.actor_logstd[:, logstd_offset:logstd_offset + space.shape[0]]
                    action_std = torch.exp(action_logstd)
                    probs = Normal(action_mean, action_std)
                    if action is None:
                        cur_action = probs.sample()
                    else:
                        cur_action = action[:, action_offset:action_offset + space.shape[0]]
                        action_offset += space.shape[0]
                    actions.append(cur_action)
                    logprobs.append(probs.log_prob(cur_action).sum(1))
                    entropies.append(probs.entropy().sum(1))
                    offset += space.shape[0]
                    logstd_offset += space.shape[0]
                elif isinstance(space, MultiDiscrete):
                    cur_logits = logits[:, offset:offset + space.nvec.sum()]
                    split_logits = torch.split(cur_logits, space.nvec.tolist(), dim=1)
                    multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
                    if action is None:
                        cur_action = torch.stack([categorical.sample() for categorical in multi_categoricals])
                    else:
                        cur_action = action[:, action_offset:action_offset + space.nvec.sum()].T
                        action_offset += space.nvec.sum()
                    current_logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(cur_action, multi_categoricals)])
                    current_entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
                    actions.append(cur_action.T)
                    logprobs.append(current_logprob.sum(0))
                    entropies.append(current_entropy.sum(0))
                    offset += space.nvec.sum()
            actions = torch.cat(actions, dim=1)
            logprobs = torch.stack(logprobs).sum(0)
            entropies = torch.stack(entropies).sum(0)
            return actions, logprobs, entropies, self.critic(x)
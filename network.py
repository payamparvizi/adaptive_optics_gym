"""
        This file contains a neural network modules to define the actor and critic networks
"""


#----------------------------- Importing modules -----------------------------#
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np


#----------------------------------- Actor -----------------------------------#
class Actor(nn.Module):

    def __init__(self, state_dim, act_dim, hidden_dim, init_w=3e-3):
        super(Actor, self).__init__()

        # Definition of the layers with weight and bias initialization:
        self.layer1a = nn.Linear(state_dim, hidden_dim)
        init.uniform_(self.layer1a.weight, -1./np.sqrt(state_dim), 1./np.sqrt(state_dim))
        init.uniform_(self.layer1a.bias, -1./np.sqrt(state_dim), 1./np.sqrt(state_dim))
        
        self.layer2a = nn.Linear(hidden_dim, hidden_dim)
        init.uniform_(self.layer2a.weight, -1./np.sqrt(hidden_dim), 1./np.sqrt(hidden_dim))
        init.uniform_(self.layer2a.bias, -1./np.sqrt(hidden_dim), 1./np.sqrt(hidden_dim))

        self.layer3a = nn.Linear(hidden_dim, hidden_dim)
        init.uniform_(self.layer3a.weight, -1./np.sqrt(hidden_dim), 1./np.sqrt(hidden_dim))
        init.uniform_(self.layer3a.bias, -1./np.sqrt(hidden_dim), 1./np.sqrt(hidden_dim))
        
        self.outputa = nn.Linear(hidden_dim, act_dim)
        self.outputa.weight.data.uniform_(-init_w, init_w)
        self.outputa.bias.data.uniform_(-init_w, init_w)

        # Dropout for Regularizing the neural network
        self.dropout = nn.Dropout(0.50)


    # forward pass on the neural network
    def forward(self, state):

        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float)

        x = F.relu(self.layer1a(state))
        x = self.dropout(x)

        x = F.relu(self.layer2a(x))
        x = self.dropout(x)

        x = F.relu(self.layer3a(x))
        x = self.dropout(x)

        mu = self.outputa(x)
        return mu


    # Generating action and log probability
    def get_action(self, obs, cov_mat):
        mean = self.forward(obs)
        dist = MultivariateNormal(mean, cov_mat)

        action = dist.rsample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob


    # Evaluating the actions and log probabilities
    def evaluate(self, batch_obs, batch_act, cov_mat):
        mean = self.forward(batch_obs)
        dist = MultivariateNormal(mean, cov_mat)

        action = dist.sample()

        log_prob = dist.log_prob(action)
        log_prob_ = dist.log_prob(batch_act)

        return mean, log_prob, log_prob_


    # training the actor
    def train(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()



#-------------------------------- Critic #1 ----------------------------------#
# Definition of the Q-network
class CriticQ1(nn.Module):
    def __init__(self, num_actions, num_inputs, hidden_size, init_w=3e-3):
        super(CriticQ1, self).__init__()

        # Definition of the layers with weight and bias initialization:
        self.layer1c1 = nn.Linear(num_inputs + num_actions, hidden_size)
        init.uniform_(self.layer1c1.weight, -1./np.sqrt(num_inputs + num_actions), 1./np.sqrt(num_inputs + num_actions))
        init.uniform_(self.layer1c1.bias, -1./np.sqrt(num_inputs + num_actions), 1./np.sqrt(num_inputs + num_actions))

        self.layer2c1 = nn.Linear(hidden_size, hidden_size)
        init.uniform_(self.layer2c1.weight, -1./np.sqrt(hidden_size), 1./np.sqrt(hidden_size))
        init.uniform_(self.layer2c1.bias, -1./np.sqrt(hidden_size), 1./np.sqrt(hidden_size))

        self.layer3c1 = nn.Linear(hidden_size, hidden_size)
        init.uniform_(self.layer3c1.weight, -1./np.sqrt(hidden_size), 1./np.sqrt(hidden_size))
        init.uniform_(self.layer3c1.bias, -1./np.sqrt(hidden_size), 1./np.sqrt(hidden_size))
        
        self.outputc1 = nn.Linear(hidden_size, 1)
        self.outputc1.weight.data.uniform_(-init_w, init_w)
        self.outputc1.bias.data.uniform_(-init_w, init_w)

        # Dropout for Regularizing the neural network
        self.dropout = nn.Dropout(0.50)


    # forward pass on the neural network
    def forward(self, state, action):

        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float)

        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float)

        x = torch.cat([state, action], 1)
        x = F.relu(self.layer1c1(x))
        x = self.dropout(x)

        x = F.relu(self.layer2c1(x))
        x = self.dropout(x)

        x = F.relu(self.layer3c1(x))
        x = self.dropout(x)

        x = self.outputc1(x)
        return x


    # training the critic
    def train(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


#-------------------------------- Critic #2 ----------------------------------#
# Definition of the Q-network
class CriticQ2(nn.Module):
    def __init__(self, num_actions, num_inputs, hidden_size, init_w=3e-3):
        super(CriticQ2, self).__init__()

    # Definition of the layers with weight and bias initialization:
        self.layer1c2 = nn.Linear(num_inputs + num_actions, hidden_size)
        init.uniform_(self.layer1c2.weight, -1./np.sqrt(num_inputs + num_actions), 1./np.sqrt(num_inputs + num_actions))
        init.uniform_(self.layer1c2.bias, -1./np.sqrt(num_inputs + num_actions), 1./np.sqrt(num_inputs + num_actions))

        self.layer2c2 = nn.Linear(hidden_size, hidden_size)
        init.uniform_(self.layer2c2.weight, -1./np.sqrt(hidden_size), 1./np.sqrt(hidden_size))
        init.uniform_(self.layer2c2.bias, -1./np.sqrt(hidden_size), 1./np.sqrt(hidden_size))

        self.layer3c2 = nn.Linear(hidden_size, hidden_size)
        init.uniform_(self.layer3c2.weight, -1./np.sqrt(hidden_size), 1./np.sqrt(hidden_size))
        init.uniform_(self.layer3c2.bias, -1./np.sqrt(hidden_size), 1./np.sqrt(hidden_size))
        
        self.outputc2 = nn.Linear(hidden_size, 1)
        self.outputc2.weight.data.uniform_(-init_w, init_w)
        self.outputc2.bias.data.uniform_(-init_w, init_w)

        # Dropout for Regularizing the neural network
        self.dropout = nn.Dropout(0.50)


    # forward pass on the neural network
    def forward(self, state, action):
        
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float)
            
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float)
            
        x = torch.cat([state, action], 1)
        x = F.relu(self.layer1c2(x))
        x = self.dropout(x)
        
        x = F.relu(self.layer2c2(x))
        x = self.dropout(x)
        
        x = F.relu(self.layer3c2(x))
        x = self.dropout(x)
        
        x = self.outputc2(x)
        return x

    # training the critic
    def train(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


#----------------------------- Value function --------------------------------#
# Definition of the value function
class CriticV(nn.Module):
    def __init__(self, num_actions, num_inputs, hidden_size, init_w=3e-3):
        super(CriticV, self).__init__()

    # Definition of the layers with weight and bias initialization:
        self.layer1c2 = nn.Linear(num_inputs, hidden_size)
        init.uniform_(self.layer1c2.weight, -1./np.sqrt(num_inputs), 1./np.sqrt(num_inputs))
        init.uniform_(self.layer1c2.bias, -1./np.sqrt(num_inputs), 1./np.sqrt(num_inputs))

        self.layer2c2 = nn.Linear(hidden_size, hidden_size)
        init.uniform_(self.layer2c2.weight, -1./np.sqrt(hidden_size), 1./np.sqrt(hidden_size))
        init.uniform_(self.layer2c2.bias, -1./np.sqrt(hidden_size), 1./np.sqrt(hidden_size))

        self.layer3c2 = nn.Linear(hidden_size, hidden_size)
        init.uniform_(self.layer3c2.weight, -1./np.sqrt(hidden_size), 1./np.sqrt(hidden_size))
        init.uniform_(self.layer3c2.bias, -1./np.sqrt(hidden_size), 1./np.sqrt(hidden_size))
        
        self.outputc2 = nn.Linear(hidden_size, 1)
        self.outputc2.weight.data.uniform_(-init_w, init_w)
        self.outputc2.bias.data.uniform_(-init_w, init_w)

        # Dropout for Regularizing the neural network
        self.dropout = nn.Dropout(0.50)


    # forward pass on the neural network
    def forward(self, state):
        
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float)
            
        x = F.relu(self.layer1c2(state))
        x = self.dropout(x)
        
        x = F.relu(self.layer2c2(x))
        x = self.dropout(x)
        
        x = F.relu(self.layer3c2(x))
        x = self.dropout(x)
        
        x = self.outputc2(x)
        return x

    # training the value function
    def train(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


#-----------------------------------------------------------------------------#
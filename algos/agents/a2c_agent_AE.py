import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from torch import nn


class A2CAgent_AE():
    def __init__(self, input_shape, action_size, seed, device, gamma, alpha,beta, update_every, actor_m, AE_m):
        """Initialize an Agent object.
        Params
        ======
            input_shape (tuple): dimension of each state (C, H, W)
            action_size (int): dimension of each action
            seed (int): random seed
            device(string): Use Gpu or CPU
            gamma (float): discount factor
            alpha (float): Actor learning rate
            beta (float): Critic learning rate 
            update_every (int): how often to update the network
            actor_m(Model): Pytorch Actor Model
            critic_m(Model): PyTorch Critic Model
        """
        self.input_shape = input_shape
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.update_every = update_every

        # Actor-Network
        self.actor_net = actor_m(input_shape, action_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.alpha)

        # Critic-Network
        self.AE_net = AE_m(input_shape,action_size).to(self.device)
        self.AE_optimizer = optim.Adam(self.AE_net.parameters(), lr=self.beta)

        # Memory
        self.log_probs = []
        self.values    = []
        self.rewards   = []
        self.masks     = []
        self.entropies = []

        self.t_step = 0

        self.activation = {}

    def step(self, state, log_prob, entropy, reward, done, next_state):

        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        
       # value = self.critic_net(state)
        
        # Save experience in  memory
       # self.log_probs.append(log_prob)
        #self.values.append(value)
        #self.rewards.append(torch.from_numpy(np.array([reward])).to(self.device))
        #self.masks.append(torch.from_numpy(np.array([1 - done])).to(self.device))
        #self.entropies.append(entropy)

        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
            loss = self.learn(state)
            self.reset_memory()
            return loss
                
    def act(self, state):
        """Returns action, log_prob, entropy for given state as per current policy."""
        
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        action_probs = self.actor_net(state)

        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        entropy = action_probs.entropy().mean()

        ##test add:
        image_reconstruction = self.AE_net(state)

        return action.item(), log_prob, entropy

    
    def learn(self,states):
        
        # Get expected Q values from policy model
        image_reconstruction = self.AE_net(states)
 
        # Compute loss
        loss = nn.MSELoss()(image_reconstruction, states)
        
        # Minimize the loss
        self.AE_optimizer.zero_grad()
        loss.backward()
        self.AE_optimizer.step()
        
        return loss.cpu().detach().numpy()

     

    def reset_memory(self):
        del self.log_probs[:]
        del self.rewards[:]
        del self.values[:]
        del self.masks[:]
        del self.entropies[:]

    def compute_returns(self, next_value, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step] + gamma * R * self.masks[step]
            returns.insert(0, R)
        return returns
    
    def getActivation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook
            
    
    def registration(self):
        
        self.h1 = self.AE_net.conv_1.register_forward_hook(self.getActivation('Conv_1'))
        self.h2 = self.AE_net.conv_2.register_forward_hook(self.getActivation('Conv_2'))
        self.h3 = self.AE_net.conv_3.register_forward_hook(self.getActivation('Conv_3'))
        self.h4 = self.AE_net.fc1.register_forward_hook(self.getActivation('fc1'))
        
        
        self.h5 = self.AE_net.conv_1_bis.register_forward_hook(self.getActivation('Conv_1_bis'))
        self.h6 = self.AE_net.conv_2_bis.register_forward_hook(self.getActivation('Conv_2_bis'))
        self.h7 = self.AE_net.conv_3_bis.register_forward_hook(self.getActivation('Conv_3_bis'))
        self.h8 = self.AE_net.fc1_bis.register_forward_hook(self.getActivation('fc1_bis'))
        
    def detach(self):
        self.h1.remove()
        self.h2.remove()
        self.h3.remove()
        self.h4.remove()
        
        self.h5.remove()
        self.h6.remove()
        self.h7.remove()
        self.h8.remove()

        
        
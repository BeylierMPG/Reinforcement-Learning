


import time
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import math
import sys
import os
sys.path.append("/Users/charlottebeylier/Documents/PhD/Atari1.0/Reinforcement-Learning_modif/cgames/02_space_invader/Analysis")
from Manifold_Analysis import Manifold_analysis
sys.path.append("/Users/charlottebeylier/Documents/PhD/Atari1.0/Reinforcement-Learning_modif")

from algos.agents import A2CAgent
from algos.models import ActorCnn, CriticCnn
from algos.preprocessing.stack_frame import preprocess_frame, stack_frame

from algos.agents.dqn_agent import DQNAgent
from algos.models.dqn_cnn import DQNCnn

from algos.preprocessing.stack_frame import preprocess_frame, stack_frame


from algos.agents.ddqn_agent import DDQNAgent
from algos.models.ddqn_cnn import DDQNCnn

class Generate_data():    
    
    
    
    def __init__(self,device):

        env = gym.make('SpaceInvaders-v0')
        env.seed(0)

        self.INPUT_SHAPE = (4, 84, 84)
        self.ACTION_SIZE = env.action_space.n
        self.SEED = 0
        self.device = device 
        
       

    def stack_frames(self,frames, state, is_new=False):
        frame = preprocess_frame(state, (8, -12, -12, 4), 84)
        frames = stack_frame(frames, frame, is_new)

        return frames




    def initialization_agents(self,structure_network):
        
        PATH_MODELS = "/Users/charlottebeylier/Documents/PhD/Atari1.0/Reinforcement-Learning_modif/cgames/02_space_invader/Analysis"
       
    
        
        if structure_network == "dqn":
            
            GAMMA = 0.99           # discount factor
            BUFFER_SIZE = 100000   # replay buffer size
            BATCH_SIZE = 64        # Update batch size
            LR = 0.0001            # learning rate 
            TAU = 1e-3             # for soft update of target parameters
            UPDATE_EVERY = 1       # how often to update the network
            UPDATE_TARGET = 10000  # After which thershold replay to be started 
            EPS_START = 0.99       # starting value of epsilon
            EPS_END = 0.01         # Ending value of epsilon
            EPS_DECAY = 100   
            
            


            agent_init = DQNAgent(self.INPUT_SHAPE, self.ACTION_SIZE, self.SEED, self.device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)
            
            agent_1000 = DQNAgent(self.INPUT_SHAPE, self.ACTION_SIZE, self.SEED, self.device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)
            
            agent_2000 = DQNAgent(self.INPUT_SHAPE, self.ACTION_SIZE, self.SEED, self.device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)
            
            agent_3000 = DQNAgent(self.INPUT_SHAPE, self.ACTION_SIZE, self.SEED, self.device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)
            
            agent_4000 = DQNAgent(self.INPUT_SHAPE, self.ACTION_SIZE, self.SEED, self.device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)
            
            agent_5000 = DQNAgent(self.INPUT_SHAPE, self.ACTION_SIZE, self.SEED, self.device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)


            Liste_agents = {
                "agent_dqn_init" : agent_init,
                "agent_dqn_1000" : agent_1000,
                "agent_dqn_2000" : agent_2000,
                "agent_dqn_3000" : agent_3000,
                "agent_dqn_4000" : agent_4000,
               
            }
            
            
            for a in Liste_agents.keys():
                
                if a != "agent_dqn_init":

                    PATH = os.path.join(PATH_MODELS,"Models_training_dqn_space_invader",a+ ".pt")
                    print(PATH)
                    checkpoint = torch.load(PATH , map_location=torch.device('cpu'))
                    Liste_agents[a].policy_net.load_state_dict(checkpoint['modelA_state_dict'])
                    Liste_agents[a].target_net.load_state_dict(checkpoint['modelB_state_dict'])
                    Liste_agents[a].optimizer.load_state_dict(checkpoint['optimizer_state_dict'])




        if structure_network == "ddqn":
            
            GAMMA = 0.99           # discount factor
            BUFFER_SIZE = 100000   # replay buffer size
            BATCH_SIZE = 32        # Update batch size
            LR = 0.0001            # learning rate 
            TAU = .1               # for soft update of target parameters
            UPDATE_EVERY = 100     # how often to update the network
            UPDATE_TARGET = 10000  # After which thershold replay to be started 
            EPS_START = 0.99       # starting value of epsilon
            EPS_END = 0.01         # Ending value of epsilon
            EPS_DECAY = 100        # Rate by which epsilon to be decayed
            
            
            
            agent_init =  DDQNAgent(self.INPUT_SHAPE, self.ACTION_SIZE, self.SEED, self.device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)

            agent_1000 = DDQNAgent(self.INPUT_SHAPE, self.ACTION_SIZE, self.SEED, self.device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)

            agent_2000 = DDQNAgent(self.INPUT_SHAPE, self.ACTION_SIZE, self.SEED, self.device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)

            agent_3000 = DDQNAgent(self.INPUT_SHAPE, self.ACTION_SIZE, self.SEED, self.device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)

            agent_4000 = DDQNAgent(self.INPUT_SHAPE, self.ACTION_SIZE, self.SEED, self.device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)
            
            agent_5000 = DDQNAgent(self.INPUT_SHAPE, self.ACTION_SIZE, self.SEED, self.device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)


            Liste_agents = {
                "agent_ddqn_init" : agent_init,
                "agent_ddqn_1000" : agent_1000,
                "agent_ddqn_2000" : agent_2000,
                "agent_ddqn_3000" : agent_3000,
                "agent_ddqn_4000" : agent_4000,
                "agent_ddqn_5000" : agent_5000
            }

            for a in Liste_agents.keys():
                if a != "agent_ddqn_init":

                    PATH = os.path.join(PATH_MODELS,"Models_training_ddqn_space_invader",a+ ".pt")
                    print(PATH)
                    checkpoint = torch.load(PATH , map_location=torch.device('cpu'))
                    Liste_agents[a].policy_net.load_state_dict(checkpoint['modelA_state_dict'])
                    Liste_agents[a].target_net.load_state_dict(checkpoint['modelB_state_dict'])
                    Liste_agents[a].optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        
        
        
        if structure_network == "a2c":
            
            GAMMA = 0.99           # discount factor
            ALPHA= 0.0001          # Actor learning rate
            BETA = 0.0005          # Critic learning rate
            UPDATE_EVERY = 100     # how often to update the network 
            

            agent_init = A2CAgent(self.INPUT_SHAPE, self.ACTION_SIZE, self.SEED, self.device, GAMMA, ALPHA, BETA, UPDATE_EVERY, ActorCnn, CriticCnn)
            agent_1000 = A2CAgent(self.INPUT_SHAPE, self.ACTION_SIZE, self.SEED, self.device, GAMMA, ALPHA, BETA, UPDATE_EVERY, ActorCnn, CriticCnn)
            agent_2000 = A2CAgent(self.INPUT_SHAPE, self.ACTION_SIZE, self.SEED, self.device, GAMMA, ALPHA, BETA, UPDATE_EVERY, ActorCnn, CriticCnn)
            agent_3000 = A2CAgent(self.INPUT_SHAPE, self.ACTION_SIZE, self.SEED, self.device, GAMMA, ALPHA, BETA, UPDATE_EVERY, ActorCnn, CriticCnn)
            agent_4000 = A2CAgent(self.INPUT_SHAPE, self.ACTION_SIZE, self.SEED, self.device, GAMMA, ALPHA, BETA, UPDATE_EVERY, ActorCnn, CriticCnn)
            agent_5000 = A2CAgent(self.INPUT_SHAPE, self.ACTION_SIZE, self.SEED, self.device, GAMMA, ALPHA, BETA, UPDATE_EVERY, ActorCnn, CriticCnn)


            Liste_agents = {
                "agent_a2c_init" : agent_init,
                "agent_a2c_1000" : agent_1000,
                "agent_a2c_2000" : agent_2000,
                "agent_a2c_3000" : agent_3000,
                "agent_a2c_4000" : agent_4000,
                "agent_a2c_5000" : agent_5000,
            }


            for a in Liste_agents.keys():
                if a != "agent_a2c_init":

                    PATH = os.path.join(PATH_MODELS,"Models_training_a2c_space_invader",a+ ".pt")
                    print(PATH)
                    checkpoint = torch.load(PATH , map_location=torch.device('cpu'))
                    Liste_agents[a].actor_net.load_state_dict(checkpoint['modelA_state_dict'])
                    Liste_agents[a].critic_net.load_state_dict(checkpoint['modelB_state_dict'])
                    Liste_agents[a].actor_optimizer.load_state_dict(checkpoint['optimizerA_state_dict'])
                    Liste_agents[a].critic_optimizer.load_state_dict(checkpoint['optimizerB_state_dict'])


        return Liste_agents



    def run_episodes(self,structure_network,length_trial,number_episode):
        
        
        env = gym.make('SpaceInvaders-v0')
        env.seed(0)
        
        EPISODES = number_episode
        start_epoch = 0
        scores_window = deque(maxlen=20)
        scores = []
        agent_number = 0
        i = 0
        
        
        if structure_network == "ddqn":
            Names_hook = ["Conv_1","Conv_2","Conv_3","advantage","value "]
        else :
            Names_hook = ["Conv_1","Conv_2","Conv_3","fc1"]
            
    

        Liste_agents = self.initialization_agents(structure_network)
        Liste_activation = [[[[] for i in range(len(Names_hook))] for j in range(len( Liste_agents))] for episode in range(EPISODES)]
        Final_score = [[[] for j in range(len( Liste_agents))] for episode in range(EPISODES)]
        
        
        
        for agent in Liste_agents.values():

            for episode in range(EPISODES):
                while i < length_trial:
                    agent.registration()
                    score = 0
                    i = 0

                    Liste_activation[episode][agent_number] = [[] for i in range(len(Names_hook))]
                    state = self.stack_frames(None, env.reset(), True)
                   
                    while True:

                        if structure_network == "a2c":
                            action, _, _ = agent.act(state)
                        
                        else:
                            action = agent.act(state)

                        for h in range(len(Names_hook)):
                            Liste_activation[episode][agent_number][h].append(torch.flatten(agent.activation[Names_hook[h]])) 

                        next_state, reward, done, _ = env.step(action)
                        score += reward
                        state = self.stack_frames(state, next_state, False)
                        i+=1
                        if done:
                            print('\nEpisode :{} \tAgent number :{} \tFinal score: {:.2f} \tNumber of steps: {}'.format(episode,agent_number, score,i), end="")

                            break 
                            
                    env.close()
                    agent.detach()
                    
                i = 0  
                Final_score[episode][agent_number].append(score)

            agent_number += 1
            
        return Liste_activation,Final_score



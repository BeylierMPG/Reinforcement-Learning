
import torch
import sys
import os

sys.path.append("/Users/charlottebeylier/Documents/PhD/Atari1.0/Reinforcement-Learning_modif")

from algos.agents import A2CAgent
from algos.models import ActorCnn, CriticCnn

from algos.agents.dqn_agent import DQNAgent
from algos.models.dqn_cnn import DQNCnn


from algos.agents.ddqn_agent import DDQNAgent
from algos.models.ddqn_cnn import DDQNCnn

from algos.agents.a2c_agent_AE import A2CAgent_AE
from algos.models.AE_cnn import AECnn


def initialization_agents(structure_network,INPUT_SHAPE,ACTION_SIZE,SEED,device):
        
        PATH_MODELS = "/Users/charlottebeylier/Documents/PhD/Atari1.0/Reinforcement-Learning_modif/cgames/02_space_invader/Experiments/Trained_Models"
       
    
        
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
            
            


            agent_init = DQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)
            
            agent_1000 = DQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)
            
            agent_2000 = DQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)
            
            agent_3000 = DQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)
            
            agent_4000 = DQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)
            
            agent_5000 = DQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)
            
            agent_6000 = DQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)
            
            agent_7000 = DQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)


            Liste_agents = {
                "agent_dqn_init" : agent_init,
                "agent_dqn_1000" : agent_1000,
                "agent_dqn_2000" : agent_2000,
                "agent_dqn_3000" : agent_3000,
                "agent_dqn_4000" : agent_4000,
                "agent_dqn_5000" : agent_5000,
                "agent_dqn_6000" : agent_6000,
                "agent_dqn_7000" : agent_7000,
               
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
            
            
            
            agent_init =  DDQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)

            agent_1000 = DDQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)

            agent_2000 = DDQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)

            agent_3000 = DDQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)

            agent_4000 = DDQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)
            
            agent_5000 = DDQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)


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
            

            agent_init = A2CAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, GAMMA, ALPHA, BETA, UPDATE_EVERY, ActorCnn, CriticCnn)
            agent_1000 = A2CAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, GAMMA, ALPHA, BETA, UPDATE_EVERY, ActorCnn, CriticCnn)
            agent_2000 = A2CAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, GAMMA, ALPHA, BETA, UPDATE_EVERY, ActorCnn, CriticCnn)
            agent_3000 = A2CAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, GAMMA, ALPHA, BETA, UPDATE_EVERY, ActorCnn, CriticCnn)
            agent_4000 = A2CAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, GAMMA, ALPHA, BETA, UPDATE_EVERY, ActorCnn, CriticCnn)
            agent_5000 = A2CAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, GAMMA, ALPHA, BETA, UPDATE_EVERY, ActorCnn, CriticCnn)
            agent_6000 = A2CAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, GAMMA, ALPHA, BETA, UPDATE_EVERY, ActorCnn, CriticCnn)
            agent_7000 = A2CAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, GAMMA, ALPHA, BETA, UPDATE_EVERY, ActorCnn, CriticCnn)
            agent_8000 = A2CAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, GAMMA, ALPHA, BETA, UPDATE_EVERY, ActorCnn, CriticCnn)



            Liste_agents = {
                "agent_a2c_0_2" : agent_init,
                "agent_a2c_1000_2" : agent_1000,
                "agent_a2c_2000_2" : agent_2000,
                "agent_a2c_3000_2" : agent_3000,
                "agent_a2c_4000_2" : agent_4000,
                "agent_a2c_5000_2" : agent_5000,
                "agent_a2c_6000_2" : agent_6000,
                "agent_a2c_7000_2" : agent_7000,
                "agent_a2c_8000_2" : agent_8000,
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


        if structure_network == "a2c_AE":
          
            
            GAMMA = 0.99           # discount factor
            ALPHA= 0.0001          # Actor learning rate
            BETA = 0.0005          # Critic learning rate
            UPDATE_EVERY = 100     # how often to update the network 

            agent_1000 = A2CAgent_AE(INPUT_SHAPE, ACTION_SIZE, SEED, device, GAMMA,ALPHA, BETA, UPDATE_EVERY, ActorCnn, AECnn)

            Liste_agents = {
                "agent_a2c_AE_1000" : agent_1000,
            }

            for a in Liste_agents.keys():
                if a != "agent_a2c_init":

                    PATH = os.path.join(PATH_MODELS,"Models_training_a2c_AE_space_invader",a+ ".pt")
                    print(PATH)
                    checkpoint = torch.load(PATH , map_location=torch.device('cpu'))
                    Liste_agents[a].actor_net.load_state_dict(checkpoint['modelA_state_dict'])
                    Liste_agents[a].AE_net.load_state_dict(checkpoint['modelB_state_dict'])
                    Liste_agents[a].actor_optimizer.load_state_dict(checkpoint['optimizerA_state_dict'])
                    Liste_agents[a].AE_optimizer.load_state_dict(checkpoint['optimizerB_state_dict'])
            
                                    
            


        return Liste_agents
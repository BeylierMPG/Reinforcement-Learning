import gym
import torch
import sys
sys.path.append("/Users/charlottebeylier/Documents/PhD/Atari1.0/Reinforcement-Learning_modif/cgames/02_space_invader/Experiment")
from Positions_agent_aliens import position_alien, position_agent
from Agents_loading import initialization_agents
sys.path.append("/Users/charlottebeylier/Documents/PhD/Atari1.0/Reinforcement-Learning_modif")
from algos.preprocessing.stack_frame import preprocess_frame, stack_frame





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



    def copy_model(self,model_init,model_hooks):
        G = []
        for param_tensor in model_hooks.actor_net.state_dict():
            G.append(param_tensor)
        H = []
        for param_tensor in model_hooks.critic_net.state_dict():
            H.append(param_tensor)
        i = 0
        for param_tensor in model_init.actor_net.state_dict():
            model_hooks.actor_net.state_dict()[G[i]].data.copy_(model_init.actor_net.state_dict()[param_tensor])
            i+=1
        i = 0
        for param_tensor in model_init.critic_net.state_dict():
            model_hooks.critic_net.state_dict()[H[i]].data.copy_(model_init.critic_net.state_dict()[param_tensor])
            i+=1

    def run_episodes(self,structure_network,length_trial,number_episode):
        
        
        env = gym.make('SpaceInvaders-v0')
        env.seed(0)
        EPISODES = number_episode
        agent_number = 0
        i = 0
        
        
        if structure_network == "ddqn":
            Names_hook = ["Conv_1","Conv_2","Conv_3","advantage","value "]

        elif structure_network == "a2c_AE" :
            Names_hook = ["Conv_1","Conv_2","Conv_3","fc1","Conv_1_bis","Conv_2_bis","Conv_3_bis","fc1_bis"]

        else :
            Names_hook = ["Conv_1","Conv_2","Conv_3","fc1"]
            
        print("Names_hook",Names_hook)

        Liste_agents = initialization_agents(structure_network,self.INPUT_SHAPE,self.ACTION_SIZE,self.SEED,self.device )

        # Liste_agents = self.initialization_agents(structure_network)
        Liste_activation = [[[[] for i in range(len(Names_hook))] for j in range(len( Liste_agents))] for episode in range(EPISODES)]
        Final_score = [[[] for j in range(len( Liste_agents))] for episode in range(EPISODES)]
        
        Liste_position_agent = [[[] for j in range(len( Liste_agents))] for episode in range(EPISODES)]
        Liste_action_agent = [[[] for j in range(len( Liste_agents))] for episode in range(EPISODES)]

        Liste_position_alien = [[[] for j in range(len( Liste_agents))] for episode in range(EPISODES)]
        
        for agent in Liste_agents.values():
         
            for episode in range(EPISODES):
                while i < length_trial:
                    agent.registration()
                    score = 0
                    i = 0

                    Liste_activation[episode][agent_number] = [[] for i in range(len(Names_hook))]
                    Liste_position_agent[episode][agent_number] = []
                    Liste_position_alien[episode][agent_number] = []
                    Liste_action_agent[episode][agent_number] = []
                    
                    state = self.stack_frames(None, env.reset(), True)
                   
                    while True:

                        if structure_network == "a2c" or structure_network == "a2c_AE":
                            action, _, _ = agent.act(state)
                        
                        else:
                            action = agent.act(state)
                        
                        next_state, reward, done, _ = env.step(action)
                        score += reward

                        for h in range(len(Names_hook)):
                            Liste_activation[episode][agent_number][h].append(torch.flatten(agent.activation[Names_hook[h]])) 

                        
                        state = self.stack_frames(state, next_state, False)
                        
                        Liste_position_agent[episode][agent_number].append(position_agent(next_state[185:195,:,:]))
                        Liste_position_alien[episode][agent_number].append(position_alien(next_state[35:150,:,:]))
                        Liste_action_agent[episode][agent_number].append(action)
                        
                        
                        i+=1
                        if done:
                            print('\nEpisode :{} \tAgent number :{} \tFinal score: {:.2f} \tNumber of steps: {}'.format(episode,agent_number, score,i), end="")

                            break 
                            
                    env.close()
                    agent.detach()
                    
                i = 0  
                Final_score[episode][agent_number].append(score)

            agent_number += 1
        
 
        return Liste_activation,Final_score,Liste_position_agent,Liste_position_alien,Liste_action_agent





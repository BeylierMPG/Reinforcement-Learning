import torch
import torch.nn as nn
import torch.autograd as autograd 
import torch.nn.functional as F

class DQNCnn(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNCnn, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU()
        )
            
        self.conv_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU()
        )
        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(3136, 512),    # initially 3136 was replaced by feature_size()
            nn.ReLU()
        )
       # print("self.feature_size()",self.feature_size())
        self.fc2 = nn.Linear(512, self.num_actions)
        
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
  
        return x
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
        #return self.conv(autograd.Variable(torch.zeros(1, 64))).view(1, -1).size(1) ###CHANGE HERE TO THE LAST LAYER NAME BEFORE FC1
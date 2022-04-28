from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets

from collections import OrderedDict
from functools import partial
from time import time
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, datasets

import torch
import numpy as np
from collections import deque
import math

class Manifold_analysis():
    
    def __init__(self,length_trial):
        
        self.length_trial = length_trial


    def length_format(self,Activation):
        if len(Activation) < self.length_trial:
            print("Error: the trial is not long enough")
        else: 
            return Activation[:][:self.length_trial]

        
        
    def prepro(self,Activation,Prepro_length):
        
        if Prepro_length:
            activation = self.length_format(Activation)
        else:
            activation = Activation

        Liste_activation = activation[0].unsqueeze(0)
        for i in range(1,len(activation)):
            Liste_activation = torch.cat((Liste_activation,activation[i].unsqueeze(0)),0)  # if no unsqueeze then does not have the right shape (steps,nodes)
       # Liste_activation = Liste_activation.squeeze(1)
       # print("Shape of the activation list is: ", Liste_activation.shape)
        return Liste_activation.cpu().detach().numpy()
    
    

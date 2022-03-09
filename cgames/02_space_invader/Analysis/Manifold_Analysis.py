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
        
        self.Names_hook = ["fc1","Conv_1","Conv_2","Conv_3"]
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
    
    
    
    def MDS_graph(self,activation,dimension,show_distance_matrice,show_MDS_plot):
        D = pairwise_distances(activation)
        model = MDS(n_components=dimension, dissimilarity='precomputed', random_state=1)
        out = model.fit_transform(D)
        
        if show_distance_matrice:
            plt.figure()
            plt.imshow(D, zorder=2, cmap='Blues', interpolation='nearest')
            plt.colorbar();
            
        if show_MDS_plot:
            plt.figure()
            plt.scatter(out[:, 0], out[:, 1])
            plt.axis('equal');

    


            
    def evolution_in_training(self):
        embedding = Isomap(n_neighbors=12,n_components=3)
        ax = plt.figure(figsize=(20, 15)).add_subplot(projection='3d')
        for id_episode in range(len(self.List_activation)):
            X= embedding.fit_transform(self.List_activation[id_episode])
            
            sc = ax.scatter3D(X[:, 0], X[:, 1], X[:, 2])
            ax.view_init(azim=80, elev=30)
            
        
    
    def Multi_comparison(self,id_episode):
        # Next line to silence pyflakes. This import is needed.
        Axes3D

        n_points = 1000
        X = self.List_activation[id_episode]
        n_neighbors = 12
        n_components = 3

        # Create figure
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(
            "Manifold Learning with %i points, %i neighbors" % (1000, n_neighbors), fontsize=14
        )

        # Set-up manifold methods
        LLE = partial(
            manifold.LocallyLinearEmbedding,
            n_neighbors=n_neighbors,
            n_components=n_components,
            eigen_solver="dense",
        )

        methods = OrderedDict()
        methods["LLE"] = LLE(method="standard")
        methods["LTSA"] = LLE(method="ltsa")
        methods["Hessian LLE"] = LLE(method="hessian")
        methods["Modified LLE"] = LLE(method="modified")
        methods["Isomap"] = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components)
        methods["MDS"] = manifold.MDS(n_components, max_iter=100, n_init=1)
        methods["SE"] = manifold.SpectralEmbedding(
            n_components=n_components, n_neighbors=n_neighbors
        )
        methods["t-SNE"] = manifold.TSNE(n_components=n_components, init="pca", random_state=0)

        # Plot results
        for i, (label, method) in enumerate(methods.items()):
            t0 = time()
            Y = method.fit_transform(X)
            t1 = time()
            print("%s: %.2g sec" % (label, t1 - t0))
            ax = fig.add_subplot(2, 5, 2 + i + (i > 3),projection="3d")
            sc = ax.scatter3D(Y[:, 0], Y[:, 1],Y[:, 2])
            ax.set_title("%s (%.2g sec)" % (label, t1 - t0))
            #ax.xaxis.set_major_formatter(NullFormatter())
            #ax.yaxis.set_major_formatter(NullFormatter())
            ax.axis("tight")

           # ax.view_init(azim=80, elev=30)
        plt.show()
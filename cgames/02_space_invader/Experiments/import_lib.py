def main ():
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
    from Generate_data_activity import Generate_data
    from Manifold_Analysis import Manifold_analysis
    from CCA_analysis import CCA_Analysis
    sys.path.append("/Users/charlottebeylier/Documents/PhD/Atari1.0/Reinforcement-Learning_modif")

    from algos.agents import A2CAgent
    from algos.models import ActorCnn, CriticCnn
    from algos.preprocessing.stack_frame import preprocess_frame, stack_frame

    # importing required libraries
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.preprocessing import scale
    from sklearn.decomposition import PCA, IncrementalPCA
    from sklearn.cross_decomposition import CCA
    import pandas as pd
    import seaborn as sns
    import cv2
    from tabulate import tabulate
    from mpl_toolkits.mplot3d.proj3d import proj_transform
    from matplotlib.text import Annotation
    import matplotlib.pyplot as plt    
    from mpl_toolkits.mplot3d import axes3d
    from mpl_toolkits.mplot3d.art3d import Line3DCollection


    import warnings
    from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix,
        csr_matrix, isspmatrix, dok_matrix, lil_matrix, bsr_matrix)
    warnings.simplefilter('ignore',SparseEfficiencyWarning)

    
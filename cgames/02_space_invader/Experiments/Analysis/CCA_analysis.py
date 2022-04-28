import torch
import numpy as np
import sys
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cross_decomposition import CCA
import pandas as pd



class CCA_Analysis():
    
    def __init__(self,Liste_table_X,Liste_table_Y,LENGTH_TRIAL, activity_layer,position_agent,position_alien,action_agent):
        self.LENGTH_TRIAL = LENGTH_TRIAL
        self.activity_layer = activity_layer
        self.position_agent = position_agent[0:LENGTH_TRIAL]
        self.position_alien = position_alien[0:LENGTH_TRIAL]
        self.action_agent = action_agent[0:LENGTH_TRIAL]
        
        self.pca = self.pca_analysis()
        self.df = self.data_frame()
        self.X_mc,self.Y_mc = self.dataset_stand(Liste_table_X,Liste_table_Y)
        self.cc_res,self.coeff_corr_first_pair,self.coeff_corr_second_pair = self.CCA_analysis()
    
    def pca_analysis(self):
        X = np.transpose(self.activity_layer)
        pca = PCA(n_components = 5) #we have 20 features
        pca.fit(X)
        return pca
    
    def data_frame(self):
        data = {'PC1':self.pca.components_[0],'PC2':self.pca.components_[1],'PC3':self.pca.components_[2],'PC4':self.pca.components_[3],'PC5':self.pca.components_[4], 'Agent_Position':self.position_agent,'Alien_Position':self.position_alien,'Action_agent':self.action_agent}  
        df = pd.DataFrame(data, index =[str(i) for i in range(self.LENGTH_TRIAL)])  
        return df


    def dataset_stand(self,Liste_table_X,Liste_table_Y):
        X = self.df[Liste_table_X]
        Y = self.df[Liste_table_Y]
        X_mc = (X-X.mean())/(X.std())
        Y_mc = (Y-Y.mean())/(Y.std())
        return X_mc,Y_mc

    
    def CCA_analysis(self):
        ca = CCA(n_components=min(self.X_mc.shape[0],self.Y_mc.shape[0]))
        ca.fit(self.X_mc, self.Y_mc)
        X_c, Y_c = ca.transform(self.X_mc, self.Y_mc)
        
        if min(X_c.shape[0],X_c.shape[0]) == 2:
            cc_res = pd.DataFrame({"CCX_1":X_c[:, 0],
                        "CCY_1":Y_c[:, 0],
                        "CCX_2":X_c[:, 1],
                        "CCY_2":Y_c[:, 1],
                        "Actions":self.df.Action_agent.tolist(),
                        })
            
            coeff_corr_first_pair = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0][1]
            coeff_corr_second_pair = np.corrcoef(X_c[:, 1], Y_c[:, 1])[0][1]
            
        else :
            cc_res = pd.DataFrame({"CCX_1":X_c[:, 0],
                        "CCY_1":Y_c[:, 0],
                        "Actions":self.df.Action_agent.tolist(),
                        })
        coeff_corr_first_pair = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0][1]
        coeff_corr_second_pair = np.NaN

        return cc_res,coeff_corr_first_pair,coeff_corr_second_pair
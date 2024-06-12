import pandas as pd
import numpy as np
import pylab as pl
import scipy as sp
import sys
import os 
import copy
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.ensemble import (
                            GradientBoostingRegressor,
                            RandomForestRegressor,
                            StackingRegressor,
                            AdaBoostRegressor
                            )
from genra.rax.skl.reg import *
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate, cross_val_predict
import time
from tqdm import tqdm
from datetime import datetime
from functools import reduce 
from mltox.db.mongo import *
from mltox.db.bc import *
from mltox.db.bio import *
class MLRegressors():

    """
    Parameters:
    -----------

    Est: list of (name,estimator) tuples, (optional)
    k : integer (optional, default=5), Number of splits for CV
    """
    def __init__(self,
                Est = [ 
                ('Random Forest', RandomForestRegressor(random_state=42,max_depth=10)),
                ('Gradient Boosting', GradientBoostingRegressor()),
                ('SVR Linear', SVR(gamma='auto',kernel='linear')),
                ('GenRA', GenRAPredValue(n_neighbors=10,metric='jaccard')),
                ],
                k=5,
                seed=42
                ):
        self.Est = Est
        self.k = k
        self.seed =seed

    def __create_pipelines(self,
                        feature_selectors,
                        n_features_list,
                        stack_regressors):
        pipe_list = []
        VT = ('Variance Threshold',VarianceThreshold(0)) # Eliminates any feature which all samples has same value
        for FS in feature_selectors:
            for n_features in n_features_list:
                FS[1].k = n_features 
                stacked_rgrs = []
                for Rgr in self.Est:
                    pipe_rgr = Pipeline([VT,copy.deepcopy(FS),Rgr]) 
                    stacked_rgrs.append((f'pipe_{FS[0]}{Rgr[0]}_N:{n_features}',pipe_rgr))
                    pipe_list.append((pipe_rgr,n_features))
                if stack_regressors:
                    stacked_model = StackingRegressor(stacked_rgrs,final_estimator=RandomForestRegressor())
                    pipe_list.append((stacked_model,n_features))

        return pipe_list
                

    def calc_regressors_performance(self,
                                    X,
                                    Y,
                                    n_features_list=[100],
                                    n_jobs=1,
                                    feature_selectors=[('ANOVA',SelectKBest(f_regression,k='all'))],
                                    stack_regressors=False
                                    ):
        """
        Parameters
        ----------
        X: Array of shape (n_samples,n_features)
        Y: Array of shape (n_samples,)
        n_features_list : list of integers (optional, default=[100])
        n_jobs : integer (optional, default=1)
        feature_selectors : list of (name,transformer) tuples
        permutation_test : boolean (optional, default=False)
        stack_regressors : boolean (optional, default=False)
        """

        Res = []
        pipe_list = self.__create_pipelines(feature_selectors,n_features_list,stack_regressors)
        for (pipe,n_features) in tqdm(pipe_list,desc='Model loop for Single Assay'):
            start_time = datetime.now()
            if hasattr(pipe,'steps'):
                feature_selector = pipe.steps[1][0]
                regressor = pipe.steps[2][0]
                if regressor== 'GenRA':
                    X = np.array(X,dtype=bool)
                elif regressor == 'DNN':
                    X = X.astype(np.float32)
                    Y = Y.astype(np.int64)
                else:
                    X = X
                    Y = Y
            else:
                feature_selector = "Stacked"
                regressor = "Stacked Regressor"

            print('\n>>',"| Feature Selector:",feature_selector,"| N_features:",n_features,"| Regressor:",regressor)        
            score = cross_validate(
                                pipe, 
                                X, 
                                Y,
                                cv=self.k,
                                scoring=['r2', 'neg_root_mean_squared_error'],
                                n_jobs=n_jobs, 
                                verbose=0
                                )
            elapsed_time = datetime.now() - start_time
            SC = pd.DataFrame(score)

            SC.insert(0,'n_features',n_features)
            SC.insert(0,'Feature_selector',feature_selector)
            SC.insert(0,'rgr',regressor)
            Res.append(SC)
            print(f"Done in {self.prTime(elapsed_time)} h")

        Perf = pd.concat(Res)

        return Perf
        

    def predict_assays(self,
                        bio_act='C3.1',
                        bio_hitc0=0.7,
                        chm_fp='mrgn',
                        fileout_name='predict_output',
                        callback=None,
                        **kwargs
                        ):
        """
        Parameters
        ------------
        bio_act : string (optional, default='C3.1')
        bio_hitc0 : float (optional, default=0.7)
        chm_fp : string (optional, default='mrgn')
        fileout_name : string (optional, default='predict_output')
        callback : callable (optional, default=None) `calc_regressors_performance` callback function 
        **kwargs : Additional keyword arguments will be passed to callback

        """
        n_pos_min= 500
        DB5 = openMongo(db='genra_dev_v5',host='pb.epa.gov',auth=True)
        Assay_list = get_bio_assays(DB5.toxcast_assays)
        PERF = []
        j = 0
        for i, Asy in enumerate(tqdm(Assay_list)):
            print(f">> {Asy}")
            if chm_fp == "chemotypes":
                chm_dbc = DB5.chemotypes_fp
            elif chm_fp == "mrgn" or chm_fp == "httr":
                chm_dbc = DB5.chms_fp
                
            DS = get_BC_data(
                            bio_dbc=DB5.toxcast_fp,
                            bio_assay=Asy,
                            bio_fld='assay_component_name',
                            bio_act=bio_act,
                            bio_hitc0=bio_hitc0,
                            chm_dbc=chm_dbc,
                            chm_fp=chm_fp,
                            chm_fill=np.uint(0)
                            )
                
            if DS==None: continue
            n_pos, n_neg = (DS.bio[Asy]!=0).sum(),(DS.bio[Asy]==0).sum()
            print("  n_pos:{}\tn_neg:{}".format(n_pos,n_neg))
            if n_pos<n_pos_min: continue
            
            X,Y = DS.chm.values,DS.bio[Asy].to_numpy()
            hitc = bio_hitc0
            if bio_act == 'B1.1':
                hitc = 0
                
            if callback:
                P = callback(X,Y,**kwargs)
            else:
                P = self.calc_regressors_performance(X,Y)
            P = P.join(pd.DataFrame(dict(bio=Asy,val=bio_act,hitc0=hitc,n_pos=n_pos,n_neg=n_neg,chm=chm_fp),index=P.index))
            P1 = P.groupby(['bio','val','hitc0','chm','n_pos','n_neg','rgr','Feature_selector','n_features']).aggregate(dict(test_r2=[np.mean,np.std],
                                                test_neg_root_mean_squared_error=[np.mean,np.std]))\
                                .round(decimals=3)\
                                .reset_index()
            print(">>> Best ROC_AUC_mn:{}".format(P1.sort_values([('test_neg_root_mean_squared_error','mean')],ascending=False).head(1)[('test_neg_root_mean_squared_error','mean')].iloc[0]))

                
            PERF.append(P)
            
            
            if j > 0:
                P1.to_csv(f'{fileout_name}.csv', mode='a',header=False)
            else:
                P1.to_csv(f'{fileout_name}.csv', mode='a')

            j += 1

    @staticmethod
    def prTime(dt):
        hours, remainder = divmod(dt.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))

    def __str__(self):

        rgr_names = "_".join([i[0] for i in self.Est])

        return f"ML_Regressors_{rgr_names}_n_splits_{self.k}"
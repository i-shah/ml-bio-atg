import pandas as pd
import numpy as np
import pylab as pl
import scipy as sp
import sys
# import rpy2 
import os 
import copy
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.ensemble import (
                            GradientBoostingRegressor,
                            RandomForestRegressor,
                            StackingRegressor,
                            AdaBoostRegressor
                            )

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate, cross_val_predict
import time
# import xgboost as xgb
from tqdm import tqdm
from datetime import datetime
from functools import reduce 

# Maybe implement VotingRegressor 

def prTime(dt):
    hours, remainder = divmod(dt.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))

from genra.rax.skl.reg import *

np.random.seed(42)

Est1 = [ 
    # ('Linear Regression',LinearRegression()),
    ('Random Forest', RandomForestRegressor(random_state=42,max_depth=10)),
          ('Gradient Boosting', GradientBoostingRegressor()),
          ('SVR Linear', SVR(gamma='auto',kernel='linear')),
          # ('KNN',KNeighborsRegressor(metric='manhattan')),
          #('ANN1',MLPRegressor(solver='sgd')),
        # ('Adaboost', AdaBoostRegressor())
          ('GenRA', GenRAPredValue(n_neighbors=10,metric='jaccard')),
        # ('XGBoost',xgb.XGBRegressor()),
      ]

def calcRegressorPerf(X,Y,Est=Est1,k=5):
    Res = []
    for (LR,rgr) in Est:
        print('\n>>',LR)
        start_time = datetime.now()
        print(LR)
        if LR == 'GenRA':
            X = np.array(X,dtype=bool)
        if LR == 'DNN':
            X = X.astype(np.float32)
            Y = Y.astype(np.int64)
        score = cross_validate(rgr, X, Y,
                               cv=k,
                               scoring=['neg_root_mean_squared_error','r2',],
                               n_jobs=-1, verbose=0)
        elapsed_time = datetime.now() - start_time
        SC = pd.DataFrame(score)
        SC.insert(0,'LR',LR)
        
        Res.append(SC)
        print("Done in {} h".format(prTime(elapsed_time)))

    Perf = pd.concat(Res)
    
    return Perf

FS_est1 = [('f_regression',SelectKBest(f_regression,k='all'))] #Feature selector


def calcRegressorPipelinePerf(
                            X,
                            Y,
                            Est=Est1,
                            feature_selectors=FS_est1,
                            feature_step_range = [100],
                            k=5,
                            stack_regressors=False,
                            include_all_features=False
                            ):

    """
    X: array of fingerprints
    Y: array of bioactivity
    Est: list of tuples (name,sklearn.regression_model.object())
    feature_selectors: list of tuples (name,sklearn.feature_selector.object())
    feature_step-range: list of integers 
    k: int, number of cross-val folds
    stack_regressors: bool
    include_all: bool
    """
    Res = []
    pipe_list = []
    n_features_range = []
    n_features_range.extend(feature_step_range)
    if include_all_features:
        n_features_range.append('all')
    VT = ('Variance Threshold',VarianceThreshold(0)) # Eliminates any feature which all samples has same value
    for FS in feature_selectors:
        for n_features in n_features_range:
            FS[1].k = n_features
            stacked_rgrs = []
            for Rgr in Est:
                if Rgr[0] == 'GenRA':
                    X = np.array(X,dtype=bool)
                if Rgr[0] == 'DNN':
                        X = X.astype(np.float32)
                        Y = Y.astype(np.int64)
                pipe_rgr = Pipeline([VT,FS,Rgr])
                stacked_rgrs.append((f'pipe_{FS[0]}{Rgr[0]}_N:{n_features}',pipe_rgr))
                pipe_list.append((pipe_rgr,n_features))
            if stack_regressors:
                stacked_model = StackingRegressor(stacked_rgrs,final_estimator=RandomForestRegressor())
                pipe_list.append((stacked_model,n_features))
    
    for (pipe,n_features) in tqdm(pipe_list,desc='Model loop for Single Assay'):
        start_time = datetime.now()
        if hasattr(pipe,'steps'):
            feature_selector = pipe.steps[1][0]
            regressor = pipe.steps[2][0]
            pipe.steps[1][1].k = n_features 
        else:
            feature_selector = "Stacked"
            regressor = "Stacked Regressor"
        #not sure why but i have to re-set the k to what it should be (n_features)
        # K keeps getting changed in above but not sure why...
        # print(pipe)
        print('\n>>',"| Feature Selector:",feature_selector,"| N_features:",n_features,"| Regressor:",regressor)        
        score = cross_validate(
                            pipe, 
                            X, 
                            Y,
                            cv=k,
                            scoring=['neg_root_mean_squared_error','r2',],
                            n_jobs=-1, 
                            verbose=0
                            )
        elapsed_time = datetime.now() - start_time
        SC = pd.DataFrame(score)
        
        SC.insert(0,'n_features',n_features)
        SC.insert(0,'Feature_selector',feature_selector)
        SC.insert(0,'rgr',regressor)
        Res.append(SC)
        print("Done in {} h".format(prTime(elapsed_time)))

    Perf = pd.concat(Res)

    return Perf


# def calcRegressorPipelinePerf(
#                             X,
#                             Y,
#                             Est=Est1,
#                             FS=FS_est1,
#                             feature_step_size = 200,
#                             k=5,
#                             ):
#     Res = []
#     for rgr in Est:
#         print('\n>>',rgr[0])
#         if rgr[0] == 'GenRA':
#             X = np.array(X,dtype=bool)
            
#         n_features_max = X.shape[1]
#         n_features_range = np.arange(100,n_features_max,feature_step_size).tolist()
#         n_features_range.append('all')
#         for n_features in n_features_range:
#             start_time = datetime.now()
#             FS[1].k = n_features
#             VT = ('Variance Threshold',VarianceThreshold(0)) # Eliminates any feature which all samples has same value
#             pipe_rgr = Pipeline([VT,FS,rgr])
            
#             print(FS)
        
#             score = cross_validate(
#                                 pipe_rgr, 
#                                 X, 
#                                 Y,
#                                 cv=5,
#                                 scoring=['neg_root_mean_squared_error','r2',],
#                                 n_jobs=-1, 
#                                 verbose=0
#                                 )
#             elapsed_time = datetime.now() - start_time
#             SC = pd.DataFrame(score)
#             SC.insert(0,'n_features',n_features)
#             SC.insert(0,'Feature_selector',str(FS).split('(')[1])
#             SC.insert(0,'LR',rgr[0])
#             Res.append(SC)
#             print("Done in {} h".format(prTime(elapsed_time)))

#     Perf = pd.concat(Res)
    
#     return Perf
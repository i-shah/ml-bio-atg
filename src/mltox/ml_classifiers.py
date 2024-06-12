import pandas as pd
import numpy as np
import pylab as pl
import scipy as sp
import sys
# import rpy2 
import os 
from tqdm import tqdm
import copy
from sklearn.feature_selection import (
                                    SelectKBest,
                                    f_classif, 
                                    chi2, 
                                    VarianceThreshold
                                    )
from sklearn.model_selection import (
                                cross_validate, 
                                cross_val_predict, 
                                StratifiedKFold, 
                                permutation_test_score
                                )
from sklearn.ensemble import (
                    GradientBoostingClassifier, 
                    RandomForestClassifier, 
                    StackingClassifier,
                    AdaBoostClassifier
                    )
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
# import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
import time
from datetime import datetime
from functools import reduce 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
def prTime(dt):
    hours, remainder = divmod(dt.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))

from genra.rax.skl.cls import *
import itertools

# Maybe implement VotingClassifier

np.random.seed(42)
Est1 = [
        ('Logistic Regression', LogisticRegression(max_iter=300,n_jobs=-1)),
        ('Random Forest', RandomForestClassifier()),
          ('Gradient Boosting', GradientBoostingClassifier()),
          # ('Linear SVC', SVC(gamma='auto',probability=True,kernel='linear')),
        # ('RBF SVC', SVC(gamma='auto',probability=True,kernel='rbf')),
          # ('KNN',KNeighborsClassifier(metric='manhattan')),
          # ('ANN1',MLPClassifier(max_iter=500,solver='sgd')),
          ('GenRA', GenRAPredClass(n_neighbors=10,metric='jaccard')),
        # ('Adaboost',AdaBoostClassifier()),
        # ('XGBoost',xgb.XGBClassifier(n_jobs=-1)),
        ('Naive Bayes', GaussianNB()),
        # ('LDA', LinearDiscriminantAnalysis())
      ]

def calcClassifierPerf(X,Y,Est=Est1,k=5):
    Res = []
    for (LR,Clf) in Est:
        print('\n>>',LR)
        start_time = datetime.now()
        if LR == 'GenRA':
            X = np.array(X,dtype=bool)
        if LR == 'DNN':
            print(LR)
            X = X.astype(np.float32)
            Y = Y.astype(np.int64)
        score = cross_validate(Clf, X, Y,
                               cv=k,
                               scoring=['f1', 'recall', 'precision', 'roc_auc'],
                               n_jobs=-1, verbose=0)
        elapsed_time = datetime.now() - start_time
        SC = pd.DataFrame(score)
        SC.insert(0,'clf',LR)
        SC.insert(0,'n_features','all')

        Res.append(SC)
        print("Done in {} h".format(prTime(elapsed_time)))

    Perf = pd.concat(Res)
    
    return Perf

FS_est1 = [('ANOVA',SelectKBest(f_classif,k='all'))] #Feature selector


def calcClassifierPipelinePerf(
                            X,
                            Y,
                            Est=Est1,
                            feature_selectors=FS_est1,
                            feature_step_range = [100],
                            k=5,
                            permutation_test = False,
                            stack_classifiers=False
                            ):
    """
    X: array of fingerprints
    Y: array of bioactivity 
    Est: list of tuples (name,sklearn.classifier_model.object())
    feature_selectors: list of tuples (name,sklearn.feature_selector.object())
    feature_step-range: list of integers 
    k: int, number of cross-val folds
    stack_classifiers: bool
    """

    Res = []
    pipe_list = []
    n_features_range = feature_step_range
    VT = ('Variance Threshold',VarianceThreshold(0)) # Eliminates any feature which all samples has same value
    for FS in feature_selectors:
        for n_features in n_features_range:
            # check FS[1], FS[1] = ?
            FS[1].k = n_features   #WHY does FS change to largest value afterwards
            stacked_clfs = []
            for Clf in Est:
                pipe_clf = Pipeline([VT,copy.deepcopy(FS),Clf]) 
                stacked_clfs.append((f'pipe_{FS[0]}{Clf[0]}_N:{n_features}',pipe_clf))
                pipe_list.append((pipe_clf,n_features))
            if stack_classifiers:
                stacked_model = StackingClassifier(stacked_clfs,final_estimator=LogisticRegression())
                pipe_list.append((stacked_model,n_features))
    for (pipe,n_features) in tqdm(pipe_list,desc='Model loop for Single Assay'):
        start_time = datetime.now()
        if hasattr(pipe,'steps'):
            feature_selector = pipe.steps[1][0]
            pipe.steps[1][1].k = n_features
            classifier = pipe.steps[2][0]
        else:
            feature_selector = "Stacked"
            classifier = "Stacked Classifier"
        # print(pipe)
        if classifier == 'GenRA':
            X = np.array(X,dtype=bool)
        if classifier == 'DNN':
            X = X.astype(np.float32)
            Y = Y.astype(np.int64)
        print('\n>>',"| Feature Selector:",feature_selector,"| N_features:",n_features,"| Classifier:",classifier)        
        if permutation_test:
            pval = permutation_test_score(
                                        pipe, 
                                        X, 
                                        Y,
                                        cv = StratifiedKFold(k),
                                        n_permutations = 100,
                                        random_state = 42,
                                        scoring='f1',
                                        n_jobs=-1, 
                                        verbose=0
                                        )

        score = cross_validate(
                            pipe, 
                            X, 
                            Y,
                            cv=StratifiedKFold(k),
                            scoring=['f1', 'recall', 'precision','roc_auc'],
                            n_jobs=-1, 
                            verbose=0
                            )
        elapsed_time = datetime.now() - start_time
        SC = pd.DataFrame(score)
        if permutation_test:
            SC.insert(0,'pval',pval[-1])
        SC.insert(0,'n_features',n_features)
        SC.insert(0,'Feature_selector',feature_selector)
        SC.insert(0,'clf',classifier)
        Res.append(SC)
        print("Done in {} h".format(prTime(elapsed_time)))

    Perf = pd.concat(Res)

    return Perf


# def calcClassifierPipelinePerf(
#                             X,
#                             Y,
#                             Est=Est1,
#                             FS=FS_est1,
#                             feature_step_size = 200,
#                             k=5,
#                             permutation_test = False
#                             ):
#     Res = []
#     for Clf in Est:
#         print('\n>>',Clf[0])
#         if Clf[0] == 'GenRA':
#             X = np.array(X,dtype=bool)
            
#         n_features_max = X.shape[1]
#         n_features_range = np.arange(100,n_features_max,feature_step_size).tolist()
#         n_features_range.append('all')
#         for n_features in n_features_range:
#             start_time = datetime.now()
#             FS[1].k = n_features
#             VT = ('Variance Threshold',VarianceThreshold(0)) # Eliminates any feature which all samples has same value
#             pipe_clf = Pipeline([VT,FS,Clf])
            
#             print(FS)
#             if permutation_test:
#                 pval = permutation_test_score(
#                                             pipe_clf, 
#                                             X, 
#                                             Y,
#                                             cv = StratifiedKFold(5),
#                                             n_permutations = 100,
#                                             random_state = 42,
#                                             scoring='f1',
#                                             n_jobs=-1, 
#                                             verbose=0
#                                             )

#             score = cross_validate(
#                                 pipe_clf, 
#                                 X, 
#                                 Y,
#                                 cv=StratifiedKFold(5),
#                                 scoring=['f1', 'recall', 'precision'],
#                                 n_jobs=-1, 
#                                 verbose=0
#                                 )
#             elapsed_time = datetime.now() - start_time
#             SC = pd.DataFrame(score)
#             if permutation_test:
#                 SC.insert(0,'pval',pval[-1])
#             SC.insert(0,'n_features',n_features)
#             SC.insert(0,'Feature_selector',str(FS).split('(')[1])
#             SC.insert(0,'LR',Clf[0])
#             Res.append(SC)
#             print("Done in {} h".format(prTime(elapsed_time)))

#     Perf = pd.concat(Res)
    
#     return Perf


# def calcClassifierStackingPerf(
#                             X,
#                             Y,
#                             Est=clf_combinations,
#                             k=5,
#                             ):
#     Res = []
    
#     combos = [combo for combo in itertools.combinations(Est1, 2)]

#     for i in range(len(combos)):
#         combo_name = combos[i][0][0] +','+ combos[i][1][0]
#         print('\n>>',combo_name)
#         start_time = datetime.now()
        
#         stack_clf = StackingClassifier(estimators=combos[i],final_estimator=LogisticRegression())

#         score = cross_validate(
#                             stack_clf, 
#                             X, 
#                             Y,
#                             cv=StratifiedKFold(5),
#                             scoring=['f1', 'recall', 'precision'],
#                             n_jobs=-1, 
#                             verbose=0
#                             )
#         elapsed_time = datetime.now() - start_time
#         SC = pd.DataFrame(score)
#         SC.insert(0,'LR',combo_name)
#         Res.append(SC)
#         print("Done in {} h".format(prTime(elapsed_time)))

#     Perf = pd.concat(Res)
    
#     return Perf
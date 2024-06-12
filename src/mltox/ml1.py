import pandas as pd
import numpy as np
import pylab as pl
import scipy as sp
import sys
import rpy2 
import os 

from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate, cross_val_predict
import time
from datetime import datetime
from functools import reduce 

def prTime(dt):
    hours, remainder = divmod(dt.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))

#from genra.rax.skl.cls import *

Est1 = [('Random Forest', RandomForestClassifier(random_state=42)),
          ('Gradient Boosting', GradientBoostingClassifier()),
          ('SVC', SVC(gamma='auto',probability=True,kernel='linear')),
          ('KNN',KNeighborsClassifier(metric='manhattan')),
          #('ANN1',MLPClassifier(solver='sgd')),
          #('GenRA', GenRAPredClass(n_neighbors=10,metric='jaccard'))
      ]

def calcMLPerf(X,Y,Est=Est1,k=5):
    Res = []
    for (LR,Clf) in Est:
        print('\n>>',LR)
        start_time = datetime.now()
        score = cross_validate(Clf, X, Y,
                               cv=k,
                               scoring=['f1', 'recall', 'precision'],
                               n_jobs=-1, verbose=0)
        elapsed_time = datetime.now() - start_time
        SC = pd.DataFrame(score)
        SC.insert(0,'LR',LR)
        
        Res.append(SC)
        print("Done in {} h".format(prTime(elapsed_time)))

    Perf = pd.concat(Res)
    
    return Perf

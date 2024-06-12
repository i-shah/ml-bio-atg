import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
import torch
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import (
                                cross_validate, 
                                StratifiedKFold, 
                                permutation_test_score
                                )
from sklearn.ensemble import (
                    GradientBoostingClassifier, 
                    RandomForestClassifier, 
                    StackingClassifier,
                    )
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from datetime import datetime
from functools import reduce 
from genra.rax.skl.cls import *
from mltox.db.mongo import *
from mltox.db.bc import *
from mltox.db.bio import *

#TODO: Implement seed for random, torch, np
class MLClassifiers():

    """
    Parameters:
    -----------

    Est: list of (name,estimator) tuples, (optional)
    """
    def __init__(self,
                Est = [
                    ('Logistic Regression', LogisticRegression(max_iter=300,n_jobs=8)),
                    ('Random Forest', RandomForestClassifier(n_jobs=8)),
                    ('Gradient Boosting', GradientBoostingClassifier()),
                    ('GenRA', GenRAPredClass(n_neighbors=10,metric='jaccard',n_jobs=8)),
                    ('Naive Bayes', GaussianNB()),
                ],
                ):
        self.Est = Est

    def __create_pipelines(self,
                        feature_selectors,
                        n_features_list,
                        stack_classifiers):
        pipe_list = []
        VT = ('Variance Threshold',VarianceThreshold(0)) # Eliminates any feature which all samples has same value
        for FS in feature_selectors:
            for n_features in n_features_list:
                FS[1].k = n_features 
                stacked_clfs = []
                for Clf in self.Est:
                    pipe_clf = Pipeline([VT,copy.deepcopy(FS),Clf]) 
                    stacked_clfs.append((f'pipe_{FS[0]}{Clf[0]}_N:{n_features}',pipe_clf))
                    pipe_list.append((pipe_clf,n_features))
                if stack_classifiers:
                    stacked_model = StackingClassifier(stacked_clfs,final_estimator=LogisticRegression())
                    pipe_list.append((stacked_model,n_features))

        return pipe_list
                

    def calc_classifiers_performance(self,
                                    X,
                                    Y,
                                    n_features_list=[100],
                                    n_jobs=1,
                                    k=5,
                                    seed=None,
                                    shuffle=False,
                                    feature_selectors=[('ANOVA',SelectKBest(f_classif,k='all'))],
                                    permutation_test = False,
                                    stack_classifiers=False
                                    ):
        """
        Parameters
        ----------
        X: Array of shape (n_samples,n_features)
        Y: Array of shape (n_samples,)
        n_features_list : list of integers (optional, default=[100])
        n_jobs : integer (optional, default=1)
        k : integer (optional, default=5), Number of splits for CV
        seed : integer (optional, default=None)
        shuffle : boolean (optional, default=False)
        feature_selectors : list of (name,transformer) tuples
        permutation_test : boolean (optional, default=False)
        stack_classifiers : boolean (optional, default=False)
        """
        Res = []
        pipe_list = self.__create_pipelines(feature_selectors,n_features_list,stack_classifiers)
        for (pipe,n_features) in tqdm(pipe_list,desc='Model loop for Single Assay'):
            start_time = datetime.now()
            if hasattr(pipe,'steps'):
                feature_selector = pipe.steps[1][0]
                classifier = pipe.steps[2][0]
                if classifier== 'GenRA':
                    Xt = np.array(X,dtype=bool)
                    Yt = Y
                elif classifier == 'DNN':
                    Xn = copy.deepcopy(X).astype(np.float32)
                    Yn = copy.deepcopy(Y).astype(np.int64)
                    Xt = torch.from_numpy(Xn)
                    Yt = torch.from_numpy(Yn)
                    print(type(Xt))
                else:
                    Xt,Yt = X,Y

            else:
                feature_selector = "Stacked"
                classifier = "Stacked Classifier"

            print('\n>>',"| Feature Selector:",feature_selector,"| N_features:",n_features,"| Classifier:",classifier)        
            if permutation_test:
                pval = permutation_test_score(
                                            pipe, 
                                            X, 
                                            Y,
                                            cv = StratifiedKFold(n_splits=k,shuffle=shuffle),
                                            n_permutations = 100,
                                            random_state = seed,
                                            scoring='f1',
                                            n_jobs=-1, 
                                            verbose=0
                                            )

            score = cross_validate(
                                pipe, 
                                X, 
                                Y,
                                cv=StratifiedKFold(n_splits=k,random_state=seed,shuffle=shuffle),
                                scoring=['f1', 'recall', 'precision','roc_auc'],
                                n_jobs=n_jobs, 
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
            print(f"Done in {self.prTime(elapsed_time)} h")

        Perf = pd.concat(Res)

        return Perf
        

    def predict_assays(self,
                        bio_act='B1.1',
                        bio_hitc0=0.7,
                        chm_fp='mrgn',
                        fileout_name='predict_output',
                        callback=None,
                        **kwargs
                        ):
        """
        Parameters
        ------------
        bio_act : string (optional, default='B1.1')
        bio_hitc0 : float (optional, default=0.7)
        chm_fp : string (optional, default='mrgn')

        fileout_name : string (optional, default='predict_output')
        callback : callable (optional, default=None) `calc_classifiers_performance` callback function 
        **kwargs : Additional keyword arguments will be passed to callback

        """
        n_pos_min= 500
        DB5 = openMongo(db='genra_dev_v5',host='pb.epa.gov',auth=True)
        Assay_list = get_bio_assays(DB5.toxcast_assays)
        PERF = []
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
            n_pos, n_neg = (DS.bio[Asy]==1).sum(),(DS.bio[Asy]==0).sum()
            print("  n_pos:{}\tn_neg:{}".format(n_pos,n_neg))
            if n_pos<n_pos_min: continue
            
            X,Y = DS.chm.values,DS.bio[Asy].to_numpy()
            hitc = bio_hitc0
            if bio_act == 'B1.1':
                hitc = 0
                
            if callback:
                P = callback(X,Y,**kwargs)
            else:
                P = self.calc_classifiers_performance(X,Y)
            P = P.join(pd.DataFrame(dict(bio=Asy,val=bio_act,hitc0=hitc,n_pos=n_pos,n_neg=n_neg,chm=chm_fp),index=P.index))
            P1 = P.groupby(['bio','val','hitc0','chm','n_pos','n_neg','clf','Feature_selector','n_features']).aggregate(dict(test_f1=[np.mean,np.std],
                                                test_recall=[np.mean,np.std],
                                                test_precision=[np.mean,np.std],
                                                test_roc_auc=[np.mean,np.std]))\
                                .round(decimals=3)\
                                .reset_index()
            print(">>> Best ROC_AUC_mn:{}".format(P1.sort_values([('test_roc_auc','mean')],ascending=False).head(1)[('test_roc_auc','mean')].iloc[0]))

                
            PERF.append(P)
            
            if os.path.exists(os.getcwd()+f'/{fileout_name}.csv'):
                P1.to_csv(f'{fileout_name}.csv', mode='a',header=False)
            else:
                P1.to_csv(f'{fileout_name}.csv', mode='a')


    @staticmethod
    def prTime(dt):
        hours, remainder = divmod(dt.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))

    def __str__(self):

        clf_names = "_".join([i[0] for i in self.Est])

        return f"ML_Classifiers_{clf_names}"
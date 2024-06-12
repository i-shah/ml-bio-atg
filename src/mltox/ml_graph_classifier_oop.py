import deepchem as dc
import torch
import numpy as np
import random
import pandas as pd
import glob
import os
from openTSNE import TSNE
from sklearn.neighbors import NearestNeighbors

from ..db.chm import get_chm_FP
from ..db.graph_utils import split_data, get_smiles
from ..db.mongo import openMongo

class GraphModel():
    """
    model: ModifiedAttentiveFP

    seed: int, default=42

    dataset:  tuple, (`dc.data.NumpyDataset` object,`dc.data.NumpyDataset` object)

    assay_list: list[str]
    """

    def __init__(self,model,assay_list,dataset,seed=42):
        self.model = model
        self.assay_list = assay_list
        self.dtxset , self.dataset = dataset
        self.train_dataset, self.test_dataset = split_data(dataset[1],seed)
        self.train_z, self.test_z = None, None
        self.test_predictions = None
        self.unlabeled_dataset = None
        self.unlabeled_preds = None
        self.unlabeled_embeds = None
        self.seed = seed
        self.training = model.model.training

    
    def run(self):

        """

        Returns
        -------
        y_pred: np.ndarray 

        results: `pd.DataFrame` object

        """


        n_tasks = self.train_dataset.y.shape[1]
        if self.training:
            print("Training model...")
            self.model.fit(self.train_dataset)
            print("Done.")
        else:
            print('Loading pre-trained model...')
        
        
        self.training = False
        for f in glob.glob("/tmp/__autograph_generated_file*"):
            os.remove(f)

        test_true = self.test_dataset.y
        test_pred = self.model.predict(self.test_dataset)
        metric = dc.metrics.roc_auc_score
        scores = []
        for i in range(n_tasks):
            score = metric(dc.metrics.to_one_hot(test_true[:,i]), test_pred[:,i])
            scores.append([self.assay_list[i],score])

        results = pd.DataFrame(scores)
        self.test_predictions = test_pred
        return test_pred, results

    def get_embeds(self):
        """
        Parameters
        ----------

        split : str
        Returns
        -------

        training_embeds: 

        test_embeds: 
        
        """
        if self.training:
            raise ValueError("Model is not yet trained. Must run model first!")

        n = self.train_dataset.X.shape[0]
        training_embeds = self.model.predict_embedding(self.train_dataset)[:n,:]
        m = self.test_dataset.X.shape[0]
        test_embeds = self.model.predict_embedding(self.test_dataset)[:m,:]
        return training_embeds, test_embeds

    def get_tsne(self):
            """
            Parameters
            ----------
            dataset: `dc.data.NumpyDataset` object

            Returns
            -------
            train_z: np.ndarray
                    Embedding of the training data in low-dimensional space.

            test_z: np.ndarray
                    Embedding of the test data in low-dimensional space.

            """
            if self.training:
                raise ValueError("Model is not yet trained. Must run model first!")

            tsne = TSNE(    
                metric="euclidean",
                n_jobs=8,
                random_state=self.seed,
                verbose=False,
            )


            train_embeds, test_embeds = self.get_embeds()

            train_z = tsne.fit(train_embeds)
            test_z = train_z.transform(test_embeds)
            self.train_z = train_z
            self.test_z = test_z
            return train_z, test_z
        
    def predict_unlabeled(self,unlabeled_data):
        """
        Parameters
        ----------

        Returns
        -------

        preds: pd.DataFrame


        """

        if self.training:
            raise Exception("Model is not yet trained. Must run model first!")

        
        if self.train_z is None:
            self.get_tsne()


        df = unlabeled_data.copy()

        df['smiles'] = df['DTXSID'].apply(get_smiles)
        df = df[df['smiles']!='NaN']

        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        smiles = df['smiles'].tolist()
        n_chemicals = df.shape[0]
        n_tasks = len(self.assay_list)

        dummy_labels = np.zeros((n_chemicals,n_tasks))
        X = featurizer.featurize(smiles)
        unlabeled_dataset = dc.data.NumpyDataset(X=X, y=dummy_labels)
        
        unlabeled_preds = self.model.predict(unlabeled_dataset)

        unlabeled_embedding = self.model.predict_embedding(unlabeled_dataset)

        
        untested_z = self.train_z.transform(unlabeled_embedding)
        df["comp-1"] = untested_z[:,0]
        df["comp-2"] = untested_z[:,1]

        self.unlabeled_dataset = df
        self.unlabeled_preds = unlabeled_preds
        self.unlabeled_embeds = unlabeled_embedding
        return unlabeled_preds, unlabeled_embedding, df

    def get_unlabeled_assay_data(self,assay):
        if self.unlabeled_preds is not None:
            assay_n = self.assay_list.index(assay)
            data = self.unlabeled_dataset
            data[f'probability active {assay}']  = self.unlabeled_preds[:,assay_n][:,1]
        else:
            raise ValueError('No Predictions available. Please run model!')

        return data

    def _get_features(self,assay,mode='all',features='mrgn'):
        DB5 = openMongo(db='genra_dev_v5',host='pb.epa.gov',auth=True)
        train_dtxset, _ = split_data(self.dtxset,seed=self.seed)
        unlabeled_data = self.unlabeled_dataset 

        if features == 'mrgn':
            train_fps  = get_chm_FP(
                        train_dtxset.X,
                        DB5.chms_fp,
                        fp='mrgn',
                        fill=np.uint(0)
                            )
            
            unlabeled_fps  = get_chm_FP(
                        unlabeled_data.DTXSID,
                        DB5.chms_fp,
                        fp='mrgn',
                        fill=np.uint(0)
                            )

        elif features == 'graph':
            train_fps, _ = self.get_embeds()
            train_dtxset, _ = split_data(self.dtxset,seed=self.seed)
            train_fps = pd.DataFrame(train_fps)
            train_fps.insert(0,column='dsstox_sid',value=train_dtxset.X)

            unlabeled_fps = pd.DataFrame(self.unlabeled_embeds)
            unlabeled_fps.insert(0,column='dsstox_sid',value=unlabeled_data['DTXSID'])

        else: 
            raise ValueError('Invalid feature type! \n Valid types: `graph` or `mrgn`')

        assay_n = self.assay_list.index(assay)
        train_fps['label'] = train_dtxset.y[:,assay_n]

        train_labels = train_fps[['dsstox_sid','label']]

        if mode == 'positive':
            train_fps = train_fps[train_fps['label']==1].iloc[:,:-1]
        elif mode == 'negative':
            train_fps = train_fps[train_fps['label']==0].iloc[:,:-1]
        elif mode == 'all':
            train_fps = train_fps.iloc[:,:-1]
        else:
            raise ValueError('Invalid mode: Valid Modes: `positive`, `negative`, `all`')

        data = unlabeled_data[['DTXSID']].rename(columns={'DTXSID':'dsstox_sid'})
        fps = data.merge(unlabeled_fps,on='dsstox_sid',how='left')
        tmp = pd.concat([train_fps,fps]).fillna(0)

        unlabel_fps = tmp.iloc[train_fps.shape[0]:,:]
        train_fps = tmp.iloc[:train_fps.shape[0],:]

        
        return train_fps, train_labels, unlabel_fps
    
    def get_neighbors(self,
                      assay,
                      mode,
                      n_neighbors,
                      metric,
                      features):
        """
        Parameters
        ----------
        assay: str

        mode: str
        
        n_neighbors: int  

        Integer representing number of neighbors

        metric: str

        features: str

        """
        if features == 'graph' and metric == 'jaccard':
            raise ValueError('Metric for `graph` features cannot be `Jaccard`')

        train_fps, train_labels, unlabeled_fps = self._get_features(assay,mode,features)        
        neigh = NearestNeighbors(n_neighbors=n_neighbors,metric=metric)
        neigh.fit(train_fps.iloc[:,1:])
        distances,indices = neigh.kneighbors(unlabeled_fps.iloc[:,1:])

        train_dtxset, _ = split_data(self.dtxset,seed=self.seed)
        df_z = pd.DataFrame(self.train_z,columns=['comp-1','comp-2'])
        df_z.insert(0,column='DTXSID',value=train_dtxset.X)

        df = self.unlabeled_dataset[['DTXSID']].copy()

        for i in range(n_neighbors):
            df[f'distance neighbor {i+1}'] = distances[:,i]
            df[f'train index neighbor {i+1}'] = indices[:,i]
            df[f'train dtxsid neighbor {i+1}'] = df[f'train index neighbor {i+1}'].apply(lambda x: train_fps[['dsstox_sid']].loc[x].values[0])
            df[f'train label neighbor {i+1}'] = df[f'train dtxsid neighbor {i+1}'].apply(lambda x: train_labels[train_labels['dsstox_sid']==x]['label'].values[0])

            tmp = df_z.rename(columns={
                        'DTXSID':f'train dtxsid neighbor {i+1}',
                        'comp-1':f'comp-1 neighbor {i+1}',
                        'comp-2':f'comp-2 neighbor {i+1}'
                        })
            df = df.merge(tmp,on=f'train dtxsid neighbor {i+1}',how='left')

        return df
            
    def make_table(self, 
                   assay,
                   mode='all',
                   compare=True,
                   n_neighbors=1,
                   metric='euclidean',
                   features='graph'):

        if self.unlabeled_preds is None:
            raise ValueError("No prediction for unlabeled data found.")

        def assign_label(x):
            if x > 0.5:
                return 1
            elif x < 0.5:   
                return 0

        def prob_predict(x):
            if x > 0.5:
                return x
            else:
                return 1-x

        preds = self.get_unlabeled_assay_data(assay)

        df_preds = preds[['DTXSID',f'probability active {assay}']].copy()
        df_preds[f'predicted label {assay}'] = df_preds[f'probability active {assay}'].apply(assign_label)
        df_preds[f'probability label {assay}'] = df_preds[f'probability active {assay}'].apply(prob_predict)
        df_preds['comp-1'] = preds['comp-1']
        df_preds['comp-2'] = preds['comp-2']

        if compare:
            df_neighbors = self.get_neighbors(assay,mode,n_neighbors,metric,features)
            
            df_preds = df_preds.merge(df_neighbors,on='DTXSID')

        return df_preds
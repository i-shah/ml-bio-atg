import deepchem as dc
import torch
import numpy as np
import random
import pandas as pd
import glob
import os
from openTSNE import TSNE
from ..db.graph_utils import *


def run_model(train_dataset,test_dataset,assay_list,model,seed=42):
    """
    Parameters
    ----------
    train_dataset: `dc.data.NumpyDataset` object

    test_dataset: `dc.data.NumpyDataset` object

    assay_list: list[str]

    seed: int, default=42

    Returns
    -------
    y_pred: np.ndarray 

    results: `pd.DataFrame` object

    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    n_tasks = train_dataset.y.shape[1]
    model = model
    model.fit(train_dataset)

    for f in glob.glob("/tmp/__autograph_generated_file*"):
        os.remove(f)

    y_true = test_dataset.y
    y_pred = model.predict(test_dataset)
    metric = dc.metrics.roc_auc_score
    scores = []
    for i in range(n_tasks):
        score = metric(dc.metrics.to_one_hot(y_true[:,i]), y_pred[:,i])
        scores.append([assay_list[i],score])

    results = pd.DataFrame(scores)
    return y_pred, results, model

def get_tsne(train_dataset,test_dataset,model):
    """
    Parameters
    ----------
    train_dataset:

    test_dataset:

    Returns
    -------
    train_z: np.ndarray
            Embedding of the training data in low-dimensional space.

    test_z: np.ndarray
            Embedding of the test data in low-dimensional space.

    """
    n = train_dataset.X.shape[0]
    training_embeds = model.predict_embedding(train_dataset)[:n,:]
    tsne = TSNE(    
        metric="euclidean",
        n_jobs=8,
        random_state=42,
        verbose=False,
    )
    train_z = tsne.fit(training_embeds)

    m = test_dataset.X.shape[0]
    test_embeds = model.predict_embedding(test_dataset)[:m,:]
    test_z = train_z.transform(test_embeds)
    return train_z, test_z

def predict_unlabeled(assay,assay_list,df,model,train_z):
    """
    Parameters
    ----------

    df: 

    model: `dc.models` object

    train_z: np.ndarray

    Returns
    -------

    df:


    """

    df['smiles'] = df['DTXSID'].apply(get_smiles)

    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    smiles = df['smiles'].tolist()
    n_chemicals = df.shape[0]
    n_tasks = len(assay_list)
    assay_n = assay_list.index(assay)

    dummy_labels = np.zeros((n_chemicals,n_tasks))
    X = featurizer.featurize(smiles)
    dataset = dc.data.NumpyDataset(X=X, y=dummy_labels)
    
    y_pred = model.predict(dataset)

    tsca_embedding = model.predict_embedding(dataset)
    untested_z = train_z.transform(tsca_embedding)

    df[f'probability active {assay}']  = y_pred[:,assay_n][:,1]
    df["comp-1"] = untested_z[:,0]
    df["comp-2"] = untested_z[:,1]

    return df

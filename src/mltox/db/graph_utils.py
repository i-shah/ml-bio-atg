import pandas as pd
import random 
from .mongo import *
from .bc import *
from .bio import *
from .chm import *
import functools
import deepchem as dc


def vote(x):
    votes = list(x)
    if sum(votes)/len(votes) < 0.5:
        return 0
    else:
        return 1

def get_smiles(DB,dsstox_sid):
    try:
        smiles = DB.compounds.find_one({'dsstox_sid':dsstox_sid})['smiles']
        return smiles
    except:
        return 'NaN'

def get_chm_name(DB,dsstox_sid):
    name = DB.compounds.find_one({'dsstox_sid':dsstox_sid})['name']
    return name

def get_bio_data(DB,Assay_list,keep_dtx=False):
    dfs = []
    asy_list = []
    for asy in Assay_list:
        print("Loading {}".format(asy))
        Y1 = get_bio_activities(DB.toxcast_fp,
                                assay=asy,
                                fld='assay_component_name',
                                h0='0.7',
                                val='B1.1',
                                full=False
                                )
        if Y1.shape[0]==0: continue
        Y = Y1.groupby('dsstox_sid').agg(lambda x: vote(x)).reset_index()
        dfs.append(Y)
        asy_list.append(asy)
            
    df_final = functools.reduce(lambda left,right: pd.merge(left,right,on='dsstox_sid',
                                                            how='outer'), dfs).fillna(-1)

    cols = [asy for asy in asy_list]
    cols.insert(0,'dsstox_sid')
    Y1 = df_final[cols]

    return Y1

def get_chm_data(DB,SID,fp='mrgn'):    
    X1 = get_chm_FP(SID,DB.chms_fp,fp=fp,fill=np.uint(0))
    X1['smiles'] = X1['dsstox_sid'].apply(lambda i: get_smiles(DB,i))
    X1 = X1.dropna(subset='smiles')

    return X1
    
    # SID = set(X1.dsstox_sid.to_list()).intersection(Y1.dsstox_sid)

    # XY1 = X1.merge(Y1,on='dsstox_sid',how='right')
    # XY1 = XY1[XY1.dsstox_sid.isin(SID)]
    
    
    # X = XY1[X1.columns[X1.columns=='smiles']]
    # Y = XY1[Y1.columns[Y1.columns!='dsstox_sid']]

    # if keep_dtx:
    #     X = XY1[X1.columns]
    #     Y = XY1[Y1.columns]

    # return X,Y

def assign_missing_values(X,counts):
    p_hit = counts[1] /(counts[0] + counts[1])
    p_nohit = 1 - p_hit
    if X == -1:
         X = random.choices((1,0),weights=[p_hit,p_nohit],k=1)[0]
    return X

def replace_missing_data(y,mode):
    y_new = y.loc[:, ~(y== -1.0).sum()>-y.shape[0]*0.1]   #drop columns with > 10% missing data
    if mode == 'impute':
        for column in y_new:
            if (y_new[column]==-1).any() and not (y_new[column]==-1).all():
                counts = y_new[column].value_counts()
                y_new[column] = y_new[column].apply(lambda x: assign_missing_values(x,counts))
    elif mode == 'fill':
        y_new = y_new.replace(-1,0)
    elif mode == None:
        pass
    return y_new.apply(np.int64)


def get_type(X):
    return str(type(X))

def task_to_graph(X,Y,featurizer,dtx=False):
    X1 = X[['dsstox_sid','smiles']].copy()

    featurizer = featurizer

    X1['features'] = featurizer.featurize(X1['smiles'])


    X1['type'] = X1['features'].apply(get_type)

    X1 = X1[X1['type']!="<class 'numpy.ndarray'>"].copy()
    X_new = X1[['dsstox_sid','smiles','features']].copy()

    XY_new = pd.merge(X_new,Y,how='inner',on='dsstox_sid')

    X_graph = XY_new['features'].tolist() #change to features

    Y_graph = XY_new.iloc[:,3:]

    if dtx:
        X_dtx = XY_new['dsstox_sid'].tolist() #change to features
        return X_dtx, X_graph, Y_graph, XY_new[['features','smiles']]
    else:
        return X_graph, Y_graph, XY_new[['features','smiles']]


def get_dataset(DB,prefix=None,deepchem=True):
    """

    Parameters
    -----------
    prefix: str or None (default: None)

    Ex: ATG, TOX21

    None uses entire assay list

    Returns
    -------
    subset_assay: list[str]
    
    dataset: `dc.data.NumpyDataset` object


    """

    Assay_list = get_bio_assays(DB.toxcast_assays)
    prefixes = set([i.split('_')[0] for i in Assay_list])

    if prefix:
        subset_assays = [i for i in Assay_list if i.startswith(prefix)]
    else:
        subset_assays = Assay_list

    if os.path.exists(f'/{prefix}_x_with_dtx.pkl') and os.path.exists(f'/{prefix}_y_with_dtx.pkl'):
        X = pd.read_pickle(f'/{prefix}_x_with_dtx.pkl')
        Y = pd.read_pickle(f'/{prefix}_y_with_dtx.pkl')
    else:
        X,Y = get_task_data(subset_assays,keep_dtx=True)
        # X.to_pickle(f'/{prefix}_x_with_dtx.pkl')
        # Y.to_pickle(f'/{prefix}_y_with_dtx.pkl')

        
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

    dtx,X1,Y1,smiles = task_to_graph(X,Y,featurizer,dtx=True)

    Y_new = replace_missing_data(Y1,"fill")
    assay_list = Y_new.columns.tolist()
    
        
    if not deepchem:
        return (dtx,X1,Y_new), assay_list


    dataset = dc.data.NumpyDataset(X=X1,y=Y_new)
    dtx_set = dc.data.NumpyDataset(X=dtx,y=Y_new)  #MODIFY

    
    return (dtx_set, dataset), assay_list


def split_data(dataset,seed=42):
    """
    dataset: `dc.data.NumpyDataset` object

    seed: int, default=42
    """
    splitter = dc.splits.RandomStratifiedSplitter()

    train_dataset, _, test_dataset = splitter.train_valid_test_split(
                                                                    dataset,
                                                                    frac_train=0.8,
                                                                    frac_test=0.2,
                                                                    frac_valid=0.0,
                                                                    seed=seed,               
                                                                    test_dir=f'{os.getcwd()}/tmp_test_data/',
                                                                    train_dir=f'{os.getcwd()}/tmp_train_data/',
                                                                    valid_dir=f'{os.getcwd()}/tmp_valid_data/'
                                                                    )

    return train_dataset, test_dataset
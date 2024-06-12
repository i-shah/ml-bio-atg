import numpy as np
import pandas as pd
import copy
import pymongo
import sys
import re

def get_chm_FP(SID,dbc,fp='mrgn',fill=None):

    Agg = [
            # Match chemicals in cluster
            {'$match': {
                     'dsstox_sid':{'$in':list(SID)}}
            },
            # Include these fields
            {'$project':{'dsstox_sid':1,'_id':0,
                        'fp':'${}.ds'.format(fp)}
            },
            # Unwind the fp 
            {'$unwind':"$fp"}
            ]

    X = pd.DataFrame(dbc.aggregate(Agg,allowDiskUse=True))
    if X.shape[0]==0:
        return X
    X.insert(0,'val',np.uint(1))

    #if X.shape[0]: return

    return X.pivot_table(index=['dsstox_sid'],columns='fp',values='val',fill_value=fill)\
            .reset_index()\
            .rename_axis(columns=None)


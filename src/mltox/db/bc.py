
import numpy as np
import pandas as pd
import copy
import pymongo
import sys

from .bio import *
from .chm import *
from box import Box
def get_BC_data(
                bio_dbc,
                chm_dbc,
                bio_assay,
                bio_fld='assay_component_name',
                bio_hitc0=0.7,
                bio_act='B1.1',
                chm_fp = 'mrgn', 
                chm_fill=0,
                drop_duplicates=True
                ):
    
    Y1 = get_bio_activities(
                            bio_dbc,
                            assay=bio_assay,
                            fld=bio_fld,
                            h0=bio_hitc0,
                            val=bio_act
                            )
    try:
        X1 = getChmFP(
                Y1.dsstox_sid,
                chm_dbc,
                fp=chm_fp,
                fill=chm_fill
                )
    except:
        return None
    if X1.shape[0]==0: return None

    SID = set(X1.dsstox_sid.to_list()).intersection(Y1.dsstox_sid)

    XY1 = X1.merge(Y1,on='dsstox_sid',how='right')
    XY1 = XY1[XY1.dsstox_sid.isin(SID)]

    return Box(chm=XY1[X1.columns[X1.columns!='dsstox_sid']],
                bio=XY1[Y1.columns[Y1.columns!='dsstox_sid']])
    
    

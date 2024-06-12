import numpy as np
import pandas as pd
import copy
import pymongo
import sys
import functools as ft

from datetime import datetime

def get_bio_sources(dbc,fld='assay_source_name'):
    return list(dbc.distinct(fld))

def get_bio_assays(dbc,src=None,fld='assay_component_name'):
    Q = {}
    if src: Q['assay_source_name']=src
    return list(dbc.find(Q).distinct(fld))
     
# def getPot(AC50):
#     if AC50 > 0:
#         return (6-np.log10(AC50))
#     else:
#         return float('NaN')

def get_bio_activities(dbc,assay,fld='assay_component_name',val='B1.1',h0=0.7,full=False):
    Fits = get_bio_responses(dbc,assay,fld)
    try:
        if val=='B1.1':
            Fits.insert(Fits.shape[1],'value',Fits.modl!='cnst')
            Fits.loc[:,'value']=Fits.value.astype(np.uint)
        elif val=='B1.2':
            Fits.insert(Fits.shape[1],'value',(Fits.hitc>=h0) & (Fits.modl!='cnst'))
            Fits.loc[:,'value']=Fits.value.astype(np.uint)
        elif val=='C3.1':
            AC50 = Fits.modl_ga.fillna(0)
            Fits.insert(Fits.shape[1],'value',AC50)
        elif val=='C3.2':
            AC50 = Fits.modl_ga.copy()
            AC50.loc[(Fits.hitc<h0) | (Fits.modl=='cnst')] = 0      
            Fits.insert(Fits.shape[1],'value',AC50)
        else:
            raise ValueError( "Invalid Value!")
    except:
        return None
    if full:
        Res = Fits.rename(columns=dict(value=assay))
    else:
        Res = Fits[['dsstox_sid','value']].rename(columns=dict(value=assay))

    return Res

    
def get_bio_responses(dbc,assay,fld='assay_component_name'):
    A = [{'$match':{'fits.{}'.format(fld):assay}},
         {'$project':{
            'dsstox_sid':1,'_id':0,
            'fits1':{
                '$filter':{
                        'input':'$fits',
                        'as': 'fit',
                        'cond':{'$eq':['$$fit.{}'.format(fld),assay]}
                    }
                }
            }
         },
         {'$unwind':{'path':'$fits1'}},     
         {'$replaceRoot':{
             'newRoot':{
                 '$mergeObjects' :[{'dsstox_sid':'$dsstox_sid'},'$fits1']
                 }
             }
         }
         ]
    Fits = pd.DataFrame(dbc.aggregate(A))    
    
    return Fits

def get_bio_data(dbc,assays,fld='assay_component_name'):
    X = [get_bio_activities(dbc,assay=asy,fld=fld,
                            h0='0.7',val='B1.1',full=False) 
         for asy in assays]
    return ft.reduce(lambda left,right: pd.merge(left,right,on='dsstox_sid',
                                                 how='left'), X).fillna(-1)
    
    

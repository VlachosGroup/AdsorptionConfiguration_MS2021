# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:40:53 2021

@author: user
"""
import json
from collections import defaultdict
from rdkit import Chem
import numpy as np
from tqdm import tqdm
import itertools
from itertools import cycle

import json
import os
import matplotlib.pyplot as plt
from glob import glob
import pickle
from util import RemoveHe, GetBondListFromAtomList, \
    RemoveLatticeAmbiguity, GetGasMolFromSmiles
from util import GetGraphDescriptors
from util import SurfHelper
from rdkit import RDLogger
from datetime import datetime
RDLogger.DisableLog('rdApp.*')       
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

def GetIterator():
    if 'get_ipython' in locals().keys(): # it doesnt work in ipython
        multiprocessing = None
        return map
    else: 
        try: 
            import multiprocessing
            Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps) # not all props gets pickled
            p = multiprocessing.Pool()
            mapper = p.imap
        except:
            mapper = map
    return mapper 

surfhelp = SurfHelper(7)

def GetCanonicalSmiles(s):
    if '*' not in s: 
        return None
    try:
        return surfhelp.GetCanonicalSmiles(s)
    except:
        return None

def GetCanonicalSmiles2(inputs):
    surf,s = inputs
    if '*' not in s: 
        return surf,None
    try:
        return surf, surfhelp.GetCanonicalSmiles(s)
    except:
        return surf, None

def StandardFrame(fig, ax):
    fig.set_size_inches(9,9)
    ax.tick_params(direction="in",width=3)
    for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
        ax.spines[side].set_linewidth(0)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(24)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(24)
    ax.xaxis.label.set_size(24)
    ax.yaxis.label.set_size(24)

def GetGraphDescriptors_(s):
    return GetGraphDescriptors(s,maxatom=3)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def GetX(inputs):
    smiles,unique_descriptors = inputs
    x = np.zeros(len(unique_descriptors))
    for d in list(itertools.chain.from_iterable(GetGraphDescriptors_(smiles))):
        try: 
            i = unique_descriptors[d]
        except:
            continue
        x[i] += 1
    return x

if __name__ == '__main__':
    mapper = GetIterator()
    mapper = map
    smiles = json.load(open('../enumeration/Output/Size3.json'))
    unique_descriptors = pickle.load(open('./Output/unique_descriptors.pkl','rb'))
    models = pickle.load(open('./Output/models.pkl','rb'))

    Yps = []
    for i,sbatch in tqdm(enumerate(chunks(smiles,50000)),total=int(len(smiles)/50000)):
        print(i+1,'/',int(len(smiles)/50000)+1)
        Xs = []
        inputs = [[s,unique_descriptors] for s in sbatch]
        for i,x in enumerate(mapper(GetX,inputs)):
            Xs.append(x)
            if i != 0 and i %10000 == 0:
                print(datetime.now().strftime("[%H:%M:%S]"),i,'/%i'%len(inputs))
        Yp = []
        for surf in models:
            yp = np.around(models[surf].predict_proba(Xs)[:,1],3)
            #yp = (yp>criterion).astype(int)
            Yp.append(yp)
        Yp = np.array(Yp).T
        Yps.append(Yp)
    Yps = np.concatenate(Yps)
    json.dump(Yps.tolist(),open('./Output/Size3Output.json','w'))

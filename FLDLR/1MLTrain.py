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
from util import GetBondListFromAtomList, \
    RemoveLatticeAmbiguity, GetGasMolFromSmiles
from util import GetGraphDescriptors
from util import SurfHelper, RemoveHe
from rdkit import RDLogger
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


if __name__ == '__main__':
    #mapper = GetIterator()
    mapper = map
    
    
    if not os.path.exists('Output/MLTrain_cache.pkl'):
        basegraph_ = []
        for p in ['../enumeration/Output/Size1.json',
                  '../enumeration/Output/Size2.json',
                  '../enumeration/Output/Size3.json']:
            basegraph_ += json.load(open(p))
        
        basegraph = []
        for s in tqdm(mapper(GetCanonicalSmiles, basegraph_),total=len(basegraph_)):
            if s is not None:
                basegraph.append(s)
                
        stables = [[s[0],RemoveHe(s[1])] for s in json.load(open('./Input/stable_smiles.json'))]
        
        data = []
        for surf,s in tqdm(mapper(GetCanonicalSmiles2, stables),total=len(stables)):
            if s is not None:
                data.append([surf,s])
        
        n = []
        for d in data:
            if d[1] not in basegraph:
                n.append(d[1])
        basegraph += n

        natom = []
        for s in basegraph:
            natom.append(s.count('C')+s.count('O'))
        natom = np.array(natom)
        
        for d in data:
            d.append(basegraph.index(d[1]))


    
        pickle.dump((basegraph,data,natom),open('Output/MLTrain_cache.pkl','wb'))
    else:
        basegraph,data,natom = pickle.load(open('Output/MLTrain_cache.pkl','rb'))
    
    
    descriptors = []
    for r in tqdm(mapper(GetGraphDescriptors,basegraph),total=len(basegraph)):
        descriptors.append(list(itertools.chain.from_iterable(r)))
        
    unique_descriptors = sorted(set(list(itertools.chain.from_iterable(descriptors))))
    unique_descriptors = {s:i for i,s in enumerate(unique_descriptors)}
    pickle.dump(unique_descriptors,open('./Output/unique_descriptors.pkl','wb'))
    
    # X matrix
    X = np.zeros((len(basegraph),len(unique_descriptors)))
    for i,ds in enumerate(descriptors):
        for d in ds:
            X[i,unique_descriptors[d]] += 1
    
    #%% 
    print(len(data))
    from sklearn.linear_model import LogisticRegression
    
    metals = ['Ag111','Au111','Co0001','Fe110','Cu111','Ir111','Ni111','Pd111','Pt111','Re0001','Rh111','Ru0001']
    
    print('%6s %6s %6s %6s'%('Surf','All','AllRC','AllFP'))
    criterion=0.5
    C = 1.0
    models = {}
    for surf in metals:
        if surf == 'Fe110': continue
        Ys = np.zeros(X.shape[0])
        for surfd,s,i in data:
            if surf == surfd:
                Ys[i] = 1
        # process
        Xs = X
        Ys = np.array(Ys)
        # all acc
        # clf = LogisticRegression(C=C).fit(Xs,Ys)
        clf = LogisticRegression(C=C,class_weight={0:1,1:np.sum(Ys==0)/np.sum(Ys==1)*1.5},n_jobs=-1,max_iter=1000).fit(Xs,Ys)
        Yps = clf.predict_proba(Xs)[:,1]
        accAll = np.mean((Ys>0.5) == (Yps>criterion))
        RCAll = np.mean(Yps[Ys>0.5]>criterion)
        FPAll = np.sum(Ys[Yps>criterion]<0.5)

        print('%6s %6.2f %6.2f %6d'%(surf,accAll,RCAll,FPAll))
        models[surf] = clf
    pickle.dump(models,open('./Output/models.pkl','wb'))

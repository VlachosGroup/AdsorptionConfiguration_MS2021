# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 13:29:16 2019

@author: user
"""
import os
from rdkit import Chem
import json
from sklearn.utils.extmath import cartesian
import numpy as np 
from datetime import datetime
from time import time

def AddMeats(smiles):
    # initialize variables
    H = Chem.Atom(1)
    O = Chem.Atom(8)
    O.SetNoImplicit(True)
    O.SetNumRadicalElectrons(1)
    
    # load smiles
    mol = Chem.MolFromSmiles(smiles,sanitize=False)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            atom.SetNoImplicit(True)
            atom.SetNumRadicalElectrons(1)
    
    # start enumerating
    AdsIdx = [] # adsorbate atom index
    Combs = [] # possible combinations
    Products = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            AdsIdx.append(atom.GetIdx())
            sbool = False # adsorbed
            nv = 4 # valency 
            for na in atom.GetNeighbors():
                if na.GetAtomicNum() == 6:
                    nv -= 1
                elif na.GetAtomicNum() == 0:
                    sbool = True
                    
            if nv == 4:
                if sbool: 
                    Possibles = ['CHHH','CHH','CH','C','OH','O']
                else:
                    Possibles = ['CHHHH','OHH']
            elif nv == 3:
                if sbool:
                    Possibles = ['CHH','CH','C','O']
                else:
                    Possibles = ['CHHH','CHH','CH','C','OH','O']
            elif nv == 2:
                if sbool:
                    Possibles = ['CH','C']
                else:
                    Possibles = ['CHH','CH','C','O']
            elif nv == 1:
                if sbool:
                    Possibles = ['C']
                else:
                    Possibles = ['CH','C']
            elif nv ==0:
                Possibles.append('C')
                
            Combs.append(Possibles)
    Combs = tuple(np.array(c,dtype=np.dtype(('U',5))) for c in Combs)
    for comb in cartesian(Combs):
        newmol = Chem.RWMol(mol.__copy__())
        for i,a in zip(AdsIdx,comb):
            if a[0] == 'O':
                newmol.ReplaceAtom(i,O)
            for _ in a[1:]:
                hi = newmol.AddAtom(H)
                newmol.AddBond(i,hi,order=Chem.rdchem.BondType.SINGLE)
        Products.append(Chem.MolToSmiles(newmol))
    return Products

if __name__ == '__main__':

    # inputpath ='./Output/1Skeleton1.json'
    # outputpath = './Output/Size1.json'
    
    # inputpath ='./Output/1Skeleton2.json'
    # outputpath = './Output/Size2.json'

    inputpath ='./Output/4Skeleton3Good.json'
    outputpath = './Output/Size3.json'
    



    smiles = json.load(open(inputpath))
    
    if 'get_ipython' in locals().keys(): # it doesnt work in ipython
        multiprocessing = None
    else: 
        try: 
            import multiprocessing
        except:
            from itertools import imap
            multiprocessing = None
        
    if multiprocessing is not None:
        p = multiprocessing.Pool()
        gen = p.imap_unordered(AddMeats,smiles)
    else:
        gen = imap(AddMeats,smiles)
        
    ndata = len(smiles)
    print(datetime.now().strftime("[%H:%M:%S]"),'0','/',ndata)
    data = []; t = time()
    for i,r in enumerate(gen):
        data += r
        if i != 0 and i %1000 == 0:
            print(datetime.now().strftime("[%H:%M:%S]"),i,'/',ndata,\
            '| %.2f/1000 sec/cif |'%((time()-t)/i*1000), '~%.2f sec left'%((ndata-i)/i*(time()-t)))

    print(len(data))
    os.makedirs(os.path.split(outputpath)[0],exist_ok=True)
    json.dump(data,open(outputpath,'w'))
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:57:31 2018

@author: Gu
"""
import json
from rdkit import Chem
from itertools import combinations
import timeit
from util import SurfHelper, SetUpReaction, RemoveLatticeAmbiguity, CheckConfig,CleanUp
from tqdm import tqdm
import time
import datetime
import pandas
class timer(object):
    def __init__(self):
        self.tt = 0
    def t(self,text):
        s = '['+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        s += ']  ' + '%.2f'%(timeit.default_timer()-self.tt) + 's'
        self.tt = timeit.default_timer()
        s += ':' + text
        print(s)
    
t = timer()

###############################################################################
AddRs = False
debug = True
useconstraint = False
###############################################################################
print('initialize')
PairConf = json.load(open('./Output/1Skeleton2.json'))[:-4]

# Make Rules
Rules = list()
for smiles in PairConf:
    Rules += SetUpReaction(smiles)

## Set up Surface
c = SurfHelper(10)
C2 = []
for smiles in PairConf:
    mol = c.AddAdsorbateToSurf(smiles)
    # set bond properties. 
    for bond in mol.GetBonds():
        if bond.GetBeginAtom().GetAtomicNum() == 0 or bond.GetEndAtom().GetAtomicNum() == 0 :
            bond.SetBondType(Chem.BondType.ZERO)
    # Set up properties to limit connecting more than 3 C to one C
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6: # Set up Valency
            atom.SetNoImplicit(True)
            atom.UpdatePropertyCache() 
            nS = 0
            for na in atom.GetNeighbors():
                if na.GetAtomicNum() == 0:
                    nS += 1
            atom.SetIntProp('nS',nS)
        else: # Set up occupancy
            for na in atom.GetNeighbors():
                if na.GetAtomicNum() == 6:
                    atom.SetBoolProp('Occ',True)
                    break
            atom.SetBoolProp('Occ',False)

    C2.append(mol)
t.t('Finished. Enumerate')
# Reaction Network Generation
Reactants = C2
Products= list()
for Reactant in tqdm(Reactants):
    for Rule in Rules:
        ProductSet = Rule.RunReactants((Reactant,))
        for mol in ProductSet:
            mol = mol[0]
            #TIP: line below is the only difference between 2ThreeAtomConfigEnum.py
            mol = Chem.RWMol(mol)
            Cidx = []
            for Atom in mol.GetAtoms():
                if Atom.GetAtomicNum() == 6:
                    Cidx.append(Atom.GetIdx())
            for pair in combinations(Cidx,2):
                if not mol.GetBondBetweenAtoms(pair[0],pair[1]):
                    mol.AddBond(pair[0],pair[1],order=Chem.BondType.SINGLE)
            CleanUp(mol)
            # Append to Products
            Products.append(mol)
print('TotalProducts: ',len(Products))
t.t('Finished. Find Unique')
# Get unique smiles
slist = list()
for mol in tqdm(Products):
    # Bond
    for bond in mol.GetBonds():
        bond.SetBondType(Chem.rdchem.BondType.SINGLE)
    smol = Chem.RWMol(RemoveLatticeAmbiguity(mol))
    for atom in smol.GetAtoms():
        atom.SetIsotope(0)
    slist.append(Chem.MolToSmiles(smol))
uniquesmiles = pandas.unique(slist)

t.t('Finished. ForcefieldCheck')
# Check strain, or being too closed
GoodSmiles = []
BadSmiles = []
for s in tqdm(uniquesmiles):
    out,_ = CheckConfig(s)
    if out == 1:
        GoodSmiles.append(s)
    else:
        BadSmiles.append(s)
        
json.dump(GoodSmiles,open('./Output/2Skeleton3ringGood.json','w'))
json.dump(BadSmiles,open('./Output/2Skeleton3ringBad.json','w'))

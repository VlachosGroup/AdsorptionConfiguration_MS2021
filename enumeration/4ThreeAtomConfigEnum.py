# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:57:31 2018

@author: Gu
"""
import json
import numpy as np
from rdkit import Chem
from itertools import combinations
import timeit
from util import SurfHelper, SetUpReaction, RemoveLatticeAmbiguity, CheckConfig,\
    SetUpConstraintMol, BridgeRule, SetUpRingReaction, CleanUp
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

print('initialize')
PairConf = json.load(open('./Output/1Skeleton2.json'))
ThreeChain = json.load(open('./Output/3Skeleton3chainGood.json'))
ThreeRing = json.load(open('./Output/2Skeleton3ringGood.json'))
## Set up Surface
c = SurfHelper(10)

# Make Rules
Rules = list()
for smiles in PairConf:
    Rules += SetUpReaction(smiles)
for smiles in ThreeChain+ThreeRing:
    Rules += SetUpRingReaction(smiles)
BrgRule = BridgeRule(c.xyz,c.sites)

C = []
for smiles in PairConf[:-1]: # exclude CC
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

    C.append(mol)
# Reaction Network Generation
Reactants = C
Products= list()
for Reactant in tqdm(Reactants):
    # Add adsorbed atom next to another adsorbed atom
    for Rule in Rules:
        ProductSet = Rule.RunReactants((Reactant,)) # This is slow try renumbering 
        for mol in ProductSet:
            mol = mol[0]
            CleanUp(mol)
            # Append to Products
            Products.append(mol)
    # Connect adsorbed atom with the bridge
    for p in range(len(Products)):
        p= Products[p]
        ProductSet = BrgRule.ConnectBrgNewAtom((p,))
        for mol in ProductSet:
            mol = mol[0]
            CleanUp(mol)
            # Append to Products
            Products.append(mol)
    # Long Bridge formation
    ProductSet = BrgRule.RunReactants((Reactant,))
    for mol in ProductSet:
        mol = mol[0]
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
        
json.dump(GoodSmiles,open('./Output/4Skeleton3Good.json','w'))
json.dump(BadSmiles,open('./Output/4Skeleton3Bad.json','w'))


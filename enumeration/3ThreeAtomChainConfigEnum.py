# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:57:31 2018

@author: Gu
"""
import json
from rdkit import Chem
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
"""
CANONICAL LIST OF PAIR CONFORMATION

11
#C1C1 - 2 atom
'C1C**1'
#C1C1 - 1 atom
'C1C*1'

12
#C1C2 - 3 atom
'C1C2*3*2*13'
#C1C2 - 2 atom
'C12C*1*2'

13
#C1C3 - 4 atom
'C1C23*45*2*34*15'
#C1C3 - 3 atom
'C1C23*4*2*134'

22
#C2C2 - 4 atom
'C12C3*45*3*14*25'
#C2C2 - 3 atom (X)
'C12C3*4*235*1*45'
#C2C2 - 3 atom (X)
'C12C3*4*2356*1*5*46'
#C2C2 - 3 atom
'C12C3*4*31*24'
#C2C2 - 2 atom
'C12C3*1*23'

23
#C2C3 - 5 atom (X)
'C12C34*567*3*45*16*27'
#C2C3 - 4 atom
'C12C34*56*3*145*26'
#C2C3 - 4 atom (X)
'C12C34*5*1367*2*6*457'
#C2C3 - 3 atom
'C12C34*5*13*245'

33
#C3C3 - 6 atom (X)
'C123C45*67*489*1*29*378*56'
#C3C3 - 5 atom (X)
'C123C45*1678*4*56*27*38'
#C3C3 - 5 atom (X)
'C123C45*6*1478*2*37*568'
#C3C3 - 5 atom (X)
'C123C45*67*8*19*2468%10%11*57*%10*39%11'
#C3C3 - 4 atom
'C123C45*167*4*256*37'
#C3C3 - 3 atom (X)
'C123C45*16*24*356'

# For adsorbing saturated atoms
1S
'CC*'

2S
'CC2**2'

3S
'CC23*4*2*34'

SS
'CC'
"""
print('initialize')
PairConf = json.load(open('./Output/1Skeleton2.json'))[:-4] # without bridge


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
        
json.dump(GoodSmiles,open('./Output/3Skeleton3chainGood.json','w'))
json.dump(BadSmiles,open('./Output/3Skeleton3chainBad.json','w'))

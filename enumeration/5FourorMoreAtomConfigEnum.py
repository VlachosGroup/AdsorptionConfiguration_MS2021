# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:57:31 2018

@author: Gu
"""
import json
from rdkit import Chem
from util import SurfHelper, SetUpReaction, RemoveLatticeAmbiguity, CheckConfig,\
    BridgeRule, SetUpRingReaction, CleanUp
from tqdm import tqdm
import pandas
import time
import datetime
import timeit
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')       
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

def Check(input):
    if isinstance(input,Chem.Mol):
        mol = input.__copy__()
    else:
        mol = Chem.MolFromSmiles(input,sanitize=False)
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
        if atom.GetAtomicNum() == 6:
            atom.SetNoImplicit(True)
            atom.SetNumRadicalElectrons(1)
            n = 0
            for na in atom.GetNeighbors():
                if na.GetAtomicNum() == 0:
                    n +=1
            atom.SetProp('smilesSymbol','%i'%n)
    mol = Chem.RWMol(mol)
    for i in reversed(range(mol.GetNumAtoms())):
        a = mol.GetAtomWithIdx(i)
        if a.GetAtomicNum() == 0:
            mol.RemoveAtom(i)
    return Chem.MolToSmiles(mol)

class timer(object):
    def __init__(self):
        self.tt = 0
    def t(self,text):
        s = '['+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        s += ']  ' + '%.2f'%(timeit.default_timer()-self.tt) + 's'
        self.tt = timeit.default_timer()
        s += ':' + text
        print(s)
    
############################### initialize. ###################################
# needs to be outside the __main__ to work properly. Multiprocessing cannot 
# pickle these data
c = SurfHelper(10)

PairConf = json.load(open('./Output/1Skeleton2.json'))
ThreeChain = json.load(open('./Output/3Skeleton3chainGood.json'))
ThreeRing = json.load(open('./Output/2Skeleton3ringGood.json'))
## Set up Surface

# Make Rules
Rules = list()
for smiles in PairConf:
    Rules += SetUpReaction(smiles)
for smiles in ThreeChain+ThreeRing:
    Rules += SetUpRingReaction(smiles)
BrgRule = BridgeRule(c.xyz,c.sites)
###############################################################################


def GetIterator():
    if 'get_ipython' in locals().keys(): # it doesnt work in ipython
        multiprocessing = None
    else: 
        try: 
            import multiprocessing
            Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps) # not all props gets pickled
            p = multiprocessing.Pool()
            mapper = p.imap_unordered
        except:
            mapper = map
    return mapper 

def AddAdsorbateToSurf(smiles):
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
    return mol
    

def AddAtom(Reactant):
    Products = []
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
    return Products

def RemoveAmbiguity(mol):
    # Bond
    for bond in mol.GetBonds():
        bond.SetBondType(Chem.rdchem.BondType.SINGLE)
    smol = Chem.RWMol(RemoveLatticeAmbiguity(mol))
    for atom in smol.GetAtoms():
        atom.SetIsotope(0)
    return Chem.MolToSmiles(smol)


if __name__ == '__main__':
    # initialize parallel processing
    mapper = GetIterator() 
    # mapper = map
    # input
    reactant_path = './Output/4Skeleton3Good.json'
    output_good_path = './Output/5Skeleton4Good.json'
    output_bad_path = './Output/5Skeleton4Bad.json'
    
    # load data
    print('initialize')
    ReactantSmiles = json.load(open(reactant_path))
    
    
    # add smiles to large surface graph
    Reactants = []
    for mol in tqdm(mapper(AddAdsorbateToSurf, ReactantSmiles),total=len(ReactantSmiles)):
        Reactants.append(mol)
    # Reaction Network Generation
    Products= list()
    for p in tqdm(mapper(AddAtom, Reactants),total=len(Reactants)):
        Products += p
    print('TotalProducts: ',len(Products))
    t = timer()
    t.t('Finished. Find Unique')
    # Get unique smiles
    slist = list()
    for smiles in tqdm(mapper(RemoveAmbiguity,Products),total=len(Products)):
        slist.append(smiles)
    uniquesmiles = pandas.unique(slist).tolist()
    print('Total uniques ',len(uniquesmiles))
    t.t('Finished. ForcefieldCheck')
    # Check strain, or being too closed
    GoodSmiles = []
    BadSmiles = []
    for out,s in tqdm(mapper(CheckConfig, uniquesmiles),total=len(uniquesmiles)):
        if out == 1:
            GoodSmiles.append(s)
        else:
            BadSmiles.append(s)
            
    json.dump(GoodSmiles,open(output_good_path,'w'))
    json.dump(BadSmiles,open(output_bad_path,'w'))


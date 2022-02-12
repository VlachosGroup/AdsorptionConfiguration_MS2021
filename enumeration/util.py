import numpy as np
from rdkit import Chem
from rdkit.Chem import rdqueries, rdForceFieldHelpers
from rdkit.Chem.rdChemReactions import ChemicalReaction
from collections import defaultdict
from itertools import combinations
from scipy.spatial.distance import pdist
import itertools
from scipy.spatial.distance import cdist
from ase import Atoms as ASEAtoms
from ase import Atom as ASEAtom
from ase.data import atomic_numbers
from rdkit.Chem import AllChem
import random
from ase.io import write,read
import os
from collections import Counter
from ast import literal_eval
from rdkit import Geometry

def GetBondListFromAtomList(Mol, AtomList):
    BondList = set()
    for idx in AtomList:
        atom = Mol.GetAtomWithIdx(idx)
        for bond in atom.GetBonds():
            if bond.GetOtherAtomIdx(atom.GetIdx()) in AtomList:
                BondList.add(bond.GetIdx())
    return list(BondList)
    
def GetSubMolFromIdx(Idxs,Mol):
    Mapping = dict() # Original Mol Idx -> New Mol Idx
    if len(Idxs) != 1:
        BondList = GetBondListFromAtomList(Mol,Idxs)
        NewMol = Chem.RWMol(Chem.PathToSubmol(Mol,BondList,atomMap = Mapping))
    else:
        NewMol = Chem.RWMol(Mol)
        # Remove Non surface Atom
        for idx in reversed(range(0,NewMol.GetNumAtoms())):
            if idx not in Idxs:
                NewMol.RemoveAtom(idx)
        Mapping[Idxs[0]] = 0
    
    ReverseMapping = dict()
    for Idx in Mapping:
        ReverseMapping[Mapping[Idx]] = Idx
    
    return NewMol, Mapping, ReverseMapping

SurfaceElements = ('Ag','Au','Co','Cu','Fe','Ir','Ni','Pd','Pt','Re','Rh','Ru')
SurfaceAtomicNumbers = tuple([0]+[atomic_numbers[s] for s in SurfaceElements])
# Elements of adsorbate atoms
AdsorbateElements = ('H','C','O','N')
AdsorbateAtomicNumbers = tuple([atomic_numbers[s] for s in AdsorbateElements])

def BFSShortestPath(mol,idxs):
    # Step1: Initialize
    predecessors = []
    for _ in idxs:
        predecessors.append([[] for _ in range(mol.GetNumAtoms())])
    for i,j in enumerate(idxs):
        predecessors[i][j] = True
    queues = [[mol.GetAtomWithIdx(i)] for i in idxs]
    Checked = [set() for i in idxs]
    MeetingPoints = [[[] for _ in idxs] for _ in idxs]
    HaveWeMet = [[False for _ in idxs] for _ in idxs]
    for i in range(len(idxs)):
        HaveWeMet[i][i] = True
    
    # Search
    for _ in range(mol.GetNumAtoms()): # maximum possible depth.
        for i,queue in enumerate(queues): # Starting node i
            
            Checked[i] |= set([q.GetIdx() for q in queue])
            newqueue = []
            newqueueidx = []
            for q in queue:
                for na in q.GetNeighbors(): # Breath first search
                    naidx = na.GetIdx()
                    if naidx not in Checked[i]:
                        # append predecessor 
                        predecessors[i][naidx].append(q.GetIdx()) 
                        # make new queues
                        if naidx not in newqueueidx: 
                            newqueue.append(na)
                            newqueueidx.append(naidx)
            
            # Check if it has met the searched nodes started from other nodes
            for naidx in newqueueidx:
                for j,pred in enumerate(predecessors):
                    if not HaveWeMet[i][j] and pred[naidx]:
                        MeetingPoints[i][j].append(naidx)
                        MeetingPoints[j][i].append(naidx)
            
            # Check if meeting points has been set
            for j in range(len(idxs)):
                if not HaveWeMet[i][j] and MeetingPoints[i][j]:
                    HaveWeMet[i][j] = True
                    HaveWeMet[j][i] = True

            if all(HaveWeMet[i]): # This has met all other nodes, so no need for further search
                queues[i] = []
            else: # It has not met all nodes. continue search
                queues[i] = newqueue
            
        # Check if every nodes have met each other
        if not any(queues):
            break
    
    shortestpath = set()
    for i in range(len(idxs)):
        for j in range(len(idxs)):
            pairshortestpath = set()
            AtomIdx2Check = MeetingPoints[i][j].copy()
            while AtomIdx2Check:
                idx = AtomIdx2Check.pop()
                if idx not in pairshortestpath:
                    pairshortestpath.add(idx)
                    if predecessors[i][idx] != True:
                        AtomIdx2Check += predecessors[i][idx]
            shortestpath |=pairshortestpath
    
    return(shortestpath)

def RemoveLatticeAmbiguity(OriginalMol):
    """
    Remove ambiguity in the subgraph provided. See manuscript for the 
    mechanism of this.
    
    Input:
        OriginalMol - Chem.Mol or RWMol Object.
        SubgraphIdx - List of Index of the atoms in subgraph.
    Output:
        Updated SubgraphIdx
    """
    # Initialize
    ## isolate surface of subgraph
    ## Extract surface atom index in the subgraph
    AAL = set() # (A)dsorbate (A)tom (L)ist
    SSAL = set() # (S)elected (S)urface (A)tom (L)ist
    SAL = set() # (S)urface (A)tom (L)ist
    for Idx in range(0,OriginalMol.GetNumAtoms()):
        atom = OriginalMol.GetAtomWithIdx(Idx)
        if atom.GetAtomicNum() in SurfaceAtomicNumbers:
            SAL.add(Idx)
            for na in atom.GetNeighbors():
                #if na.GetAtomicNum() in AdsorbateAtomicNumbers[1:]: # exclude hydrogen
                if na.GetAtomicNum() in AdsorbateAtomicNumbers: 
                    SSAL.add(Idx)
                    break

        elif atom.GetAtomicNum() in AdsorbateAtomicNumbers:
            AAL.add(Idx)
    SubgraphIdx = AAL | SSAL
    ## Check if surface atoms are fragmented
    AtomsToCheckList = list(SSAL)
    Surface_Fragments = list()
    while AtomsToCheckList:
        # initialize
        # here a single bridge is identified
        Atom = OriginalMol.GetAtomWithIdx(AtomsToCheckList.pop())
        Surface_Fragment = set()
        Surface_Fragment.add(Atom.GetIdx())
        NeighborsToCheck = list(Atom.GetNeighbors())
        # find all possible surface atoms in this fragment
        while NeighborsToCheck:
            AtomBeingChecked = NeighborsToCheck.pop()
            if AtomBeingChecked.GetIdx() in SSAL and \
                AtomBeingChecked.GetIdx() not in Surface_Fragment:
                Surface_Fragment.add(AtomBeingChecked.GetIdx())
                NeighborsToCheck += AtomBeingChecked.GetNeighbors()
        # Add to fragment list
        Surface_Fragments.append(Surface_Fragment)
        # Remove checked atoms
        AtomsToCheckList = [value for value in AtomsToCheckList if value not in Surface_Fragment]
    

    # if the length Surface_Fragments is more than 1, then the surface is fragmented
    if len(Surface_Fragments) > 1:
        # Extract surface
        BondToBreak = set()
        for idx in SSAL:
            Atom = OriginalMol.GetAtomWithIdx(idx)
            for Bond in Atom.GetBonds():
                if Bond.GetOtherAtom(Atom).GetIdx() not in SAL:
                    BondToBreak.add(Bond.GetIdx())
        SurfaceGraph = Chem.FragmentOnBonds(OriginalMol,list(BondToBreak),addDummies=False)

        ########################################################################
        Surf = Surface_Fragments[0]
        for s in Surface_Fragments[1:]:
            Surf.update(s)
        # BRS Based shortest path find
        NewSurfIdx = BFSShortestPath(SurfaceGraph,list(Surf))
        SubgraphIdx |= NewSurfIdx
        SSAL |= NewSurfIdx

        #######################################################################
    
    SubgraphIdx = list(SubgraphIdx)
    ## initialize 
    NSAD = defaultdict(int) # (N)eighbor (S)urface (A)tom (D)ict
    ## intial dict list
    for SSAIdx in SSAL:
        for NeighborAtom in OriginalMol.GetAtomWithIdx(SSAIdx).GetNeighbors():
            if NeighborAtom.GetAtomicNum() in SurfaceAtomicNumbers:
                if NeighborAtom.GetIdx() not in SSAL:
                    NSAD[NeighborAtom.GetIdx()] += 1
    # add nonselected surface atoms to subgraph
    for idx in NSAD:
        if NSAD[idx] > 1:
            SubgraphIdx.append(idx)
    
    ## add second layer 
    for atom in OriginalMol.GetAtoms():
        if atom.GetAtomicNum() == 2:
            na = 0
            for natom in atom.GetNeighbors():
                if natom.GetIdx() in SubgraphIdx:
                    na += 1
            if na ==3:
                SubgraphIdx.append(atom.GetIdx())
    
    BondList = GetBondListFromAtomList(OriginalMol, SubgraphIdx)
    if BondList:
        mol = Chem.PathToSubmol(OriginalMol,BondList)
    else: # if adsorbate is one atom, there is no bond list, so just return the atom.
        mol = Chem.RWMol(Chem.Mol())
        mol.AddAtom(OriginalMol.GetAtomWithIdx(list(AAL)[0]).__copy__())
    return mol

class Site(object):
    """
    Object for a site
    Attributes:
    SiteType    - site type in integer
    Coordinate  - 2D coordinates of the site location
    Neighbors   - id connected neighbor
    DuplicateNeighborError - if True, the code spits error if there is a 
        intersection between the site neighbor list and the one being appended
    """
    def __init__(self, SiteType, Coordinate, DuplicateNeighborError=False):
        # Error check
        assert isinstance(SiteType, int), 'SiteType is not an integer.'
        assert isinstance(Coordinate, list) or isinstance(Coordinate, np.ndarray), 'Coordinate is not a list.'
        assert len(Coordinate) == 3,'Coordinate is not 3 dimensional.'
        assert (isinstance(Coordinate[0], float) or isinstance(Coordinate[0], int))\
            and (isinstance(Coordinate[1], float) or isinstance(Coordinate[1], int))\
            and (isinstance(Coordinate[2], float) or isinstance(Coordinate[2], int)), 'Coordinate element is not a float or int.'
        # Construct a site
        self._SiteType = SiteType # e.g. atop bridge hollow sites
        self._Coordinate = np.array(Coordinate, float)
        self._DuplicateNeighborError = DuplicateNeighborError
        self._SiteNeighbors = set()
        self._AtomNeighbors = set()
        self._RepresentedAtoms = set() # list of actual Pt atoms
        
    def __str__(self):
        return '<Site(Type:%i,xyz:[%.2f,%.2f,%.2f],Number of Neighbors: %i>' \
            %(self._SiteType,self._Coordinate[0],self._Coordinate[1],\
            self._Coordinate[2],len(self._SiteNeighbors))
    def __repr__(self):
        s = 'Site(Type:%i, xyz:[%.2f,%.2f,%.2f], Neighbors:' \
            %(self._SiteType,self._Coordinate[0],self._Coordinate[1],\
            self._Coordinate[2])
        for Neighbor in self._SiteNeighbors:
            s += str(Neighbor) + ','
        s += ', Associated_Pt_Atoms: '
        for Pt_atoms in self._RepresentedAtoms:
            s += str(Pt_atoms) + ','
        s += ')'
        return s
        
    def AppendSiteNeighbors(self, Neighbors):
        # Error check
        try:
            if not isinstance(Neighbors,(int,np.int64)):
                A = iter(Neighbors)
                for a in A:
                    if not isinstance(a,(int,np.int64)):
                        raise Exception
        except Exception:
            raise Exception("Neighbors is not iterable object with integer or an integer.")
        # append neighbor
        if isinstance(Neighbors,(int,np.int64)):
            if self._DuplicateNeighborError:
                if Neighbors in self._SiteNeighbors:
                    raise Exception("Neighbor " + str(Neighbors) + " is already in the neighbor list")
            self._SiteNeighbors.add(Neighbors)
        else:
            for Neighbor in Neighbors:
                if self._DuplicateNeighborError:
                    if Neighbor in self._SiteNeighbors:
                        raise Exception("Neighbor " + Neighbor +" is already in the neighbor list")
                self._SiteNeighbors.add(Neighbor)
                
    def AppendAtomNeighbors(self, Neighbors):
        # Error check
        try:
            if not isinstance(Neighbors, (int,np.int64)):
                A = iter(Neighbors)
                for a in A:
                    if not isinstance(a,(int,np.int64)):
                        raise Exception
        except Exception:
            raise Exception("Neighbors is not iterable object with integer or an integer.")
        # append neighbor
        if isinstance(Neighbors, (int,np.int64)):
            if self._DuplicateNeighborError:
                if Neighbors in self._AtomNeighbors:
                    raise Exception("Neighbor " + str(Neighbors) + " is already in the neighbor list")
            self._AtomNeighbors.add(Neighbors)
        else:
            for Neighbor in Neighbors:
                if self._DuplicateNeighborError:
                    if Neighbor in self._AtomNeighbors:
                        raise Exception("Neighbor " + Neighbor +" is already in the neighbor list")
                self._AtomNeighbors.add(Neighbor)
    
    def AppendRepresentedAtoms(self, Pt_indexes):
        # this is for actual Pt atoms associated with sites
        # Error check
        try:
            if isinstance(Pt_indexes, str) and Pt_indexes == 'self':
                pass
            elif not isinstance(Pt_indexes, (int,np.int64)):
                A = iter(Pt_indexes)
                for a in A:
                    if not isinstance(a,(int,np.int64)):
                        raise Exception
        except Exception:
            raise Exception("Neighbors is not iterable object with integer or an integer.")
        # append neighbor
        if isinstance(Pt_indexes, str) and Pt_indexes == 'self':
            self._RepresentedAtoms.add('self')
        elif isinstance(Pt_indexes, (int,np.int64)):
            if self._DuplicateNeighborError:
                if Pt_indexes in self._RepresentedAtoms:
                    raise Exception(str(Pt_indexes) + " is already in the associated Pt site list")
            self._RepresentedAtoms.add(Pt_indexes)
        else:
            for index in Pt_indexes:
                if self._DuplicateNeighborError:
                    if index in self._RepresentedAtoms:
                        raise Exception(str(index) + " is already in the associated Pt site list")
                self._RepresentedAtoms.add(index)
    
    def GetCoordinate(self):
        return self._Coordinate.copy()
        
    def GetSiteType(self):
        return self._SiteType.copy()


class Lattice(object):
    def __init__(self,Sites=[],SiteNames=[], DistanceMultiplier=[],Cell=np.eye(3),PBC=False):
        # Error Check
        assert isinstance(Sites, list), 'Sites is not a list.'
        for site in Sites:
            assert isinstance(site,Site), 'Site is not a Site object'
        self._SiteNames = SiteNames
        self._DistanceMultiplier = DistanceMultiplier #This number is multiplied before deciding which atom is at which site.
        self._Sites = Sites
        self.SetCell(Cell)
        self.SetPBC(PBC)
        
    def SetCell(self, Cell, KeepAbsCoord=False):
        Cell = np.array(Cell, float)
        if Cell.shape == (3,):
            Cell = np.diag(Cell)
        elif Cell.shape != (3, 3):
            raise ValueError('Cell must be length 3 sequence or 3x3 matrix')
        
        
        if KeepAbsCoord:
            Cell_inv = np.linalg.inv(Cell.transpose())
            for i in range(0,len(self._Sites)):
                pos = np.dot(self._Cell.transpose(),self._Sites[i]._Coordinate.transpose()).transpose()
                self._Sites[i]._Coordinate = np.dot(Cell_inv,pos.transpose()).transpose()  
                
        self._Cell = Cell
    def SetPBC(self, PBC):
        """Set periodic boundary condition flags."""
        if isinstance(PBC, bool):
            PBC = (PBC,) * 3
        else:
            try:
                iter(PBC)
            except TypeError:
                raise TypeError('PBC must be iterable or a bool')
            assert len(PBC) == 3, 'iterable PBC must be 3 sequence'
            for cond in PBC:
                assert isinstance(cond, bool), \
                'each element in PBC must be bool'
        self._PBC = np.array(PBC, bool)
        
    def GetRdkitMol(self,SurfaceAtomSymbol = 'Pt',queryatom=True):
        # initialize
        surface = Chem.RWMol(Chem.Mol())
        # add toms
        for site in self._Sites:
            if 'self' in site._RepresentedAtoms:
                if queryatom:
                    atom = rdqueries.HasStringPropWithValueQueryAtom('Type','S')
                    atom.ExpandQuery(rdqueries.HasBoolPropWithValueQueryAtom('Occupied',False))
                    atom.SetProp('smilesSymbol','M')
                    atom.SetProp('Type','S')
                    atom.SetBoolProp('Occupied',False)
                else:
                    if SurfaceAtomSymbol:
                        atom = Chem.Atom(SurfaceAtomSymbol)
                    else:
                        atom = Chem.Atom(0)
                    atom.SetProp('smilesSymbol','M')
                    atom.SetProp('Type','S')
                    atom.SetBoolProp('Occupied',False)
                surface.AddAtom(atom)
        # add bonds
        for i in range(0,len(self._Sites)):
            if 'self' in self._Sites[i]._RepresentedAtoms:
                for j in self._Sites[i]._AtomNeighbors:
                    if not surface.GetBondBetweenAtoms(i,int(j)):
                        surface.AddBond(i,int(j),order=Chem.rdchem.BondType.ZERO)
        Chem.SanitizeMol(surface)
        surface = surface.GetMol()
        Chem.SanitizeMol(surface)
        return surface
    def GetRdkitMolEnum(self):
        # initialize
        surface = Chem.RWMol(Chem.Mol())
        # add toms
        for site in self._Sites:
            if 'self' in site._RepresentedAtoms:
                atom = Chem.Atom(0)
                surface.AddAtom(atom)
        # add bonds
        for i in range(0,len(self._Sites)):
            if 'self' in self._Sites[i]._RepresentedAtoms:
                for j in self._Sites[i]._AtomNeighbors:
                    if not surface.GetBondBetweenAtoms(i,int(j)):
                        surface.AddBond(i,int(j),order=Chem.rdchem.BondType.SINGLE)
        Chem.SanitizeMol(surface)
        surface = surface.GetMol()
        Chem.SanitizeMol(surface)
        return surface
    def AppendSurfaceToRdkitMol(self,mol,SurfaceAtomSymbol = 'Pt',queryatom=True):
        # initialize
        if isinstance(mol,Chem.Mol):
            mol = Chem.RWMol(mol)
        assert isinstance(mol,Chem.RWMol)
        NAtoms = mol.GetNumAtoms()
        LatticeToMolMap = dict()
        MolToLatticeMap = dict()
        # add atoms
        for i in range(0,len(self._Sites)):
            if 'self' in self._Sites[i]._RepresentedAtoms:
                if queryatom:
                    atom = rdqueries.HasStringPropWithValueQueryAtom('Type','S')
                    atom.ExpandQuery(rdqueries.HasBoolPropWithValueQueryAtom('Occupied',False))
                    atom.SetProp('smilesSymbol','M')
                    atom.SetProp('Type','S')
                    atom.SetBoolProp('Occupied',False)
                else:
                    atom = Chem.Atom(SurfaceAtomSymbol)
                    atom.SetProp('smilesSymbol','M')
                    atom.SetProp('Type','S')
                    atom.SetBoolProp('Occupied',False)
                rdkitidx = mol.AddAtom(atom)
                LatticeToMolMap[i] = rdkitidx
                MolToLatticeMap[rdkitidx] = i
        # add bonds
        for i in range(0,len(self._Sites)):
            if 'self' in self._Sites[i]._RepresentedAtoms:
                for j in self._Sites[i]._AtomNeighbors:
                    try:
                        mol.AddBond(NAtoms+i,NAtoms+int(j),order=Chem.rdchem.BondType.ZERO)
                    except:
                        pass
        return mol, LatticeToMolMap, MolToLatticeMap
 
    def GetFracCoordinates(self):
        mat = list()
        for site in self._Sites:
            mat.append(site._Coordinate)
        return np.array(mat)
    
    def GetCoordinates(self):
        mat = self.GetFracCoordinates()
        return np.dot(self._Cell.transpose(),mat.transpose()).transpose()
        
    def GetCoordinatesWithCell(self,Cell):
        if not Cell.__class__ == np.ndarray:
            Cell = np.array(Cell)
        mat = self.GetFracCoordinates()
        
        return np.dot(Cell.transpose(),mat.transpose()).transpose()
    
    def TranslateCoordinates(self,coordinate):
        B_inv = np.linalg.inv(self._Cell.transpose())
        for site in self._Sites:
            pos = np.dot(self._Cell.transpose(),site._Coordinate.transpose()).transpose()
            pos += coordinate
            site._Coordinate = np.dot(B_inv,pos.transpose()).transpose()
            
    def MakeASEAtoms(self,highlight = None):
        atoms = ASEAtoms()
        coord = self.GetCoordinates()*2.5
        for i in range(0,coord.shape[0]):
            if highlight and i in highlight:
                atoms.append(ASEAtom('Pt',coord[i,:]))
            elif self._Sites[i]._SiteType == 0:
                atoms.append(ASEAtom('C',coord[i,:]))
            elif self._Sites[i]._SiteType == 1:
                atoms.append(ASEAtom('O',coord[i,:]))
            elif self._Sites[i]._SiteType == 2:
                atoms.append(ASEAtom('N',coord[i,:]))
        return atoms

    @classmethod
    def ConstructRectangularClosePackedLattice(cls, x_max,y_max, PBC=True):
        # option
        rd = 10 # rounding decimals   
        
        # Error check
        assert x_max > 1, "x_max too small"
        assert y_max > 1, "y_max too small"
        # set unit cell size
        Cell = [[2*x_max,0,0],[0,2*np.sqrt(3)/2*y_max,0],[0,0,1]]
        Cell = np.array(Cell)
        
        # Construct atop site coordinates
        ac = np.zeros((4*x_max*y_max,3))
        for y in range(0,y_max):
            for x in range(0,x_max):
                ac[4*(x+y*x_max),0] = 2*x
                ac[4*(x+y*x_max),1] = 2*np.sqrt(3)/2*y
                ac[4*(x+y*x_max)+1,0] = 2*x + 1
                ac[4*(x+y*x_max)+1,1] = 2*np.sqrt(3)/2*y
                ac[4*(x+y*x_max)+2,0] = 2*x + 0.5
                ac[4*(x+y*x_max)+2,1] = np.sqrt(3)/2 + 2*np.sqrt(3)/2*y
                ac[4*(x+y*x_max)+3,0] = 2*x + 1.5
                ac[4*(x+y*x_max)+3,1] = np.sqrt(3)/2 + 2*np.sqrt(3)/2*y
        # Construct bridge site coordinates
        bc = np.zeros((12*x_max*y_max,3))
        for y in range(0,y_max):
            for x in range(0,x_max):
                bc[12*(x+y*x_max),0] = 0.5+2*x
                bc[12*(x+y*x_max),1] = 2*np.sqrt(3)/2*y
                bc[12*(x+y*x_max)+1,0] = 1.5+2*x
                bc[12*(x+y*x_max)+1,1] = 2*np.sqrt(3)/2*y
                
                bc[12*(x+y*x_max)+2,0] = 0.25+2*x
                bc[12*(x+y*x_max)+2,1] = np.sqrt(3)/2/2 + 2*np.sqrt(3)/2*y
                bc[12*(x+y*x_max)+3,0] = 0.75+2*x
                bc[12*(x+y*x_max)+3,1] = np.sqrt(3)/2/2 + 2*np.sqrt(3)/2*y
                bc[12*(x+y*x_max)+4,0] = 1.25+2*x
                bc[12*(x+y*x_max)+4,1] = np.sqrt(3)/2/2 + 2*np.sqrt(3)/2*y
                bc[12*(x+y*x_max)+5,0] = 1.75+2*x
                bc[12*(x+y*x_max)+5,1] = np.sqrt(3)/2/2 + 2*np.sqrt(3)/2*y
                
                bc[12*(x+y*x_max)+6,0] = 2*x
                bc[12*(x+y*x_max)+6,1] = np.sqrt(3)/2 + 2*np.sqrt(3)/2*y
                bc[12*(x+y*x_max)+7,0] = 1+2*x
                bc[12*(x+y*x_max)+7,1] = np.sqrt(3)/2 + 2*np.sqrt(3)/2*y
                
                bc[12*(x+y*x_max)+8,0] = 0.25+2*x
                bc[12*(x+y*x_max)+8,1] = np.sqrt(3)/2/2*3 + 2*np.sqrt(3)/2*y
                bc[12*(x+y*x_max)+9,0] = 0.75+2*x
                bc[12*(x+y*x_max)+9,1] = np.sqrt(3)/2/2*3 + 2*np.sqrt(3)/2*y
                bc[12*(x+y*x_max)+10,0] = 1.25+2*x
                bc[12*(x+y*x_max)+10,1] = np.sqrt(3)/2/2*3 + 2*np.sqrt(3)/2*y
                bc[12*(x+y*x_max)+11,0] = 1.75+2*x
                bc[12*(x+y*x_max)+11,1] = np.sqrt(3)/2/2*3 + 2*np.sqrt(3)/2*y
                

        # Construct fcc site
        fccc = np.zeros((x_max*y_max*4,3))
        for x in range(0,x_max):
            for y in range(0,y_max):
                
                fccc[4*(x+y*(x_max)),0] = 0.5 + 2*x
                fccc[4*(x+y*(x_max)),1] = np.sqrt(3)/6+2*np.sqrt(3)/2*y
                fccc[4*(x+y*(x_max))+1,0] = 1.5 + 2*x
                fccc[4*(x+y*(x_max))+1,1] = np.sqrt(3)/6+2*np.sqrt(3)/2*y
                fccc[4*(x+y*(x_max))+2,0] = 2*x
                fccc[4*(x+y*(x_max))+2,1] = np.sqrt(3)/2 + np.sqrt(3)/6+2*np.sqrt(3)/2*y
                fccc[4*(x+y*(x_max))+3,0] = 1 + 2*x
                fccc[4*(x+y*(x_max))+3,1] = np.sqrt(3)/2 + np.sqrt(3)/6+2*np.sqrt(3)/2*y
                
                
        hcpc = np.zeros((x_max*y_max*4,3))
        for x in range(0,x_max):
            for y in range(0,y_max):
                
                hcpc[4*(x+y*(x_max)),0] = 2*x
                hcpc[4*(x+y*(x_max)),1] = np.sqrt(3)/6*2+2*np.sqrt(3)/2*y
                hcpc[4*(x+y*(x_max))+1,0] = 1 + 2*x
                hcpc[4*(x+y*(x_max))+1,1] = np.sqrt(3)/6*2+2*np.sqrt(3)/2*y
                hcpc[4*(x+y*(x_max))+2,0] = 0.5 + 2*x
                hcpc[4*(x+y*(x_max))+2,1] = np.sqrt(3)/2 + np.sqrt(3)/6*2+2*np.sqrt(3)/2*y
                hcpc[4*(x+y*(x_max))+3,0] = 1.5 + 2*x
                hcpc[4*(x+y*(x_max))+3,1] = np.sqrt(3)/2 + np.sqrt(3)/6*2+2*np.sqrt(3)/2*y

        # Construct Sites list
        SiteNames = ['Atop','Bridge','Hollow']
        DistanceMultiplier = [1,2.5, 2.5]
        Sites = list()
        ## Atop Site
        for i in range(0,ac.shape[0]):
            Sites.append(Site(0,ac[i]))
        ## Bridge Site
        for i in range(0,bc.shape[0]):
            Sites.append(Site(1,bc[i]))
        ## Hollow Site
        for i in range(0,fccc.shape[0]):
            Sites.append(Site(2,fccc[i]))
        ## Hollow Site
        for i in range(0,hcpc.shape[0]):
            Sites.append(Site(3,hcpc[i]))
        # Append Neighbors
        # set up periodic condition
        if PBC:
            pcs = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[-1,1,0],[-1,0,0],[-1,-1,0],[0,-1,0],[1,-1,0]])
        else:
            pcs = np.array([[0,0,0]])
        # actually calculate how much translation is requred
        pcts = list()
        for pc in pcs:
            pcts.append([2*x_max*pc[0],2*np.sqrt(3)/2*y_max*pc[1],0])
        pcts = np.array(pcts)    
        
        # periodic coordinate
        for pc in pcts:
            try: 
                apc = np.concatenate((apc,np.add(ac,pc)))
                bpc = np.concatenate((bpc,np.add(bc,pc)))
                fccpc = np.concatenate((fccpc,np.add(fccc,pc)))
                hcppc = np.concatenate((hcppc,np.add(hcpc,pc)))
            except NameError:
                apc = np.add(ac,pc)
                bpc = np.add(bc,pc)
                fccpc = np.add(fccc,pc)
                hcppc = np.add(hcpc,pc)
                
                
        ## atop site
        for i in range(0,ac.shape[0]):
            Sites[i].AppendRepresentedAtoms('self')
            # to other atop sites
            match = FindNeighbor(ac[i],apc,rd,1.0)
            match = np.remainder(match,ac.shape[0])
            Sites[i].AppendAtomNeighbors(match)
            # to other bridge sites
            match = FindNeighbor(ac[i],bpc,rd,0.5)
            match = np.remainder(match,bc.shape[0])
            Sites[i].AppendSiteNeighbors(match+ac.shape[0])
            # to other fcc sites
            match = FindNeighbor(ac[i],fccpc,rd,np.sqrt(3)/6*2)
            match = np.remainder(match,fccc.shape[0])
            Sites[i].AppendSiteNeighbors(match+ac.shape[0]+bc.shape[0])
            # to other hcp sites
            match = FindNeighbor(ac[i],hcppc,rd,np.sqrt(3)/6*2)
            match = np.remainder(match,hcpc.shape[0])
            Sites[i].AppendSiteNeighbors(match+ac.shape[0]+bc.shape[0]+fccc.shape[0])
        ## bridge site
        for i in range(0,bc.shape[0]):
            # to other atop sites
            match = FindNeighbor(bc[i],apc,rd,0.5)
            match = np.remainder(match,ac.shape[0])
            Sites[i+ac.shape[0]].AppendSiteNeighbors(match)
            Sites[i+ac.shape[0]].AppendRepresentedAtoms(match)
            # to other bridge sites
#            match = FindNeighbor(bc[i],bpc,rd,0.5)
#            match = np.remainder(match,bc.shape[0])
#            Sites[i+ac.shape[0]].AppendSiteNeighbors(match+ac.shape[0])
            # to other hollow sites
            match = FindNeighbor(bc[i],fccpc,rd,np.sqrt(3)/6)
            match = np.remainder(match,fccc.shape[0])
            Sites[i+ac.shape[0]].AppendSiteNeighbors(match+ac.shape[0]+bc.shape[0])
            # hollow
            match = FindNeighbor(bc[i],hcppc,rd,np.sqrt(3)/6)
            match = np.remainder(match,hcpc.shape[0])
            Sites[i+ac.shape[0]].AppendSiteNeighbors(match+ac.shape[0]+bc.shape[0]+fccc.shape[0])
        ## fcc site
        for i in range(0,fccc.shape[0]):
            # to other atop sites
            match = FindNeighbor(fccc[i],apc,rd,np.sqrt(3)/6*2)
            match = np.remainder(match,ac.shape[0])
            Sites[i+ac.shape[0]+bc.shape[0]].AppendSiteNeighbors(match)
            Sites[i+ac.shape[0]+bc.shape[0]].AppendRepresentedAtoms(match)
            # to other bridge sites
            match = FindNeighbor(fccc[i],bpc,rd,np.sqrt(3)/6)
            match = np.remainder(match,bc.shape[0])
            Sites[i+ac.shape[0]+bc.shape[0]].AppendSiteNeighbors(match+ac.shape[0])
            # to other hcp sites
#            match = FindNeighbor(fccc[i],hcppc,rd,np.sqrt(3)/3)
#            match = np.remainder(match,hcpc.shape[0])
#            Sites[i+ac.shape[0]+bc.shape[0]].AppendSiteNeighbors(match+ac.shape[0]+bc.shape[0]+fccc.shape[0])
        ## hcp site
        for i in range(0,hcpc.shape[0]):
            # to other atop sites
            match = FindNeighbor(hcpc[i],apc,rd,np.sqrt(3)/6*2)
            match = np.remainder(match,ac.shape[0])
            Sites[i+ac.shape[0]+bc.shape[0]+fccc.shape[0]].AppendSiteNeighbors(match)
            Sites[i+ac.shape[0]+bc.shape[0]+fccc.shape[0]].AppendRepresentedAtoms(match)
            # to other bridge sites
            match = FindNeighbor(hcpc[i],bpc,rd,np.sqrt(3)/6)
            match = np.remainder(match,bc.shape[0])
            Sites[i+ac.shape[0]+bc.shape[0]+fccc.shape[0]].AppendSiteNeighbors(match+ac.shape[0])
            # to other hcp sites
#            match = FindNeighbor(hcpc[i],fccpc,rd,np.sqrt(3)/3)
#            match = np.remainder(match,fccc.shape[0])
#            Sites[i+ac.shape[0]+bc.shape[0]+fccc.shape[0]].AppendSiteNeighbors(match+ac.shape[0]+bc.shape[0])
        # change basis from absolute to fractional
        # Basis1' * coordinate1' = Basis2' * coordinate2'
        B_inv = np.linalg.inv(Cell.transpose())
        for site in Sites:
            site._Coordinate = np.dot(B_inv,site._Coordinate).transpose()            
        # periodic boundary condition
        if PBC:
            PBC = (True,True,False)
        else:
            PBC = (False,False,False)
        # Return
        return cls(Sites=Sites,SiteNames=SiteNames,DistanceMultiplier=DistanceMultiplier,Cell=Cell,PBC=PBC)

def FindNeighbor(xyz,mat,round_decimal,desired_distance):
    mat = np.subtract(mat,xyz)
    ds = np.linalg.norm(mat,axis=1)
    ds = np.around(ds,decimals=round_decimal)
    desired_distance = np.around(desired_distance,decimals=round_decimal)
    return np.where(np.equal(ds,desired_distance))[0] # because it gives tuple of tuple
                

        


class SurfHelper(object):
    def __init__(self,size):
        # Construct Surface
        surf = Lattice.ConstructRectangularClosePackedLattice(size,size,PBC=False)
        # Get Surfrace Mol
        atomidx = []
        for i,s in enumerate(surf._Sites):
            if 'self' in s._RepresentedAtoms:
                atomidx.append(i)
        self.xyz = surf.GetCoordinates()[atomidx,:]
        
        self.sites = []
        for i,s in enumerate(surf._Sites):
            if 'self' in s._RepresentedAtoms:
                self.sites.append(frozenset([int(i)]))
            else:
                self.sites.append(frozenset([int(ss) for ss in s._RepresentedAtoms]))
        self.SurfMol = surf.GetRdkitMolEnum()
        # Get Center Atom index in rdkit mol
        SurfAtomCoordinates = list()
        for i in range(0,len(surf._Sites)):
            if 'self' in surf._Sites[i]._RepresentedAtoms:
                SurfAtomCoordinates.append(surf._Sites[i].GetCoordinate())
        CenterAtomIdx = np.linalg.norm(SurfAtomCoordinates - np.array([0.5,0.5,0]),axis=1).argmin()
        self.SurfMol.GetAtomWithIdx(int(CenterAtomIdx)).SetBoolProp('CenterSurfAtom',True)
        
        
        
    def AddAdsorbateToSurf(self,AdsorbateSmiles):
        # Prepare Adsorbate
        AdsorbateMol = Chem.MolFromSmiles(AdsorbateSmiles,sanitize=False)
        
        # Get list of Surface Atom Indices
        SurfIdxs = list()
        GasIdxs = list()
        for atom in AdsorbateMol.GetAtoms():
            if atom.GetAtomicNum() not in [1,6,8]:
                SurfIdxs.append(atom.GetIdx())
            else:
                GasIdxs.append(atom.GetIdx())
        #Chem.SanitizeMol(AdsorbateMol)
        AdsorbateMol.UpdatePropertyCache(False)
        ## Get SurfMol
        AdsorbateSurfMol, AdsorbateToAdsorbateSurf, AdsorbateSurfToAdsorbate =  GetSubMolFromIdx(SurfIdxs,AdsorbateMol)
        ### Set up for matching
        SA = rdqueries.AtomNumEqualsQueryAtom(0)
        for idx in range(0,AdsorbateSurfMol.GetNumAtoms()):
            AdsorbateSurfMol.ReplaceAtom(idx,SA)
        SA.ExpandQuery(rdqueries.HasPropQueryAtom('CenterSurfAtom'))
        AdsorbateSurfMol.ReplaceAtom(0,SA)
        ## Get GasMol
        AdsorbateGasMol, AdsorbateToAdsorbateGas, AdsorbateGasToAdsorbate =  GetSubMolFromIdx(GasIdxs,AdsorbateMol)
        Chem.SanitizeMol(AdsorbateGasMol)
        AdsorbateGasMol = AdsorbateGasMol.GetMol()
        ## Match Surface
        ProjectedSurfIdxs = self.SurfMol.GetSubstructMatches(AdsorbateSurfMol)[0]
    
        # Combine Two mol
        NewMol = Chem.RWMol(Chem.CombineMols(self.SurfMol,AdsorbateGasMol))
        OccupiedSurfIdxs = set()
        for bond in AdsorbateMol.GetBonds():
            # Find Surface-Adsorbate bond
            SurfAtomIdx = None
            GasAtomIdx = None
            atoms = [bond.GetBeginAtom(),bond.GetEndAtom()]
            for atom in atoms:
                if atom.GetAtomicNum() in [1,6,8]:
                    GasAtomIdx = atom.GetIdx()
                else:
                    SurfAtomIdx = atom.GetIdx()
            # if the bond between adsorbate and surface
            if SurfAtomIdx is not None and GasAtomIdx is not None:
                GasMappedIdx = AdsorbateToAdsorbateGas[GasAtomIdx] + self.SurfMol.GetNumAtoms()
                SurfMappedIdx = ProjectedSurfIdxs[AdsorbateToAdsorbateSurf[SurfAtomIdx]]
                NewMol.AddBond(GasMappedIdx,SurfMappedIdx,order=Chem.rdchem.BondType.SINGLE)
                OccupiedSurfIdxs.add(SurfMappedIdx)
                
        # Set up property
        for idx in AdsorbateGasToAdsorbate:
            idx = idx + self.SurfMol.GetNumAtoms()
            atom = NewMol.GetAtomWithIdx(idx)
            atom.SetNumRadicalElectrons(0)
            
        M = Chem.Atom(0)
        for idx in OccupiedSurfIdxs:
            NewMol.ReplaceAtom(idx,M)
            
        # Indexing via isotope
        # This is done to record original index
        for i,atom in enumerate(NewMol.GetAtoms()):
            atom.SetIsotope(i+1) # start counting from 1 since 0 is default value
        
        return NewMol

    def GetCanonicalSmiles(self,s):
        reloadedmol = self.AddAdsorbateToSurf(s)
        reloadedmol = Chem.RWMol(RemoveLatticeAmbiguity(reloadedmol))
        reloadedmol = reloadedmol.GetMol()
        for atom in reloadedmol.GetAtoms():
            atom.SetIsotope(0)
            if atom.GetAtomicNum() in [1,6,8]:
                atom.SetNumRadicalElectrons(1)
            
        return Chem.MolToSmiles(reloadedmol)
    
    


def SetUpReaction(smiles):
    """
    Pair Enumeration rules
    """
    Rules = []
    # Prepare molecule
    Graph = Chem.MolFromSmiles(smiles,sanitize=False)
    # renumber for speed
    a = []
    s = []
    for atom in Graph.GetAtoms():
        if atom.GetAtomicNum() == 0:
            s.append(atom.GetIdx())
        else:
            a.append(atom.GetIdx())
    Graph = Chem.RenumberAtoms(Graph,a+s)
    Graph = Chem.RWMol(Graph)
    # set bond properties. Needed to limit connecting more than 3 C to one C
    for bond in Graph.GetBonds():
        if bond.GetBeginAtom().GetAtomicNum() == 0 or bond.GetEndAtom().GetAtomicNum() == 0 :
            bond.SetBondType(Chem.BondType.ZERO)
    for atom in Graph.GetAtoms():
        atom.SetNoImplicit(True)
    ## Set up molAtomMapNumber
    i = 1
    for atom in Graph.GetAtoms():
        atom.SetProp('molAtomMapNumber',str(i))
        i += 1
    ## Set Atom Type
    Anchors = list()
    for Atom in Graph.GetAtoms(): 
        if Atom.GetAtomicNum()==6:
            nS = 0
            for NBRAtom in Atom.GetNeighbors():
                if NBRAtom.GetAtomicNum() == 0:
                    nS += 1
            Atom.SetIntProp('nS',nS)
            Anchors.append(Atom.GetIdx())
        elif Atom.GetAtomicNum()==0:
            Atom.SetBoolProp('Occ',False)
            for NBRAtom in Atom.GetNeighbors():
                if NBRAtom.GetAtomicNum() == 6:
                    Atom.SetBoolProp('Occ',True)
                    break
    ## Check for Symmetry
    if len(set([Graph.GetAtomWithIdx(i).GetIntProp('nS') for i in Anchors])) == 1:
        symm = True
    else:
        symm = False
    #Chem.SanitizeMol(Graph)
    #Graph.UpdatePropertyCache(False)
    # Set up Product
    p = Graph.__copy__()
    ## set atom properties for occupide and nonoccupied surface atom.
    OSA = rdqueries.AtomNumEqualsQueryAtom(0)
    OSA.SetBoolProp('Occ',True)
    OSA.ExpandQuery(rdqueries.HasBoolPropWithValueQueryAtom('Occ',True))
    ## Set up unoccupied Surface Atom
    NOSA = rdqueries.AtomNumEqualsQueryAtom(0)
    NOSA.SetBoolProp('Occ',False)
    NOSA.ExpandQuery(rdqueries.HasBoolPropWithValueQueryAtom('Occ',False))
    # Rule 1 Set up
    ## Set up Reactant
    r = p.__copy__()
    ## Set up Other Anchor Atom
    AdsorbedAnchor = rdqueries.AtomNumEqualsQueryAtom(6)
    AdsorbedAnchor.ExpandQuery(rdqueries.HasIntPropWithValueQueryAtom('nS',
        r.GetAtomWithIdx(Anchors[1]).GetIntProp('nS')))
    AdsorbedAnchor.ExpandQuery(rdqueries.TotalValenceLessQueryAtom(3))
    AdsorbedAnchor.SetProp('molAtomMapNumber',r.GetAtomWithIdx(Anchors[1]).GetProp('molAtomMapNumber'))
    r.ReplaceAtom(Anchors[1],AdsorbedAnchor)
    ## replace surfaceatom with query atom
    for Atom in r.GetAtoms():
        if Atom.GetAtomicNum() == 0: 
            Occupied = False
            for NBRAtom in Atom.GetNeighbors():
                if NBRAtom.GetAtomicNum() == 6:
                    Occupied = True
                    break
            if not Occupied:
                NOSA.SetProp('molAtomMapNumber',Atom.GetProp('molAtomMapNumber'))
                r.ReplaceAtom(Atom.GetIdx(),NOSA)
    ## Remove Anchor Atom
    r.RemoveAtom(Anchors[0])
    if len(Chem.GetMolFrags(r)) == 1: # Fragmented can be [C].[*][*][*] which is like unconstrained bridge rule
        ## set reaction
        rxn = ChemicalReaction()
        ## add reactant
        #Chem.SanitizeMol(r)
        #r.UpdatePropertyCache(False)
        rxn.AddReactantTemplate(r.GetMol())
        ## add product
        #Chem.SanitizeMol(p)
        #p.UpdatePropertyCache(False)
        p.GetAtomWithIdx(Anchors[0]).SetBoolProp('NewAtom',True)
        rxn.AddProductTemplate(p.GetMol())
        rxn.Initialize()
        Rules.append(rxn)
    
    
    # Make rule2 if applicable
    if not symm:
        # Rule 1 Set up
        ## Set up Reactant
        r = p.__copy__()
        ## Set up Other Anchor Atom
        AdsorbedAnchor = rdqueries.AtomNumEqualsQueryAtom(6)
        AdsorbedAnchor.ExpandQuery(rdqueries.HasIntPropWithValueQueryAtom('nS',
            r.GetAtomWithIdx(Anchors[0]).GetIntProp('nS')))
        AdsorbedAnchor.ExpandQuery(rdqueries.TotalValenceLessQueryAtom(3))
        AdsorbedAnchor.SetProp('molAtomMapNumber',r.GetAtomWithIdx(Anchors[0]).GetProp('molAtomMapNumber'))
        r.ReplaceAtom(Anchors[0],AdsorbedAnchor)
        ## replace surfaceatom with query atom
        for Atom in r.GetAtoms():
            if Atom.GetAtomicNum() == 0: 
                Occupied = False
                for NBRAtom in Atom.GetNeighbors():
                    if NBRAtom.GetAtomicNum() == 6:
                        Occupied = True
                        break
                if not Occupied:
                    NOSA.SetProp('molAtomMapNumber',Atom.GetProp('molAtomMapNumber'))
                    r.ReplaceAtom(Atom.GetIdx(),NOSA)
        ## Remove Anchor Atom
        r.RemoveAtom(Anchors[1])
        if len(Chem.GetMolFrags(r)) == 1: # Fragmented can be [C].[*][*][*] which is like unconstrained bridge rule
            ## set reaction
            rxn = ChemicalReaction()
            ## add reactant
            #Chem.SanitizeMol(r)
            #r.UpdatePropertyCache(False)
            rxn.AddReactantTemplate(r.GetMol())
            ## add product
            p.GetAtomWithIdx(Anchors[1]).SetBoolProp('NewAtom',True)
            p.GetAtomWithIdx(Anchors[0]).ClearProp('NewAtom')
            #Chem.SanitizeMol(p)
            #p.UpdatePropertyCache(False)
            rxn.AddProductTemplate(p.GetMol())
            rxn.Initialize()
            Rules.append(rxn)
    return Rules

def SetUpRingReaction(smiles):
    """
    Ring Enumeration rules
    """
    Rules = []
    # Prepare molecule
    Graph = Chem.MolFromSmiles(smiles,sanitize=False)
    # renumber for speed
    a = []
    s = []
    for atom in Graph.GetAtoms():
        if atom.GetAtomicNum() == 0:
            s.append(atom.GetIdx())
        else:
            a.append(atom.GetIdx())
    Graph = Chem.RenumberAtoms(Graph,a+s)
    Graph = Chem.RWMol(Graph)
    # set bond properties. Needed to limit connecting more than 3 C to one C
    for bond in Graph.GetBonds():
        if bond.GetBeginAtom().GetAtomicNum() == 0 or bond.GetEndAtom().GetAtomicNum() == 0 :
            bond.SetBondType(Chem.BondType.ZERO)
    for atom in Graph.GetAtoms():
        atom.SetNoImplicit(True)
    ## Set up molAtomMapNumber
    i = 1
    for atom in Graph.GetAtoms():
        atom.SetProp('molAtomMapNumber',str(i))
        i += 1
    ## Set Atom Type
    Anchors = list()
    for Atom in Graph.GetAtoms(): 
        if Atom.GetAtomicNum()==6:
            nS = 0
            for NBRAtom in Atom.GetNeighbors():
                if NBRAtom.GetAtomicNum() == 0:
                    nS += 1
            Atom.SetIntProp('nS',nS)
            Anchors.append(Atom.GetIdx())
        elif Atom.GetAtomicNum()==0:
            Atom.SetBoolProp('Occ',False)
            for NBRAtom in Atom.GetNeighbors():
                if NBRAtom.GetAtomicNum() == 6:
                    Atom.SetBoolProp('Occ',True)
                    break
    #Chem.SanitizeMol(Graph)
    # Set up Product
    p = Graph.__copy__()
    ## set atom properties for occupide and nonoccupied surface atom.
    OSA = rdqueries.AtomNumEqualsQueryAtom(0)
    OSA.SetBoolProp('Occ',True)
    OSA.ExpandQuery(rdqueries.HasBoolPropWithValueQueryAtom('Occ',True))
    ## Set up unoccupied Surface Atom
    NOSA = rdqueries.AtomNumEqualsQueryAtom(0)
    NOSA.SetBoolProp('Occ',False)
    NOSA.ExpandQuery(rdqueries.HasBoolPropWithValueQueryAtom('Occ',False))
    
    # Check whether it's a ring or chain
    bonds = []
    for anchorpair in itertools.combinations(Anchors,2):
        if Graph.GetBondBetweenAtoms(anchorpair[0],anchorpair[1]):
            bonds.append(anchorpair)

    if len(bonds) == 2: # Chain
        AnchorsToRemoves = list(set(bonds[0]).intersection(bonds[1]))
    else:
        AnchorsToRemoves = Anchors.copy()

    uniquesmiles = []
    for AnchorToRemove in AnchorsToRemoves:
        anc = Anchors.copy()
        del anc[anc.index(AnchorToRemove)]
        p.GetAtomWithIdx(AnchorToRemove).SetBoolProp('NewAtom',True)
        for a in anc:
            p.GetAtomWithIdx(a).ClearProp('NewAtom')

        # Rule 1 Set up
        ## Set up Reactant
        r = p.__copy__()
        ## Set up Anchor
        for an in anc:
            AdsorbedAnchor = rdqueries.AtomNumEqualsQueryAtom(6)
            AdsorbedAnchor.ExpandQuery(rdqueries.HasIntPropWithValueQueryAtom('nS',
                r.GetAtomWithIdx(an).GetIntProp('nS')))
            AdsorbedAnchor.ExpandQuery(rdqueries.TotalValenceLessQueryAtom(3))
            AdsorbedAnchor.SetProp('molAtomMapNumber',r.GetAtomWithIdx(an).GetProp('molAtomMapNumber'))
            r.ReplaceAtom(an,AdsorbedAnchor)
        ## replace surfaceatom with query atom
        for Atom in r.GetAtoms():
            if Atom.GetAtomicNum() == 0: 
                Occupied = False
                for NBRAtom in Atom.GetNeighbors():
                    if NBRAtom.GetAtomicNum() == 6:
                        Occupied = True
                        break
                if not Occupied:
                    NOSA.SetProp('molAtomMapNumber',Atom.GetProp('molAtomMapNumber'))
                    r.ReplaceAtom(Atom.GetIdx(),NOSA)
        ## Remove Anchor Atom
        r.RemoveAtom(AnchorToRemove)
        smiles = Chem.MolToSmiles(r)
        if smiles not in uniquesmiles:
            uniquesmiles.append(smiles)
            ## set reaction
            rxn = ChemicalReaction()
            ## add reactant
            #Chem.SanitizeMol(r)
            #r.UpdatePropertyCache(False)
            rxn.AddReactantTemplate(r.GetMol())
            ## add product
            #Chem.SanitizeMol(p)
            #p.UpdatePropertyCache(False)
            rxn.AddProductTemplate(p.GetMol())
            rxn.Initialize()
            Rules.append(rxn)

    return Rules

class BridgeRule(object): 
    _a=2.125210
    _b=-0.992577
    def __init__(self,xyz,siteidx,maxbridge=12):
        sitexyz = []
        for sidx in siteidx:
            xyzs = []
            for s in sidx:
                xyzs.append(xyz[s,:])
            sitexyz.append(np.mean(xyzs,0))
        sitexyz = np.array(sitexyz)[:,0:2]
        self.Data = {idxs:[[] for _ in range(maxbridge-3)] for idxs in siteidx}
        dists = pdist(sitexyz)
        n=0
        for i in range(len(siteidx)):
             for j in range(i+1,len(siteidx)):
                 nmaxgaslength = self._a*dists[n]+self._b
                 for k in range(maxbridge-3): # Index starts from gas length 3
                     if k+3>nmaxgaslength:
                         self.Data[siteidx[i]][k].append(siteidx[j])
                         self.Data[siteidx[j]][k].append(siteidx[i])
                 n+=1
        self.C = Chem.Atom('C')
        
    def _AnalyzeReactant(self,reactant):
        """
        Set up reactant properties. Identify Bridges
        """
                # Identify Bridges
        ## find all non-surface bound atoms
        AtomsToCheckList = list()
        for a in reactant.GetAtoms():
            if a.GetAtomicNum() == 6:
                adsorbed = False
                for na in a.GetNeighbors():
                    if na.GetAtomicNum() ==0:
                        adsorbed = True
                        break
                if not adsorbed:
                    a.SetBoolProp('Adsorbed',False)
                    AtomsToCheckList.append(a)
                else:
                    a.SetBoolProp('Adsorbed',True)
        HangingC_Anchor_BridgeLens = list()
        """
        HangingC_Anchor_BridgeLens:
            List of [HangingC, AnchorInfo]
        AnchorInfo:
            List of [Anchor Idx (using Isotope to refer to original lattice), Bridge Length]
        """
        while AtomsToCheckList:
            # initialize
            # here a single bridge is identified
            NeighborsToCheck = [AtomsToCheckList.pop()]
            CheckedAtomIdx = []
            Anchors = []
            HangingCs = []
            while NeighborsToCheck:
                AtomBeingChecked = NeighborsToCheck.pop()
                CheckedAtomIdx.append(AtomBeingChecked.GetIdx())
                if AtomBeingChecked.GetBoolProp('Adsorbed'): # if it's adsorbed, it's anchor
                    Anchors.append(AtomBeingChecked)
                else:
                    # if not anchor, check whether it's hanging, or to continue search 
                    nhbs = AtomBeingChecked.GetNeighbors()
                    if len(nhbs) == 1: # This is a Hanging atom
                        HangingCs.append(AtomBeingChecked.GetIdx())
                    for neighbor_atom in AtomBeingChecked.GetNeighbors():
                        if neighbor_atom.GetIdx() not in CheckedAtomIdx:
                            NeighborsToCheck.append(neighbor_atom)

            # Remove checked atoms
            AtomsToCheckList = [atom for atom in AtomsToCheckList if atom.GetIdx() not in CheckedAtomIdx]
            
            # For Path through organic atoms
            BondToBreak = list()
            for Atom in Anchors:
                for Bond in Atom.GetBonds():
                    if Bond.GetOtherAtom(Atom).GetIdx() not in CheckedAtomIdx:
                        BondToBreak.append(Bond.GetIdx())
            MolForMolPath = Chem.FragmentOnBonds(reactant,list(set(BondToBreak)))
            
            # Get Bridge Length      
            for HangingC in HangingCs:
                Anchor_BridgeLen = []
                for Anchor in Anchors:
                    bridgelen = len(Chem.GetShortestPath(MolForMolPath,HangingC,Anchor.GetIdx()))
                    Anchor_BridgeLen.append([frozenset([GetBeforeIdx(na) for na in Anchor.GetNeighbors()
                    if na.GetAtomicNum() == 0]),bridgelen])
                HangingC_Anchor_BridgeLens.append([HangingC,Anchor_BridgeLen])
        return HangingC_Anchor_BridgeLens
    
    def RunReactants(self,reactants):
        # Initialize
        reactant = Chem.RWMol(reactants[0])
        HangingC_Anchor_BridgeLens = self._AnalyzeReactant(reactant)

        ## End of While
        products = []
        # iterate over Each HangingC+Anchors
        for HangingC_Anchor_BridgeLen in HangingC_Anchor_BridgeLens:
            HangingC = HangingC_Anchor_BridgeLen[0]
            AvailableSites = []
            # There could be multiple anchor. Intersecting available sites are foudn
            for Anchor_BridgeLen in HangingC_Anchor_BridgeLen[1]: 
                AvailableSites.append(set(self.Data[Anchor_BridgeLen[0]][Anchor_BridgeLen[1]-2]))
            AvailableSites = set.intersection(*AvailableSites)
            # Add Bond
            for sidx in AvailableSites:
                p = reactant.__copy__()
                NewAnchorCIdx = p.AddAtom(self.C)
                p.AddBond(HangingC, NewAnchorCIdx, order=Chem.BondType.SINGLE)
                for s in sidx:
                    p.AddBond(s, NewAnchorCIdx, order=Chem.BondType.SINGLE)
                products.append((p,))
        return products
    
    def ConnectBrgNewAtom(self,reactants):
        # Initialize
        reactant = Chem.RWMol(reactants[0])
        HangingC_Anchor_BridgeLens = self._AnalyzeReactant(reactant)
        
        # Find the new atom Anchor
        NewAtomAnchor = []
        for atom in reactant.GetAtoms():
            if atom.HasProp('NewAtom'):
                NewAtomIdx = atom.GetIdx()
                for na in atom.GetNeighbors():
                    if na.GetAtomicNum() == 0:
                        NewAtomAnchor.append(GetBeforeIdx(na))
        NewAtomAnchor = frozenset(NewAtomAnchor)
        ## End of While
        products = []
        # iterate over Each HangingC+Anchors
        for HangingC_Anchor_BridgeLen in HangingC_Anchor_BridgeLens:
            HangingC = HangingC_Anchor_BridgeLen[0]
            AvailableSites = []
            # There could be multiple anchor. Intersecting available sites are foudn
            for Anchor_BridgeLen in HangingC_Anchor_BridgeLen[1]: 
                AvailableSites.append(set(self.Data[Anchor_BridgeLen[0]][Anchor_BridgeLen[1]-2]))
            AvailableSites = set.intersection(*AvailableSites)
            if NewAtomAnchor in AvailableSites:
                p = reactant.__copy__()
                p.AddBond(HangingC,NewAtomIdx, order=Chem.BondType.SINGLE)
                products.append((p,))
        return products


def GetBeforeIdx(atom):
    iso = atom.GetIsotope()
    if iso != 0:
        return atom.GetIsotope()-1
    else:
        return None

def CleanUp(mol):
    # Set properties
    for Atom in mol.GetAtoms():
        if Atom.GetAtomicNum() == 0: # update occupancy
            Occupied = False
            for na in Atom.GetNeighbors():
                if na.GetAtomicNum() == 6:
                    Occupied = True
                    break
            Atom.SetBoolProp('Occ',Occupied)
        if Atom.GetAtomicNum() == 6: # This update Total valence
            Atom.UpdatePropertyCache()
def SetUpConstraintMol(s):
    mol = Chem.RWMol(Chem.MolFromSmiles(s,sanitize=False))
    todelete = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            occupied = False
            for na in atom.GetNeighbors():
                if na.GetAtomicNum() == 6:
                    occupied = True
                    break
            if not occupied:
                todelete.append(atom.GetIdx())
    # Remove unoccupied atoms
    path = []
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() not in todelete or bond.GetEndAtomIdx() not in todelete:
            path.append(bond.GetIdx())
    mol = Chem.PathToSubmol(mol,path)
#    # bond set
#    for bond in mol.GetBonds():
#        if bond.GetBeginAtom().GetAtomicNum() == 0 or bond.GetEndAtom().GetAtomicNum() == 0 :
#            bond.SetBondType(Chem.BondType.ZERO)
    # Renumbers
    a = []
    s = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            a.append(atom.GetIdx())
        else:
            s.append(atom.GetIdx())
    aa = []
    unprocessed = [a[0]]
    while unprocessed:
        i = unprocessed.pop()
        aa.append(i)
        atom = mol.GetAtomWithIdx(i)
        for atom in atom.GetNeighbors():
            if atom.GetIdx() not in aa and atom.GetAtomicNum() == 6 :
                unprocessed.insert(0,atom.GetIdx())
    return Chem.RenumberAtoms(mol,aa+s)
    
                

def CheckConfig(s):

    # Optimize
    out, atoms = GraphToOptStruc(s,SurfAtomNum=0)
    if out == -1:
        return -1, s # Embeding failde
    ans = atoms.get_atomic_numbers()

    # set rcov
    ## 1.65 is max for CC, and 0.8 is the min for CC
    rcov = np.zeros((len(atoms)))
    rcov[ans==6] = 0.7174
    rcov[ans==0] = 1.3695565
    # Determine connectivity of the organic atoms
    pos = atoms.get_positions()
    # Get pairwise distance
    dist = pdist(pos)

    
    # See if CC distance is too short or long
    # Check if atoms are too close
    mol = Chem.MolFromSmiles(s,sanitize=False)
    CC=[]
    for bond in mol.GetBonds():
        if bond.GetBeginAtom().GetAtomicNum() == 6 and bond.GetEndAtom().GetAtomicNum() == 6:
            i = bond.GetBeginAtom().GetIdx()
            j = bond.GetEndAtom().GetIdx()
            if j < i:
                t = i
                i = j
                j = t
            CC.append([i,j])
    CC = np.array(CC)
    n = mol.GetNumAtoms()
    CCIdx = CC[:,0]*n + CC[:,1] - CC[:,0]*(CC[:,0]+1)/2 - CC[:,0] - 1
    CCIdx = CCIdx.astype(int)
    if np.any(dist[CCIdx] > 1.65) or np.any(dist[CCIdx] < 0.8):
        return -2, s # distance criterum didn't meet
    
    
    # get index
    index = np.array(list(combinations(range(len(atoms)),2)))
    # Get distance criteria
    dist_max = np.sum(rcov[index],axis=1)*1.15
    # Bool mask for atoms with bond
    YesBond = dist<dist_max
    # Make Mol
    RdkitMol = Chem.RWMol(Chem.Mol())
    for an in ans:
        atom = Chem.Atom(int(an))
        RdkitMol.AddAtom(atom)
    for i,j in index[YesBond]:
        RdkitMol.AddBond(int(i),int(j),order=Chem.rdchem.BondType.SINGLE)
    if Chem.MolToSmiles(RdkitMol) == s:
        return 1, s
    else:
        return -3, s # Wrong smiles

CovalentRadius = {'Ag':1.46, 'Al':1.11, 'As':1.21, 'Au':1.21, 'C':0.77, 'Ca':1.66, 'Cd':1.41,
    'Co':1.21, 'Cr':1.26, 'Cu':1.21, 'Fe':1.26, 'Ga':1.16, 'Ge':1.22, 'H':0.37,
    'In':1.41, 'Ir':1.21, 'Mn':1.26, 'Mo':1.31, 'N':0.74, 'Na':1.66, 'Nb':1.31,
    'Ni':1.21, 'O':0.74, 'Os':1.16, 'Pb':1.66, 'Pd':1.26, 'Pt':1.21, 'Re':1.21,
    'Rh':1.21, 'Ru':1.16, 'S':1.04, 'Sb':1.41, 'Se':1.17, 'Si':1.17, 'Sn':1.4,
    'Ti':1.26, 'V':1.21, 'W':1.21, 'Zn':1.21}

class AtomDB(object):
    def __init__(self):

        PT = Chem.GetPeriodicTable()
        
        self.SurfaceAtomicNumbers = set()
        self.AdsorbateAtomicNumbers = set()
        self.CovalentRadius = dict()
        
        for Symbol in SurfaceElements:
            self.SurfaceAtomicNumbers.add(PT.GetAtomicNumber(Symbol))
        self.SurfaceAtomicNumbers.add(0)
        for Symbol in AdsorbateElements:
            self.AdsorbateAtomicNumbers.add(PT.GetAtomicNumber(Symbol))
        for Symbol in CovalentRadius:
            self.CovalentRadius[PT.GetAtomicNumber(Symbol)] = CovalentRadius[Symbol]
            
        
    def IsAdsorbateAtomNum(self,AtomicNumber):
        if AtomicNumber in self.AdsorbateAtomicNumbers:
            return True
        return False
    
    def IsSurfaceAtomNum(self,AtomicNumber):
        if AtomicNumber in self.SurfaceAtomicNumbers:
            return True
        return False
        
    def GetCovalentRadius(self, AtomicNumber):
        if AtomicNumber in self.CovalentRadius:
            return self.CovalentRadius[AtomicNumber]
        else:
            raise(NotImplementedError, 'Missing covalent radius information')
            
def IsSurfaceAtomOccupied(Atom):
    # Assumes supplied atom is surface atom
    if not isinstance(Atom,(Chem.Atom,Chem.QueryAtom)):
        raise TypeError('Atom has to be rdkit.Chem.rdchem.Atom/QueryAtom')
    for NeighborAtom in Atom.GetNeighbors():
        if ADB.IsAdsorbateAtomNum(NeighborAtom.GetAtomicNum()):
            return True
            break
    return False

ADB = AtomDB()

def GetCovalentRadius(AtomicNumber):
    # Assumes supplied atom is adsorbate atom
    if not isinstance(AtomicNumber,(int,np.int64,np.int32)):
        raise TypeError('AtomicNumber has to be int')
    return ADB.GetCovalentRadius(AtomicNumber)
    

def IsAdsorbateAtomNum(AtomicNumber):
    if not isinstance(AtomicNumber,(int,np.int32,np.int64)):
        raise TypeError('AtomicNumber has to be int')
    return ADB.IsAdsorbateAtomNum(AtomicNumber)
    
def IsAdsorbateAtomAdsorbed(Atom):
    # Assumes supplied atom is adsorbate atom
    if not isinstance(Atom,(Chem.Atom,Chem.QueryAtom)):
        raise TypeError('Atom has to be rdkit.Chem.rdchem.Atom/QueryAtom')
    for NeighborAtom in Atom.GetNeighbors():
        if ADB.IsSurfaceAtomNum(NeighborAtom.GetAtomicNum()):
            return True
            break
    return False

def IsSurfaceAtomNum(AtomicNumber,ZeroIsMetal=True):
    if not isinstance(AtomicNumber,(int,np.int32,np.int64)):
        raise TypeError('AtomicNumber has to be int')
    if ZeroIsMetal and AtomicNumber==0:
        return True
    else:
        return ADB.IsSurfaceAtomNum(AtomicNumber)


def GetNumSurfAtomNeighbor(Atom):
    # Assumes supplied atom is adsorbate atom
    if not isinstance(Atom,(Chem.Atom,Chem.QueryAtom)):
        raise TypeError('Atom has to be rdkit.Chem.rdchem.Atom/QueryAtom')
    n = 0
    for NeighborAtom in Atom.GetNeighbors():
        if NeighborAtom.HasProp('Type'):
            if NeighborAtom.GetProp('Type') == 'S':
                n += 1
        elif ADB.IsSurfaceAtomNum(NeighborAtom.GetAtomicNum()):
            n += 1
    return n


def SetAdsorbateMolAtomProps(Mol,ZeroIsMetal = True):
    if not isinstance(Mol,(Chem.Mol,Chem.RWMol,Chem.EditableMol)):
        raise TypeError('Mol has to be rdkit.Chem.rdchem.Mol/RWMol/EditableMol')
    
    #Set Atom Type
    for Atom in Mol.GetAtoms():
        if Atom.HasProp('Type'):
            pass
        if ADB.IsAdsorbateAtomNum(Atom.GetAtomicNum()):
            Atom.SetProp('Type','A')
        elif ADB.IsSurfaceAtomNum(Atom.GetAtomicNum()) or (ZeroIsMetal and Atom.GetAtomicNum() == 0):
            Atom.SetProp('Type','S')
            Atom.SetProp('smilesSymbol','M')
            if IsSurfaceAtomOccupied(Atom):
                Atom.SetBoolProp('Occupied',True)
            else:
                Atom.SetBoolProp('Occupied',False)
    # Set Bond Type and assign radical electrons
    for Bond in Mol.GetBonds():
        if Bond.GetBeginAtom().GetProp('Type') == 'S' or Bond.GetEndAtom().GetProp('Type') == 'S':
            Bond.SetBondType(Chem.rdchem.BondType.ZERO)
        else:
             Bond.SetBondType(Chem.rdchem.BondType.SINGLE)   
    Chem.AssignRadicals(Mol)
    # Set smilesSymbol and Adsorbed
    for Atom in Mol.GetAtoms():
        if Atom.GetProp('Type') == 'A':
            NSurf = GetNumSurfAtomNeighbor(Atom)
            if Atom.GetAtomicNum() != 1:
                Atom.SetProp('smilesSymbol',Atom.GetSymbol() + str(Atom.GetNumRadicalElectrons())+ str(NSurf))
            if NSurf != 0:
                Atom.SetBoolProp('Adsorbed',True)
            else:
                Atom.SetBoolProp('Adsorbed',False)
            if Atom.GetAtomicNum() != 1 and Atom.GetNumRadicalElectrons() == 0:
                Atom.SetProp('smilesSymbol','[' + Atom.GetSymbol() + '0]')
        
        else:
            ValueError, 'Unrecognized Atom Element Type! See GraphLearning.Settings'


def _PretreatSMILESorMol(SMILESorMol):

    if isinstance(SMILESorMol,str):
        SMILESorMol = Chem.MolFromSmiles(SMILESorMol,sanitize=False)
        SetAdsorbateMolAtomProps(SMILESorMol)
#        for bond in SMILESorMol.GetBonds():
#            if bond.GetBeginAtom().GetProp('Type') == 'S' or bond.GetBeginAtom().GetProp('Type') == 'S':
#                bond.SetBondType(Chem.rdchem.BondType.ZERO)
#        SMILESorMol = AllChem.AddHs(SMILESorMol)
        
    if isinstance(SMILESorMol,(Chem.Mol,Chem.EditableMol,Chem.RWMol)):
        SMILESorMol = SMILESorMol.__copy__()
        SetAdsorbateMolAtomProps(SMILESorMol,ZeroIsMetal=True)
        mol = Chem.RWMol(SMILESorMol)
        Chem.SanitizeMol(mol)
        for i in range(0,mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            if atom.GetProp('Type') == 'S':
                SurfAtom = Chem.Atom(78) # Platinum. This needs to be done otherwise rdkit forcefield does not work
                SurfAtom.SetProp('Type','S')
                SurfAtom.SetBoolProp('Occupied',atom.GetBoolProp('Occupied'))
                mol.ReplaceAtom(i,SurfAtom)
        mol.UpdatePropertyCache()
    else:
        raise TypeError('Unrecognized adsorbate graph input')

    return mol

def GraphToOptStruc(SMILESorMol, OutputPath=None, LatticeConstant=3.924, Quiet = True, SurfAtomNum = 46,ZStrain=150.0):
    # Initialize
    mol = _PretreatSMILESorMol(SMILESorMol)
    NearestNeighborDistance = LatticeConstant/np.sqrt(2)

    # Get list of Surface Atom Indices
    SurfIdxs = list()
    for atom in mol.GetAtoms():
        if atom.GetProp('Type') == 'S':
            SurfIdxs.append(atom.GetIdx())
    
    # Get Surface conformer
    output = AllChem.EmbedMolecule(mol)
    if output == -1:
        return -1,None # Embedding failed
    conf = mol.GetConformer(0)
    if len(SurfIdxs) == 0:
        # Gas Phase Molecule
        ff = AllChem.UFFGetMoleculeForceField(mol)
        # Optimize Molecule
        ff.Initialize()
        output = ff.Minimize()
        
    else:
        # Set up Surface Coordinate
        """
        Algorithm:
        Set first and second atoms to eliminate two degree of freedom, and then start 
        setting other atom's position based on first two
        """
        
        if len(SurfIdxs) == 1:
            conf.SetAtomPosition(SurfIdxs[0], (0,0,0))
        elif len(SurfIdxs) == 2:
            conf.SetAtomPosition(SurfIdxs[0], (0,0,0))
            conf.SetAtomPosition(SurfIdxs[1], (NearestNeighborDistance,0,0))
        elif len(SurfIdxs) > 2:
            # Error check
            for idx in SurfIdxs:
                Atom = mol.GetAtomWithIdx(idx)
                NSurfNeighbor = 0
                for NeighborAtom in Atom.GetNeighbors():
                    if NeighborAtom.GetProp('Type') == 'S':
                        NSurfNeighbor += 1
                if NSurfNeighbor < 2:
                    raise ValueError('Dangling Surface Atom detected. Make sure surface atoms are attached to at least 2 other connected surface atoms')
                    
            # Vector that checks whether or not surface atom is plotted
            Plotted = list()
            # (N)on-(p)lotted (S)urface Atom (I)dx (T)o (P)lotted (S)urface (N)eighbor (C)ount     
            NPSITPSNC  = defaultdict(int) 
            # plot first atom
            conf.SetAtomPosition(SurfIdxs[0], (0,0,0))
            Plotted.append(SurfIdxs[0])
            FirstAtom = mol.GetAtomWithIdx(SurfIdxs[0])
            # plot second atom and update NPSITPSNC
            for NeighborAtom in FirstAtom.GetNeighbors():
                if NeighborAtom.GetProp('Type') == 'S':
                    NPSITPSNC[NeighborAtom.GetIdx()] += 1
                    SecondAtom = NeighborAtom
            
            conf.SetAtomPosition(SecondAtom.GetIdx(), (NearestNeighborDistance,0,0))
            Plotted.append(SecondAtom.GetIdx())
            del NPSITPSNC[SecondAtom.GetIdx()]
            for NeighborAtom in SecondAtom.GetNeighbors():
                if NeighborAtom.GetProp('Type') == 'S' and NeighborAtom.GetIdx() not in Plotted:
                    NPSITPSNC[NeighborAtom.GetIdx()] += 1
            

            # plot other atoms
            while len(NPSITPSNC) != 0:
                # Find Atom with more than two plotted neighbor atom
                NonPlottedIdx = list(NPSITPSNC.keys())
                random.shuffle(NonPlottedIdx)
                for AtomIdx in NonPlottedIdx:
                    if NPSITPSNC[AtomIdx] >= 2:
                        Atom = mol.GetAtomWithIdx(AtomIdx)
                        break
                
                # Find Two Neighbor Atoms that are connected to each other.
                ## Find plotted Neighbors
                NeighborIdx = list()
                for NeighborAtom in Atom.GetNeighbors():
                    if NeighborAtom.GetProp('Type') == 'S' and NeighborAtom.GetIdx() in Plotted:
                        NeighborIdx.append(NeighborAtom.GetIdx())
                
                match = False
                ## Find two atoms that are connected to each other
                for idx in NeighborIdx:
                    # get neighbor atom object
                    Atom1 = mol.GetAtomWithIdx(idx)
                    # see if its neighbor is also neighbor of picked atom
                    for Atom1Neighbor in Atom1.GetNeighbors():
                        if Atom1Neighbor.GetIdx() in NeighborIdx:
                            Atom2 = Atom1Neighbor
                            match = True
                            break
                    if match:
                        break
                
                if not match:
                    continue # There could be non plotted surface atom with two plotted surface atom that are not neighbor to each other
                else:
                    # make a vector relative to the first atom                    
                    vector = np.array([NearestNeighborDistance/2,3 ** (0.5)/2*NearestNeighborDistance])
                    # rotate the vector
                    atom1pos = conf.GetAtomPosition(Atom1.GetIdx())
                    atom2pos = conf.GetAtomPosition(Atom2.GetIdx())
                    angle = np.arctan2((atom2pos.y-atom1pos.y),(atom2pos.x-atom1pos.x))
                    rotation_matrix = np.matrix([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
                    vector = np.dot(rotation_matrix, vector)
                    # move to where first atom is
                    vector +=  [atom1pos.x, atom1pos.y]
                    # vector is set so that it's normal to the bond direction,
                    # however, the space could be occupied by other surface atom,
                    # so we check for duplicate and if found, assign negative normal direction
                    for idx in Plotted:
                        if round(vector[0,0] - conf.GetAtomPosition(idx).x,2) == 0 \
                        and round(vector[0,1] - conf.GetAtomPosition(idx).y,2) == 0:
                            vector = np.array([NearestNeighborDistance/2,-3 ** (0.5)/2*NearestNeighborDistance])
                            vector = np.dot(rotation_matrix, vector)
                            vector +=  [atom1pos.x, atom1pos.y]
                            break
                    conf.SetAtomPosition(AtomIdx, (vector[0,0],vector[0,1],0))
                    # Update
                    Plotted.append(AtomIdx)
                    del NPSITPSNC[AtomIdx]
                    for NeighborAtom in Atom.GetNeighbors():
                        if NeighborAtom.GetProp('Type') == 'S' and NeighborAtom.GetIdx() not in Plotted:
                            NPSITPSNC[NeighborAtom.GetIdx()] += 1  
                    

    
        ff = _SetUpForceField(mol,InitialGuessRun=True,ZStrain=ZStrain)
        # Optimize Molecule
        ff.Initialize()
        output = ff.Minimize();
#        output = ff.Minimize(maxIts=1000000, forceTol=1e-10, energyTol=1e-010);
#        
        ff = _SetUpForceField(mol,InitialGuessRun=False,ZStrain=ZStrain)
        # Optimize Molecule
        ff.Initialize()
        output = ff.Minimize(maxIts=1000000, forceTol=1e-10, energyTol=1e-010);
    
    if output == -1:
        output = -2
    # report minimization result
    if not Quiet:
        if output == -2:
            print('Minimization did not converge ('+str(output)+')')
        else:
            print('Minimization Successful ('+str(output)+')')
            
    # Output to XSD
    ## Save atomic number
    AtomicNumbers= list()
    for atom in mol.GetAtoms():
        AtomicNumber = atom.GetAtomicNum()
        if AtomicNumber in [0,78]:
            AtomicNumber = SurfAtomNum
        AtomicNumbers.append(AtomicNumber)
    
    ## Save Positions
    positions = list()
    for i in range(0, mol.GetNumAtoms()):
        pos = mol.GetConformer().GetAtomPosition(i)
        positions.append([pos.x, pos.y, pos.z])
    positions = np.array(positions)
    
    ## Make ASE atoms object
    aseatoms = ASEAtoms(numbers = AtomicNumbers, positions = positions)
#    ASEAtoms.cell = np.ones((3,3))
    if OutputPath:
        ## make connectivity object
        connectivity = np.zeros((mol.GetNumAtoms(),mol.GetNumAtoms()))
        for i in range(0,mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            for neighboratom in atom.GetNeighbors():
                connectivity[i,neighboratom.GetIdx()] = 1
        
        ## Make xsd file
        write(OutputPath,aseatoms,connectivity = connectivity)
    return output, aseatoms

def _SetUpForceField(mol, SetHybridization = True, AdsorbateSurfaceRepulsion = True, cell = np.diag((1,1,1)), ZLattVecI = 2, InitialGuessRun=False,ZStrain=150.0):
    ## Compute Perpendicular direction and other diections
    OtherVeci = [i for i in [0,1,2] if i != ZLattVecI]
    Zvector = cell[ZLattVecI,:]
    Zvector = Zvector/np.linalg.norm(Zvector)
    Xvector = cell[OtherVeci[0],:]
    Xvector = Xvector/np.linalg.norm(Xvector)
    XZPerpvector = np.cross(Zvector,Xvector)
    XZPerpvector = XZPerpvector/np.linalg.norm(XZPerpvector)
    Yvector = cell[OtherVeci[1],:]
    Yvector = Yvector/np.linalg.norm(Yvector)
    
    CenterOfSurf = list()
    for i in range(0, mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetProp('Type') == 'S':
            CenterOfSurf.append(mol.GetConformer().GetAtomPosition(i))
    CenterOfSurf = np.average(CenterOfSurf,axis=0)
    # Manually add force field

    
    # Set up hybrdization: Some bugs in Rdkit. hydrogen get positioned on top of each other
    AtomsToSetFF = []
    if SetHybridization:
        for atom in mol.GetAtoms():
            if atom.GetProp('Type') == 'A' and atom.GetNumRadicalElectrons() > 0:
                NumNeighbors = atom.GetTotalDegree()
                if 'S' in [na.GetProp('Type') for na in atom.GetNeighbors()]:
                    if atom.GetAtomicNum() == 6:
                        if NumNeighbors == 4:
                            atom.SetHybridization(Chem.HybridizationType.SP3)
                        elif NumNeighbors == 3:
                            atom.SetHybridization(Chem.HybridizationType.SP2)
                        elif NumNeighbors == 2:
                            atom.SetHybridization(Chem.HybridizationType.SP)
                        elif NumNeighbors == 1:
                            atom.SetHybridization(Chem.HybridizationType.SP)
                    elif atom.GetAtomicNum() == 8:
                        if NumNeighbors == 2:
                            atom.SetHybridization(Chem.HybridizationType.SP3)
                        elif NumNeighbors == 1:
                            atom.SetHybridization(Chem.HybridizationType.SP2)
                else: # These are error prone and parameters are set later
                    atom.SetHybridization(Chem.HybridizationType.UNSPECIFIED)
                    AtomsToSetFF.append(atom.GetIdx())
    ff = AllChem.UFFGetMoleculeForceField(mol)

    # Take care of Error Prone one. Just manually setting what UFF should have done
    
    if SetHybridization:
        for atomi in AtomsToSetFF:
            atom = mol.GetAtomWithIdx(atomi)
            NumNeighbors = atom.GetTotalDegree()
            if atom.GetAtomicNum() == 6:
                if NumNeighbors == 4:
                    atom.SetHybridization(Chem.HybridizationType.SP3)
                elif NumNeighbors == 3:
                    atom.SetHybridization(Chem.HybridizationType.SP2)
                elif NumNeighbors == 2:
                    atom.SetHybridization(Chem.HybridizationType.SP)
                elif NumNeighbors == 1:
                    atom.SetHybridization(Chem.HybridizationType.SP)
            elif atom.GetAtomicNum() == 8:
                if NumNeighbors == 2:
                    atom.SetHybridization(Chem.HybridizationType.SP3)
                elif NumNeighbors == 1:
                    atom.SetHybridization(Chem.HybridizationType.SP2)
        
        
        for atomi in AtomsToSetFF:
            atom = mol.GetAtomWithIdx(atomi)
            idx = [na.GetIdx() for na in atom.GetNeighbors()]
            # Bond stretch
            for a1 in idx:
                ka, r = rdForceFieldHelpers.GetUFFBondStretchParams(mol,a1,atomi)
                ff.UFFAddDistanceConstraint(a1,atomi,False,r,r,ka)
            # Angle
            for a1,a2 in combinations(idx,2):
                ka, ang = rdForceFieldHelpers.GetUFFAngleBendParams(mol,a1,atomi,a2)
                ff.UFFAddAngleConstraint(a1,atomi,a2,False,ang,ang,ka)
            # Angle between neighbor atom and the atom
            for natom in atom.GetNeighbors():
                idx = [na.GetIdx() for na in natom.GetNeighbors() if na.GetIdx() != atomi and IsAdsorbateAtomNum(na.GetAtomicNum())]
                for j in idx:
                    ka, ang = rdForceFieldHelpers.GetUFFAngleBendParams(mol,j,natom.GetIdx(),atomi)
                    ff.UFFAddAngleConstraint(j,natom.GetIdx(),atomi,False,ang,ang,ka)
    
    
    ## Vertical
    pos = CenterOfSurf + Xvector*10000000
    IdxSurfFixedPlusX = ff.AddExtraPoint(pos[0],pos[1],pos[2],fixed=True)-1
    
    pos = CenterOfSurf + XZPerpvector*10000000
    IdxSurfFixedPlusXZPerp = ff.AddExtraPoint(pos[0],pos[1],pos[2],fixed=True)-1
    ## Fix surface atom
    for atom in mol.GetAtoms():
        
        if IsSurfaceAtomNum(atom.GetAtomicNum()) or (atom.HasProp('smilesSymbol') and atom.GetProp('smilesSymbol') == 'M'):

            # More Flexible in Z direction but not X and Y direction
            ff.UFFAddPositionConstraint(atom.GetIdx(), 0.0, 10.0e2)  
            ff.UFFAddDistanceConstraint(IdxSurfFixedPlusX, atom.GetIdx(), True, 0.0,0.0, 10.0e4)
            ff.UFFAddDistanceConstraint(IdxSurfFixedPlusXZPerp, atom.GetIdx(), True, 0.0,0.0, 10.0e4)
            

    
    ## Find surface-adsorbates bond
    surfacebond = list()
    for bonds in mol.GetBonds():
        atom1 = bonds.GetBeginAtom()
        atom2 = bonds.GetEndAtom()
        surfbond = 0;
        # non-organic atoms are treated as surface atom
        if IsAdsorbateAtomNum(atom1.GetAtomicNum()):
            surfbond += 1; 
        if IsAdsorbateAtomNum(atom2.GetAtomicNum()):
            surfbond += 1;
        # surfbond == 0, then two bonded atoms are metal.
        # surfbond == 1, then two bonded atoms are metal and organic atom.
        # surfbond == 2, then two bonded atoms are both organic atoms.
        if surfbond == 1:
            # write surfacebond, but put it so that metal index comes first
            if IsSurfaceAtomNum(atom1.GetAtomicNum()):
                surfacebond.append([bonds.GetBeginAtomIdx(), bonds.GetEndAtomIdx()]) # [Surf atom, adsorbate atom]
            elif IsSurfaceAtomNum(atom2.GetAtomicNum()):
                surfacebond.append([bonds.GetEndAtomIdx(), bonds.GetBeginAtomIdx()])
    # make it into array
    surfacebond = np.array(surfacebond)
    # Distance constraing for surface bonds
    for i in range(0, surfacebond.shape[0]):
        if InitialGuessRun:
            ff.UFFAddDistanceConstraint(int(surfacebond[i,0]), int(surfacebond[i,1]), False, 2.0, 2.0, 5000.0)
        else:
            atom = mol.GetAtomWithIdx(int(surfacebond[i,1]))
            atom.UpdatePropertyCache()

            ff.UFFAddDistanceConstraint(int(surfacebond[i,0]), int(surfacebond[i,1]), False, 2.0,2.0, 4000.0)
            
    # constraint for atoms wanting to be perpendicular to the surface atom.
    
                    
    pos = CenterOfSurf - Zvector*10000000
    IdxSurfFixedMinusZ = ff.AddExtraPoint(pos[0],pos[1],pos[2],fixed=True)-1
    for i in range(0, surfacebond.shape[0]):
        # find corresponding fixed point
        if InitialGuessRun:
            ff.UFFAddAngleConstraint(int(surfacebond[i,0]), int(surfacebond[i,1]), IdxSurfFixedMinusZ, False, 0, 0, 10000.0)
        else:
            ff.UFFAddAngleConstraint(int(surfacebond[i,0]), int(surfacebond[i,1]), IdxSurfFixedMinusZ, False, 0, 0, ZStrain)
    
    # Add repulsive force between atoms and surface atom.
    # this is pseudo done by setting a point above surface, and apply distance constraint
    if AdsorbateSurfaceRepulsion:

        pos = CenterOfSurf + Zvector*10000000
        IdxSurfFixed = ff.AddExtraPoint(pos[0],pos[1],pos[2],fixed=True)-1
        # distance strain method
        for atom in mol.GetAtoms():
            if atom.HasProp('Adsorbed'):
                if InitialGuessRun:
                    ff.UFFAddDistanceConstraint(atom.GetIdx(), IdxSurfFixed, False, 0, 9999997.8, 1000.0)
                else:
                    ff.UFFAddDistanceConstraint(atom.GetIdx(), IdxSurfFixed, False, 0, 9999997.8, 200.0)

                
    # Add angle constraint for neighbors of adsorbed atoms (AdsorbedAtomNeighbor-AdsorbedAtom-Metal)
    for i in range(0, surfacebond.shape[0]):
        centeratom = mol.GetAtomWithIdx(int(surfacebond[i,1]))
        surfaceatomidx = int(surfacebond[i,0])
        centeratomidx = int(surfacebond[i,1])
        bondedorganicatom = list()
        # go through each bond record each bonded atom
        for NBRAtom in centeratom.GetNeighbors():
            if NBRAtom.GetProp('Type') == 'A':
                bondedorganicatom.append(NBRAtom.GetIdx())
        #following debugging code print out index of each atoms
        #print bondedorganicatom
        #print 'surface atom: {0:.0f}, center atom: {1:.0f}'.format(surfaceatom, centeratom)        
        
        ## (AdsorbedAtomNeighbor-AdsorbedAtom-Metal)
        for organicatomidx in bondedorganicatom:
#            print 'surf', mol.GetAtomWithIdx(surfaceatomidx).GetSymbol()
#            print 'cent', mol.GetAtomWithIdx(centeratomidx).GetSymbol()
#            print 'org', mol.GetAtomWithIdx(organicatomidx).GetSymbol()
            if mol.GetAtomWithIdx(centeratomidx).GetAtomicNum() == 6:
                if len(bondedorganicatom) == 3:
                    if mol.GetAtomWithIdx(organicatomidx).GetHybridization() == Chem.rdchem.HybridizationType.SP2 and\
                        mol.GetAtomWithIdx(organicatomidx).GetAtomicNum() == 6:
                        ff.UFFAddAngleConstraint(surfaceatomidx, centeratomidx, organicatomidx, False, 90, 90, 300.0)
                    else:
                        ff.UFFAddAngleConstraint(surfaceatomidx, centeratomidx, organicatomidx, False, 109.5, 109.5, 300.0)
                if len(bondedorganicatom) == 2:
                    ff.UFFAddAngleConstraint(surfaceatomidx, centeratomidx, organicatomidx, False, 145.0, 180.0, 150.0)
                if len(bondedorganicatom) == 1:
                    ff.UFFAddAngleConstraint(surfaceatomidx, centeratomidx, organicatomidx, False, 180.0, 180.0, 150.0)
            elif mol.GetAtomWithIdx(centeratomidx).GetAtomicNum() == 8:
                if len(bondedorganicatom) == 2:
                    ff.UFFAddAngleConstraint(surfaceatomidx, centeratomidx, organicatomidx, False, 109.5, 109.5, 150.0)
                if len(bondedorganicatom) == 1:
                    ff.UFFAddAngleConstraint(surfaceatomidx, centeratomidx, organicatomidx, False, 120.0, 120.0, 150.0)
        
        ## (AdsorbedAtomNeighbor-AdsorbedAtom-AdsorbedAtomNeighbor)
        for organicatomidx1 in bondedorganicatom:
            for organicatomidx2 in bondedorganicatom:
                if organicatomidx1 != organicatomidx2:
                    if InitialGuessRun:
                        ff.UFFAddAngleConstraint(organicatomidx1, centeratomidx, organicatomidx2, False, 120, 180.0, 1000.0)
                    else:
                        if len(bondedorganicatom) == 3:
                            ff.UFFAddAngleConstraint(organicatomidx1, centeratomidx, organicatomidx2, False, 109.5, 120.0, 200.0)
                        elif len(bondedorganicatom) == 2:
                            ff.UFFAddAngleConstraint(organicatomidx1, centeratomidx, organicatomidx2, False, 145, 180, 200.0)
                        elif len(bondedorganicatom) == 1:
                            ff.UFFAddAngleConstraint(organicatomidx1, centeratomidx, organicatomidx2, False, 180, 180, 200.0)
    for Atom in mol.GetAtoms():
        if Atom.GetProp('Type') == 'A' and IsAdsorbateAtomAdsorbed(Atom):
            nS = 0
            SurfIdx = list()
            for NBRAtom in Atom.GetNeighbors():
                if NBRAtom.GetProp('Type') == 'S':
                    nS += 1
                    SurfIdx.append(NBRAtom.GetIdx())
            if nS > 1:
                combs = combinations(SurfIdx,2)
                if Atom.GetTotalValence() + nS == 4:
                    for comb in combs:
                        ff.UFFAddAngleConstraint(comb[0], Atom.GetIdx(), comb[1], False, 109.5, 109.5, 4000.0)

    return ff

        
class Surface(object):
    
    def __init__(self,path, name=None,ZLattVecI = 2,SecondLayerAtom='He'):
        self.Surf = LoadNonPeriodicGraphByCovalentRadius(path,PBCContainingAdsorbateOnly=False)
        if name:
            self.name = name
        else:
            self.name = self.Surf.aseatoms.get_chemical_formula()
        self.ZLattVecI = ZLattVecI
        self.ns = self.Surf.RdkitMol.GetNumAtoms()
        self.SecondLayerAtom = Chem.Atom(SecondLayerAtom)
        # Find second layers' connectivity to first layer
        layer2xyz = self.Surf.aseatoms.get_scaled_positions()[self.Surf.LayerIdxs[1]]
        layer2xyz = np.concatenate(np.array(self.Surf.AddedPBCs)[:,None,:]+layer2xyz[None,:,:])
        layer2xyz = np.dot(layer2xyz,self.Surf.aseatoms.cell)
        # Add Bond
        xyzs = []
        refxyzs = self.Surf.aseatoms.get_scaled_positions(wrap=False)
        for i in range(len(self.Surf.RdKitAtomIndex2ASEAtomIndex)):
            l = literal_eval(self.Surf.RdKitAtomIndex2ASEAtomIndex[i])
            xyzs.append(refxyzs[l[0]]+l[1:])

        xyzs = np.dot(xyzs,self.Surf.aseatoms.cell)
        dist = cdist(layer2xyz,xyzs)
        mdist = np.min(dist)
        self.SecondLayerConnectivity = [[] for _ in range(layer2xyz.shape[0])]
        for i,j in zip(*np.where(np.isclose(dist,mdist,atol=0.2))):
            self.SecondLayerConnectivity[int(i)].append(int(j))


        # Remove those at the end
        lengths = Counter([len(i) for i in self.SecondLayerConnectivity])
        nbond = lengths.most_common(1)[0][0]
        self.SecondLayerConnectivity = [set(i) for i in self.SecondLayerConnectivity if len(i) == nbond]
    
    SurfAtom = Chem.Atom(0)
    
    def __repr__(self):
        return '<GraphLearning.io.Surface|'+self.name+'>'
    
    @classmethod
    def GetCanonicalSmiles(cls,mol):
        # Convert surface atom to *
        for i in reversed(range(mol.GetNumAtoms())):
            if mol.GetAtomWithIdx(i).HasProp('Type') and mol.GetAtomWithIdx(i).GetProp('Type') == 'S':
                mol.ReplaceAtom(i,cls.SurfAtom)
        
        # change atom set up
        for atom in mol.GetAtoms():
#            atom.SetIsotope(0)# Not sure what this does...
            atom.ClearProp('smilesSymbol')
            if atom.GetAtomicNum() != 0:
                atom.SetNoImplicit(True)
                atom.SetNumRadicalElectrons(1)
        # Change bond to single bond
        for bond in mol.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)# Put bracket around atoms
            
        return Chem.MolToSmiles(mol)
    
    def GetProjection(self,SMILESorMol, Quiet = True,ZStrain=150.0):
        """
        Output:
            output : -1 Minimisation failed, -2 No surface
        """
        # Initialize
        mol = _PretreatSMILESorMol(SMILESorMol)
        # Get list of Surface Atom Indices
        SurfIdxs = list()
        for atom in mol.GetAtoms():
            if atom.GetProp('Type') == 'S':
                SurfIdxs.append(atom.GetIdx())
                
        if len(SurfIdxs) == 0:
    #        print 'No Connectivity To Surface'
            return -2, None
        OriginalToSurf = dict() # Original Mol Idx -> New Mol Idx
        # Get Surface Graph
        if len(SurfIdxs) != 1:
            BondList = GetBondListFromAtomList(mol,SurfIdxs)
            SurfMol = Chem.RWMol(Chem.PathToSubmol(mol,BondList,atomMap = OriginalToSurf).__copy__())
        else:
            SurfMol = mol.__copy__()
            ## Non surface Atom
            for idx in reversed(range(0,SurfMol.GetNumAtoms())):
                atom = SurfMol.GetAtomWithIdx(idx)
                if atom.GetProp('Type') == 'A':
                    SurfMol.RemoveAtom(atom.GetIdx())
            OriginalToSurf[SurfIdxs[0]] = 0
        # Get mapping
        SurfToOriginal = dict()
        for OriginalIdx in OriginalToSurf:
            SurfToOriginal[OriginalToSurf[OriginalIdx]] = OriginalIdx
        
        # For searching the pattern on ASERdkit, we gotta use special atoms
        SurfMolForSearch = Chem.RWMol(SurfMol.__copy__())
        SA = rdqueries.HasStringPropWithValueQueryAtom('Type','S')
        SA.ExpandQuery(rdqueries.HasBoolPropWithValueQueryAtom('Occupied',False))        
        SA.SetProp('smilesSymbol','M')
        for idx in range(0,SurfMolForSearch.GetNumAtoms()):
            SurfMolForSearch.ReplaceAtom(idx,SA)

        # The adsorbate surface is projected to surface
        # Also find the ones that are closest to the center of the cell
        ProjectedSurfIdxsSetsTemp = self.Surf.RdkitMol.GetSubstructMatches(SurfMolForSearch)
        # remove projection that includes surface atoms at the edge
        ProjectedSurfIdxsSets = []
        for ProjectedSurfIdxsSet in ProjectedSurfIdxsSetsTemp:
            if not set(self.Surf.EdgeSurf) & set(ProjectedSurfIdxsSet):
                ProjectedSurfIdxsSets.append(ProjectedSurfIdxsSet)

        Dist2Centers = defaultdict(list) # Distance from center
        ProjectedSurfIdxsSetsCategorized = defaultdict(list)
        Mol = {}
        Center = np.average(self.Surf.aseatoms.cell[0:2,:],axis=0)[:2]
        scaledxyz = self.Surf.aseatoms.get_scaled_positions()
        for ProjectedSurfIdxs in ProjectedSurfIdxsSets:
            SurfCent = list()
            for idx in ProjectedSurfIdxs:
                ProjectedASEIdx = literal_eval(self.Surf.RdKitAtomIndex2ASEAtomIndex[idx])
                SurfCent.append(np.dot(scaledxyz[ProjectedASEIdx[0]] + ProjectedASEIdx[1:],self.Surf.aseatoms.cell)[:2])
            SurfCent = np.average(SurfCent,axis=0)
            dist = np.linalg.norm(SurfCent - Center)
            tmol = mol.__copy__()
            for SLC in self.SecondLayerConnectivity:
                if SLC and SLC.issubset(set(ProjectedSurfIdxs)):
                    i = tmol.AddAtom(self.SecondLayerAtom)
                    for j in SLC:
                        tmol.AddBond(i,SurfToOriginal[ProjectedSurfIdxs.index(j)])
            s = Chem.MolToSmiles(tmol)
            Dist2Centers[s].append(dist)
            ProjectedSurfIdxsSetsCategorized[s].append(ProjectedSurfIdxs)
            if s not in Mol:
                Mol[s] = tmol

        # Select closest to the center for each smiles
        SelectedProjectedSurfIdxsSets = dict()
        for s in Dist2Centers:
            i = np.argmin(Dist2Centers[s])
            SelectedProjectedSurfIdxsSets[s] = ProjectedSurfIdxsSetsCategorized[s][i]
                
        # The adsorbate surface is projected to surface
        atoms = []
        for s in SelectedProjectedSurfIdxsSets:
            Tmol = mol.__copy__()
            # Get Coordinates
            ProjectedASEIdxs = list()
            for idx in SelectedProjectedSurfIdxsSets[s]:
                ProjectedASEIdxs.append(literal_eval(self.Surf.RdKitAtomIndex2ASEAtomIndex[idx]))
            
            # Record Mol To ASE 
            MolToASE = dict()
            for i in range(0,len(ProjectedASEIdxs)):
                MolToASE[SurfToOriginal[i]] = ProjectedASEIdxs[i][0]
            
            # Set Surface Atom Position
            ## initialize adsorbate atom positions
            SurfCoordMap = dict()
            CenterSurf = list()
            for i in range(0,len(ProjectedASEIdxs)):
                ProjectedASEIdx = ProjectedASEIdxs[i]
                pos = self.Surf.aseatoms[ProjectedASEIdx[0]].position + np.dot(ProjectedASEIdx[1:],self.Surf.aseatoms.cell)
                CenterSurf.append(pos)
                Coord = Geometry.Point3D(pos[0],pos[1],pos[2])
                SurfCoordMap[SurfToOriginal[i]] = Coord
            CenterSurf = np.average(CenterSurf,axis=0) 

            conf = Chem.Conformer()
            for atom in Tmol.GetAtoms():
                if atom.GetProp('Type') == 'A':
                    conf.SetAtomPosition(atom.GetIdx(),(CenterSurf[0]+np.random.rand()*10-5,CenterSurf[1]+np.random.rand()*10-5,CenterSurf[2]+20))
            for idx in SurfCoordMap:
                conf.SetAtomPosition(idx,SurfCoordMap[idx])
            Tmol.AddConformer(conf)
            ## Preliminary treatment before optimization
            # More options available here:
            # http://www.rdkit.org/Python_Docs/rdkit.Chem.rdDistGeom.EmbedParameters-class.html
            # More Discussions
            # https://sourceforge.net/p/rdkit/mailman/message/32082674/
            #EmbedTmolecule(class RDKit::ROTmol {lvalue} Tmol, unsigned int maxAttempts=0, 
            #              int randomSeed=-1, bool clearConfs=True, bool useRandomCoords=False, 
            #              double boxSizeMult=2.0, bool randNegEig=True, unsigned int numZeroFail=1, 
            #              class boost::python::dict {lvalue} coordMap={}, double forceTol=0.001, 
            #              bool ignoreSmoothingFailures=False, bool enforceChirality=True, 
            #              bool useExpTorsionAnglePrefs=False, bool useBasicKnowledge=False, 
            #              bool printExpTorsionAngles=False)
            
                
            ff = _SetUpForceField(Tmol,cell = self.Surf.aseatoms.cell, ZLattVecI = self.ZLattVecI,InitialGuessRun=True,ZStrain=ZStrain)
            
            # Optimize Tmolecule
            ff.Initialize()
            output = ff.Minimize();
    #        output = ff.Minimize(maxIts=10000000, forceTol=1e-10, energyTol=1e-010);
    #        
            ff = _SetUpForceField(Tmol,cell = self.Surf.aseatoms.cell, ZLattVecI = self.ZLattVecI,InitialGuessRun=False,ZStrain=ZStrain)
            # Optimize Tmolecule
            ff.Initialize()
            output = ff.Minimize(maxIts=10000000, forceTol=1e-12, energyTol=1e-012);
        
            # report minimization result
            if not Quiet:
                if output == -1:
                    print('Minimization did not converge ('+str(output)+')')
                else:
                    print('Minimization Successful ('+str(output)+')')
            
            ## Append Position
            aseatoms = self.Surf.aseatoms.copy()
            for i in range(0, Tmol.GetNumAtoms()):
                atom = Tmol.GetAtomWithIdx(i)
                if atom.GetProp('Type') == 'A':
                    atom = ASEAtom(atom.GetSymbol(),Tmol.GetConformer().GetAtomPosition(i))
                    aseatoms.append(atom)
                    MolToASE[i] = len(aseatoms)-1
                elif atom.GetProp('Type') == 'S':
                    aseatoms[MolToASE[i]].position = Tmol.GetConformer().GetAtomPosition(i)
            atoms.append((aseatoms,self.GetCanonicalSmiles(Mol[s]),output))
        return atoms
    
def LoadNonPeriodicGraphByCovalentRadius(CoordinateFPathOrASEAtoms, \
    rfacup = 1.35,rfacdown = 0.6, z_vector = 2, PBCContainingAdsorbateOnly=False, CutOffTol=0.3, SetMetalAtomNumToZero = False):
    
    def MakeAdsorbateAtom(AtomicNumber):
        if isinstance(AtomicNumber,(np.int64,np.int32)):
            AtomicNumber = int(AtomicNumber)
        atom = Chem.Atom(AtomicNumber)
        atom.SetNoImplicit(True) # this allows molecule to have radical atoms
        atom.SetProp('Type','A')
        atom.SetBoolProp('Adsorbed',False)
        return atom
    def MakeSurfAtom(AtomicNumber):
        if isinstance(AtomicNumber,(np.int64,np.int32)):
            AtomicNumber = int(AtomicNumber)
        
        if SetMetalAtomNumToZero:
            atom = Chem.Atom(0)
        else:
            atom = Chem.Atom(AtomicNumber)
        atom.SetProp('Type','S')
        atom.SetBoolProp('Occupied',False)
        return atom
    """ 
    This function reads file using ASE read, and construts molecular graph
    in rdkit object, Mol. Then, the cell is enlarged to include neighbor cells,
    and the adsorbates are isolated. Useful for getting graph descriptors
    
    
    Input List
    CoordinateFPathOrASEAtoms:    path to ASE readable coordinate file or ASE atoms object
    rfacup:             Upper percentage limit for determining connectivity.
    rfacdown:           Lower percentage limit for determining connectivity.
    z_vector:           index of cell basis vector that is orthogonal to surface.
    
    Output List
    adsorbate class
    """
    # load POSCAR
    if isinstance(CoordinateFPathOrASEAtoms,str) and os.path.exists(CoordinateFPathOrASEAtoms):
        AseAtoms = read(CoordinateFPathOrASEAtoms)
    elif isinstance(CoordinateFPathOrASEAtoms,ase_Atoms):
        AseAtoms = CoordinateFPathOrASEAtoms
    else:
        raise ValueError(CoordinateFPathOrASEAtoms, 'Unrecognized input format, or nonexisting file path')
    
    # initialize
    ASEAtomIndex2RdKitAtomIndex = dict()
    RdKitAtomIndex2ASEAtomIndex = dict()
    

    # (p)eriodic (b)oundary (c)ondition(s)
    PBCs = [[0,0,0]]
    if AseAtoms.pbc[0]:
        temp = np.add(PBCs,[1,0,0])
        temp = np.concatenate((temp,np.add(PBCs,[-1,0,0])))
        PBCs = np.concatenate((PBCs,temp))
    if AseAtoms.pbc[1]:
        temp = np.add(PBCs,[0,1,0])
        temp = np.concatenate((temp,np.add(PBCs,[0,-1,0])))
        PBCs = np.concatenate((PBCs,temp))
    if AseAtoms.pbc[2]:
        temp = np.add(PBCs,[0,0,1])
        temp = np.concatenate((temp,np.add(PBCs,[0,0,-1])))
        PBCs = np.concatenate((PBCs,temp))
    if not AseAtoms.pbc[0] and not AseAtoms.pbc[1] and not AseAtoms.pbc[2]:
        AseAtoms.cell = np.diag((1,1,1))
    PBCs = list(PBCs)
    for i in range(0,len(PBCs)):
        PBCs[i] = list(PBCs[i])
    # Get organic atoms from the DFT calculations (their index and atomic number)
    oai = list() #organic atom index in the atoms object
    ASEIdxToCheck = list() 
    for i in range(0,AseAtoms.__len__()):
        if IsAdsorbateAtomNum(int(AseAtoms[i].number)):
            oai.append(i)
            ASEIdxToCheck.append(i)
    # construct mol object
    RdkitMol = Chem.Mol()
    RdkitMol = Chem.RWMol(RdkitMol)
    #%%  Determine connectivity and each atoms' periodic condition.
    Adsorbates = list()
    while ASEIdxToCheck:
        InitialASEIdx = ASEIdxToCheck.pop()
        # Pick and atom find all connected atoms to make an adsorbate
        MolASEIdxToCheck = list()
        MolASEIdxToCheck.append(InitialASEIdx)
        # List of Picked atoms and PBC
        MolASEIdxAndPBC = dict()
        MolASEIdxAndPBC[InitialASEIdx] = [0,0,0]
        # Add Atom
        RdkitIdx = RdkitMol.AddAtom(MakeAdsorbateAtom(AseAtoms[InitialASEIdx].number))
        ASEAtomIndex2RdKitAtomIndex[InitialASEIdx] = RdkitIdx
        RdKitAtomIndex2ASEAtomIndex[RdkitIdx] = InitialASEIdx
        # recursively find all atoms in the adsorbate containing this atom
        while MolASEIdxToCheck:
            ASEIdxBeingChecked = MolASEIdxToCheck.pop()
            # Determine Neighbors
            ## potential atoms
            NeighborIdx = [RdKitAtomIndex2ASEAtomIndex[atom.GetIdx()] for atom in RdkitMol.GetAtomWithIdx(ASEAtomIndex2RdKitAtomIndex[ASEIdxBeingChecked]).GetNeighbors()]
            ASEidxlist = [oai[i] for i in range(0,len(oai)) if oai[i] not in NeighborIdx]
            for j in ASEidxlist:
                # if this atom has already been accounted
                if j in MolASEIdxAndPBC:
                    Bool,_,_ = _DetermineConnectivity(AseAtoms,ASEIdxBeingChecked,j,[MolASEIdxAndPBC[j]],1.15,rfacdown,PBCi=MolASEIdxAndPBC[ASEIdxBeingChecked])
                    if Bool:
                        RdkitMol.AddBond(ASEAtomIndex2RdKitAtomIndex[ASEIdxBeingChecked],ASEAtomIndex2RdKitAtomIndex[j],order=Chem.rdchem.BondType.SINGLE)
                else:
                    Bool,PBC,_ = _DetermineConnectivity(AseAtoms,ASEIdxBeingChecked,j,PBCs,1.15,rfacdown,PBCi=MolASEIdxAndPBC[ASEIdxBeingChecked])
                    if Bool:
                        MolASEIdxAndPBC[j] = list(PBC)
                        MolASEIdxToCheck.append(j)
                        # Add Atom
                        RdkitIdx = RdkitMol.AddAtom(MakeAdsorbateAtom(AseAtoms[j].number))
                        ASEAtomIndex2RdKitAtomIndex[j] = RdkitIdx
                        RdKitAtomIndex2ASEAtomIndex[RdkitIdx] = j
                        RdkitMol.AddBond(ASEAtomIndex2RdKitAtomIndex[ASEIdxBeingChecked],ASEAtomIndex2RdKitAtomIndex[j],order=Chem.rdchem.BondType.SINGLE)
        # Add made molecule to the adsorbate list
        Adsorbates.append(MolASEIdxAndPBC)
        ASEIdxToCheck = [Idx for Idx in ASEIdxToCheck if Idx not in MolASEIdxAndPBC]
        
    # For each adsorbate, adjust its PBC location to where most adsorbate atom is found
    PBCWithAdsorbateList = list()
    AllMolASEIdxAndPBC = dict()
    for MolASEIdxAndPBC in Adsorbates:
        PBCList = list()
        Count = list()
        for idx in MolASEIdxAndPBC:
            if MolASEIdxAndPBC[idx] not in PBCWithAdsorbateList:
                PBCWithAdsorbateList.append(MolASEIdxAndPBC[idx])
            if MolASEIdxAndPBC[idx] not in PBCList:
                PBCList.append(MolASEIdxAndPBC[idx])
                Count.append(1)
            else:
                i = PBCList.index(MolASEIdxAndPBC[idx])
                Count[i] +=1
        # Adjust PBC of the adsorbate
        PBC = PBCList[np.argmax(Count)]
        for idx in MolASEIdxAndPBC:
            AllMolASEIdxAndPBC[idx] = np.subtract(MolASEIdxAndPBC[idx],PBC)
    # %% Get Surface. 
    ## if none given for surface layer z coordinate, average the top layer atomic coordinate
    _, SurfaceAtomIndex,LayerIdxs = _DetermineSurfaceLayerZ(AseAtoms, ZVecIndex = z_vector)

    ## Construct Surface in each PBC
    positions = dict()
    SurfMol = Chem.RWMol(Chem.Mol())
    for Idx in SurfaceAtomIndex:
        RdkitIdx = SurfMol.AddAtom(MakeSurfAtom(AseAtoms[Idx].number))
        ASEAtomIndex2RdKitAtomIndex[str([Idx,0,0,0])] = RdkitIdx+RdkitMol.GetNumAtoms()
        RdKitAtomIndex2ASEAtomIndex[RdkitIdx+RdkitMol.GetNumAtoms()] = str([Idx,0,0,0])
        positions[RdkitIdx+RdkitMol.GetNumAtoms()] = AseAtoms[Idx].position

    ## Make Bonds and find bond to other 
    BondsToOtherPBC = list()
    AddedPBC = list()
#    print(SurfaceAtomIndex) # TODO:
#    print(AseAtoms[22].position,AseAtoms[31].position,np.linalg.norm(AseAtoms[22].position-AseAtoms[31].position))# TODO:
    for i in range(0,len(SurfaceAtomIndex)):
        for j in range(i+1,len(SurfaceAtomIndex)):
            
            Bool,PBC,_ = _DetermineConnectivity(AseAtoms,SurfaceAtomIndex[i],SurfaceAtomIndex[j],PBCs,rfacup,rfacdown)
#            if SurfaceAtomIndex[i] == 22 and SurfaceAtomIndex[j] ==31:# TODO:
#                print(Bool,PBC)# TODO:
                   
            if Bool:
                if PBC == [0,0,0]:
                    # Add Atom
                    SurfMol.AddBond(i,j,order=Chem.rdchem.BondType.ZERO)
                else:
                    if PBC not in AddedPBC:
                        AddedPBC.append(PBC)
                    NPBC = [-PBC[0],-PBC[1],-PBC[2]]
                    if NPBC not in AddedPBC:
                        AddedPBC.append(NPBC)
                    BondsToOtherPBC.append([SurfaceAtomIndex[i],0,0,0,SurfaceAtomIndex[j]]+PBC)
    # BondToOtherPBC: [idx1,pbc,idx2,pbc]
#        print ASEAtomIndex2RdKitAtomIndex #DEBUG

    ## assign radicals
    Chem.AssignRadicals(RdkitMol)
    
    ## set smilesSymbol
    for atom in RdkitMol.GetAtoms():
        if atom.GetSymbol() in ['C','O'] and atom.GetNumRadicalElectrons() == 0:
            atom.SetProp("smilesSymbol",'[' + atom.GetSymbol() + str(atom.GetNumRadicalElectrons())+ '0]')
        elif atom.GetNumRadicalElectrons() > 0:
            atom.SetProp("smilesSymbol",atom.GetSymbol() + str(atom.GetNumRadicalElectrons()))
        
    #%%  Find surface binding adsorbate atom. This is done by finding all the radical atoms
    rai_rdkit = list() # radical atom index for rdkit mol
    rai_ase = list() # radical atom index for rdkit ase atoms object
    for atom in RdkitMol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            rai_rdkit.append(atom.GetIdx())
            rai_ase.append(RdKitAtomIndex2ASEAtomIndex[atom.GetIdx()])
            
    
    # %% Surface connectivity
    SurfBondDict = dict() #{AtomIdx:BondDistDict}
    for i in rai_ase:
        PBCi = AllMolASEIdxAndPBC[i]
        BondDistPBCDict = dict() #{SurfAtomIdx:(Distance,PBCj)}
        for j in SurfaceAtomIndex:
            Bool,PBCj,d = _DetermineConnectivity(AseAtoms,i,j,PBCs,rfacup,rfacdown,PBCi = PBCi)
            if Bool:
                BondDistPBCDict[j] = (d,PBCj)
                if PBCj not in PBCWithAdsorbateList:
                    PBCWithAdsorbateList.append(PBCj)
        if len(BondDistPBCDict) != 0:
            SurfBondDict[i] = BondDistPBCDict
    

    # %% Apend surface
    if not PBCContainingAdsorbateOnly:
        # This just add PBCs that the surface spans on
        ## Find absolute number PBCs that surface spans on.

        PBCMax = np.max(np.abs(AddedPBC),axis=0)
        ## Other PBC to other PBC bonds
        PBCToAdd = [[0,0,0]]
        if PBCMax[0]:
            temp = np.add(PBCToAdd,[1,0,0])
            temp = np.concatenate((temp,np.add(PBCToAdd,[-1,0,0])))
            PBCToAdd = np.concatenate((PBCToAdd,temp))
        if PBCMax[1]:
            temp = np.add(PBCToAdd,[0,1,0])
            temp = np.concatenate((temp,np.add(PBCToAdd,[0,-1,0])))
            PBCToAdd = np.concatenate((PBCToAdd,temp))
        if PBCMax[2]:
            temp = np.add(PBCToAdd,[0,0,1])
            temp = np.concatenate((temp,np.add(PBCToAdd,[0,0,-1])))
            PBCToAdd = np.concatenate((PBCToAdd,temp))
        PBCToAdd = PBCToAdd.tolist()
        
    else:
        # Add all PBC with adsorbates on it
        ## here if we have e.g., [0,0,0] and [-1,1,0], the following for loop
        ## enumerates [0,0,0],[-1,1,0],[-1,0,0],[0,1,0]
        PBCToAdd = copy.deepcopy(PBCWithAdsorbateList)
        for PBC in PBCToAdd:
            nonzeros = list()
            for i in range(0,3):
                if PBC[i] != 0:
                    nonzeros.append(i)
            combs = [p for p in itertools.product([0,1], repeat=len(nonzeros))]
            for comb in combs:
                TempPBC = [0,0,0]
                for i in range(0,len(nonzeros)):
                    if comb[i] == 1:
                        TempPBC[nonzeros[i]] = PBC[nonzeros[i]]
                if TempPBC not in PBCToAdd:
                    PBCToAdd.append(TempPBC)
    ## Make Bonds
    NewBondsToOtherPBC = list()
    for PBC in PBCToAdd:
        for j in range(0,len(BondsToOtherPBC)):
            pbc1 = list(np.add(BondsToOtherPBC[j][1:4],PBC))
            pbc2 = list(np.add(BondsToOtherPBC[j][5:8],PBC))
            if np.all(np.abs(pbc1)<2) and np.all(np.abs(pbc2)<2) and\
                pbc1 in PBCToAdd and pbc2 in PBCToAdd:
                NewBondsToOtherPBC.append([BondsToOtherPBC[j][0]]+list(pbc1)+[BondsToOtherPBC[j][4]]+list(pbc2))
                
    ## Add 0,0,0 Surface
    RdkitMol = Chem.RWMol(Chem.CombineMols(RdkitMol,SurfMol))
    ## Add Other Surface
    for PBC in PBCToAdd:
        if PBC != [0,0,0]:
            for k in range(0,len(SurfaceAtomIndex)):
                ASEAtomIndex2RdKitAtomIndex[str([SurfaceAtomIndex[k]]+PBC)] = k+RdkitMol.GetNumAtoms()
                RdKitAtomIndex2ASEAtomIndex[k+RdkitMol.GetNumAtoms()] = str([SurfaceAtomIndex[k]]+PBC)
                positions[k+RdkitMol.GetNumAtoms()] = AseAtoms[SurfaceAtomIndex[k]].position  + np.dot(PBC,AseAtoms.cell)
            RdkitMol = Chem.RWMol(Chem.CombineMols(RdkitMol,SurfMol))

    ## Make bonds between surfaces
    for bond in NewBondsToOtherPBC:
        RdkitMol.AddBond(ASEAtomIndex2RdKitAtomIndex[str(bond[0:4])],ASEAtomIndex2RdKitAtomIndex[str(bond[4:8])],order=Chem.rdchem.BondType.ZERO)
    
    #%% Apply cut off
    for i in SurfBondDict: # i is idx of surface bonding adsorbate atom.
        # Determine Minimum Distance
        MinD = 1000 # Fake Large number.
        for j in SurfBondDict[i]: # j is idx of binding surface atom.
            if SurfBondDict[i][j][0] < MinD:
                MinD = SurfBondDict[i][j][0]
        # Apply cut off
        for j in SurfBondDict[i]: # j is idx of binding surface atom.
            if SurfBondDict[i][j][0] < MinD + CutOffTol:

                RdkitMol.AddBond(ASEAtomIndex2RdKitAtomIndex[i],ASEAtomIndex2RdKitAtomIndex[str([j]+list(SurfBondDict[i][j][1]))],order=Chem.rdchem.BondType.ZERO)
                RdkitMol.GetAtomWithIdx(ASEAtomIndex2RdKitAtomIndex[str([j]+list(SurfBondDict[i][j][1]))]).SetBoolProp('Occupied',True)
                RdkitMol.GetAtomWithIdx(ASEAtomIndex2RdKitAtomIndex[i]).SetBoolProp('Adsorbed',True)
    #%% Find surface atoms at the edges
    nsurf = defaultdict(int)
    for atom in RdkitMol.GetAtoms():
        if atom.GetProp('Type') == 'S':
            for neighbor_atom in atom.GetNeighbors():
                if neighbor_atom.GetProp('Type') == 'S':
                    nsurf[atom.GetIdx()] += 1
                    
    nbond = Counter(nsurf.values()).most_common(1)[0][0]
    edgesurf = []
    for idx in nsurf:
        if nsurf[idx] != nbond:
            edgesurf.append(idx)
    # %%assign binding site.
    for i in rai_rdkit:
        a = RdkitMol.GetAtomWithIdx(i)
        nsurf = 0
        for neighbor_atom in a.GetNeighbors():
            if neighbor_atom.GetProp('Type') == 'S':
                nsurf += 1
        a.SetProp("smilesSymbol",a.GetProp("smilesSymbol") + str(nsurf))
        
    adsorbate = AdsorbateDatum(AseAtoms,RdkitMol, \
             ASEAtomIndex2RdKitAtomIndex, RdKitAtomIndex2ASEAtomIndex)
    adsorbate.LayerIdxs = LayerIdxs
    adsorbate.AddedPBCs = PBCToAdd
    adsorbate.EdgeSurf = edgesurf
    return adsorbate



    
def _DetermineSurfaceLayerZ(aseatoms, ZVecIndex = 2, ztol = 1.65):
    
    """
    Find top layer surface atom z coordinates by averaging
    atoms within ztol (angstrom) of the top most atoms are selected for averaging
    
    Input List
    aseatoms:           ASE atoms containing adsorbate/surface system.
    ZVecIndex:          index of cell basis vector that is orthogonal to surface.
    ztol:               Atoms within ztol(angstrom) of the top most atoms are selected as 
                        surface atoms.
    Output List
    SurfaceLayerZ:      z coordinate of surface layer.
    SurfaceAtomIndex:   Index of surface atoms.
    
    Ideas: This may be smartly done by first finding surf atom connected to adsorbates
    """
    
    assert isinstance(aseatoms,ASEAtoms)
    # get highest surface atom coordinate
    zmax = 0
    zs = aseatoms.get_scaled_positions()[:,ZVecIndex]
    zs = np.round(zs,decimals = 5)
    zs[zs==1.0] = 0.0
    for i in range(0,len(aseatoms)):
        if IsSurfaceAtomNum(aseatoms[i].number) and zmax < zs[i]:
            zmax = zs[i]
        
    # determine z coordinate. average out top layer
    ztol = ztol/np.linalg.norm(aseatoms.cell[2,:])
    SurfaceAtomIndex = list()
    SurfZs = list()

    for i in range(0,len(aseatoms)):
        if IsSurfaceAtomNum(aseatoms[i].number) and zmax - ztol < zs[i]:
            SurfZs.append(zs[i])
            SurfaceAtomIndex.append(i)
    SurfaceLayerZ = np.array(SurfZs).mean()
    OrderedIdx = np.argsort(zs)[::-1]
    nl = 0
    LayerIdxs = []
    while (nl+1)*len(SurfZs) <=len(zs):
        LayerIdxs.append(OrderedIdx[len(SurfZs)*nl:len(SurfZs)*(nl+1)].tolist())
        nl +=1
    return SurfaceLayerZ, SurfaceAtomIndex, LayerIdxs

    

def _DetermineConnectivity(AseAtoms,i,j,PBCs,rfacup,rfacdown,PBCi = [0,0,0]):
    """
    Determine connectivity between atom i and j. See equation (1) in the 
    manuscript.
    
    Input List
    ASEAtoms:           ASE atoms containing adsorbate/surface system
    PBCs:               Periodic Boundary Conditions. e.g., (1,0,0) means 
                        cell repeats in first basis vector but not others.
    rfacup:             upper tolerance factor
    rfacdown:           lower tolerance factor
    PBCi:               PBC of atom i
    
    Output List
    Bool:               True if connected, false if not.
    PBC:                What PBC it's connected to
    """
    xyz1 = AseAtoms[i].position + np.dot(PBCi,AseAtoms.cell)

    # compute distances to each periodic cell
    d = np.linalg.norm(np.dot(PBCs,AseAtoms.cell) + AseAtoms[j].position - xyz1, axis=1)
    idx = np.argmin(d)
    d = d[idx]
    
    i_d = GetCovalentRadius(AseAtoms[i].number) + GetCovalentRadius(AseAtoms[j].number) # ideal distance
    if d <= i_d*rfacup and d >= i_d*rfacdown:
        return True, PBCs[idx], d    
    else:
        return False, [0,0,0], 0
    

class AdsorbateDatum(object):
    """
    This is an object contains aseatoms and the extracted graph
    
    Class Attributes
    aseatoms:                   ASE Atoms object.
    RdkitMol:                   Rdkit Mol object.
    ASEAtomIndex2RdKitAtomIndex: Index mapping from ASE atoms to Rdkit Mol
    RdKitAtomIndex2ASEAtomIndex: Index mapping from Rdkit Mol to ASE Atoms.
    """
    
    def __init__(self,aseatoms,RdkitMol, ASEAtomIndex2RdKitAtomIndex, \
        RdKitAtomIndex2ASEAtomIndex):
        
        assert isinstance(aseatoms,ASEAtoms)
        assert isinstance(RdkitMol,Chem.Mol)
        assert isinstance(ASEAtomIndex2RdKitAtomIndex,dict)
        assert isinstance(RdKitAtomIndex2ASEAtomIndex,dict)
        self.aseatoms = aseatoms
        self.RdkitMol = RdkitMol
        self.ASEAtomIndex2RdKitAtomIndex = ASEAtomIndex2RdKitAtomIndex
        self.RdKitAtomIndex2ASEAtomIndex = RdKitAtomIndex2ASEAtomIndex
        
    def GetLatticeAppendedASEAtom(self, Lattice):
        """
        This one returns ase atoms that can be turned into XSD
        """
        # remove surface atoms
        atoms = self.AseAtoms.copy()
        for i in range(len(atoms)-1,-1,-1):
            if IsSurfaceAtomNum(atoms[i].number):
                #TODO: assumes 3rd vector is z-axis
                del atoms[i]
        
        # append surface atom
        for site in Lattice._Sites:
            pos = np.append(site._Coordinate[0:2],self._SurfaceLayerZ)
#            pos = np.append(site._Coordinate[0:2],0)
            pos = np.dot(atoms.get_cell().transpose(),pos.transpose()).transpose()
            if site._SiteType == 0:
                atoms.append(ase_Atom('Pt', pos))
            elif site._SiteType == 1:
                atoms.append(ase_Atom('B', pos))
            elif site._SiteType == 2:
                atoms.append(ase_Atom('F', pos))
        return atoms

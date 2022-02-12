from rdkit import Chem
import numpy as np
from rdkit.Chem import rdqueries
from ase.data import atomic_numbers
from collections import defaultdict

SurfaceElements = ('Ag','Au','Co','Cu','Fe','Ir','Ni','Pd','Pt','Re','Rh','Ru')
SurfaceAtomicNumbers = tuple([0]+[atomic_numbers[s] for s in SurfaceElements])
# Elements of adsorbate atoms
AdsorbateElements = ('H','C','O','N')
AdsorbateAtomicNumbers = tuple([atomic_numbers[s] for s in AdsorbateElements])

Valence = {6:4,8:2}
def GetGraphDescriptors(smiles,maxatom=3):
    # get mol
    mol = Chem.MolFromSmiles(smiles,sanitize=False)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in [1,6,8]:
            atom.SetNoImplicit(True)
            atom.SetNumRadicalElectrons(1)
    mol = Chem.RWMol(mol)
    for i in reversed(range(mol.GetNumAtoms())):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() == 2:
            mol.RemoveAtom(i)
            
    # atom descriptors
    descriptors = []
    for atom in mol.GetAtoms():
        an = atom.GetAtomicNum()
        if an in [6,8]:
            OrganicDegree = 0
            SurfaceDegree = 0
            for nn in atom.GetNeighbors(): # calculate degree
                nnan = nn.GetAtomicNum()
                if nnan in [1,6,8]:
                    OrganicDegree +=1
                elif nnan == 0:
                    SurfaceDegree +=1
            valency = Valence[an] - OrganicDegree
            descriptor = '_'.join([str(an),str(valency),str(SurfaceDegree)])
            descriptors.append('['+descriptor+']')
    descriptors = [descriptors]
    
    # Extract just the organic atoms
    omol = Chem.RWMol(mol.__copy__())
    omolidx = []
    for i in reversed(range(mol.GetNumAtoms())):
        atom = omol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() not in [6,8]:
            omol.RemoveAtom(i)
        else:
            omolidx.append(i)
    omol2mol = {i:v for i,v in enumerate(reversed(omolidx))}
    
    # find organic molecule based subgraphs and map atom index back to mol atom index
    subgraph_atom_idxss = []
    for bidxss in Chem.FindAllSubgraphsOfLengthMToN(omol,1,maxatom-1):
        subgraph_atom_idxs = []
        for bidxs in bidxss:
            aidx = []
            for bidx in bidxs:
                bond = omol.GetBondWithIdx(bidx)
                aidx.append(bond.GetBeginAtomIdx())
                aidx.append(bond.GetEndAtomIdx())
            aidx = list(set(aidx))
            aidx = [omol2mol[i] for i in aidx]
            subgraph_atom_idxs.append(aidx)
        subgraph_atom_idxss.append(subgraph_atom_idxs)
    del omol
    # find subgraphs
    ## surface idx
    surf_idx = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            surf_idx.append(atom.GetIdx())
    for subgraph_atom_idxs in subgraph_atom_idxss:
        subgraph_descriptors_of_size_i = []
        for organic_atoms_to_preserve in subgraph_atom_idxs:
            # extract atoms of the bond and those connected to those atoms
            organic_atoms_to_change = []
            for i in organic_atoms_to_preserve:
                atom = mol.GetAtomWithIdx(i)
                for natom in atom.GetNeighbors():
                    if natom.GetAtomicNum() in [1,6,8]:
                        organic_atoms_to_change.append(natom.GetIdx())
            organic_atoms_to_preserve = set(organic_atoms_to_preserve)
            organic_atoms_to_change = set(organic_atoms_to_change)
            organic_atoms_to_change -= organic_atoms_to_preserve
            # extract subgraph
            submol = Chem.RWMol(mol.__copy__())
            for i in organic_atoms_to_change:
                R = Chem.Atom(1)
                submol.ReplaceAtom(i,R)
            for i in reversed(range(submol.GetNumAtoms())):
                if i not in organic_atoms_to_preserve and i not in organic_atoms_to_change \
                    and i not in surf_idx:
                    submol.RemoveAtom(i)
            submol = RemoveLatticeAmbiguity(submol)
            subgraph_descriptors_of_size_i.append(Chem.MolToSmiles(submol))
            del submol
        descriptors.append(subgraph_descriptors_of_size_i)
    return descriptors

def GetGasMolFromSmiles(smiles):
    mol = Chem.MolFromSmiles(smiles,sanitize=False)
    mol = Chem.RWMol(mol)
    for i in reversed(range(mol.GetNumAtoms())):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() in [1,6,8]:
            atom.SetNoImplicit(True)
            atom.SetNumRadicalElectrons(1)
        else:
            mol.RemoveAtom(i)
    return Chem.MolToSmiles(mol)

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


def GetBondListFromAtomList(Mol, AtomList):
    """
    Given a mol object and Atom Index list, Bond between atoms in atom list are
    printed. Typically used to use rdkit.Chem.PathToSubmol to extract subgraph.
    PathToSubmol is the most efficient subgraph extraction. See
    "def RemoveUnoccupiedSurfaceAtom" below for an example this.
    
    Input:
        Mol - Chem.Mol or RWMol Object.
        AtomList - Indexes of atom.
    Output:
        List of bond idx.
    """
    BondList = set()
    for idx in AtomList:
        atom = Mol.GetAtomWithIdx(idx)
        for bond in atom.GetBonds():
            if bond.GetOtherAtomIdx(atom.GetIdx()) in AtomList:
                BondList.add(bond.GetIdx())
    return list(BondList)



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


def FindNeighbor(xyz,mat,round_decimal,desired_distance):
    mat = np.subtract(mat,xyz)
    ds = np.linalg.norm(mat,axis=1)
    ds = np.around(ds,decimals=round_decimal)
    desired_distance = np.around(desired_distance,decimals=round_decimal)
    return np.where(np.equal(ds,desired_distance))[0] # because it gives tuple of tuple
                
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

def RemoveHe(smiles):
    mol = Chem.MolFromSmiles(smiles,sanitize=False)
    mol = Chem.RWMol(mol)
    for i in reversed(range(mol.GetNumAtoms())):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() in [1,6,8]:
            atom.SetNoImplicit(True)
            atom.SetNumRadicalElectrons(1)
        elif atom.GetAtomicNum() ==2 :
            mol.RemoveAtom(i)
    return Chem.MolToSmiles(mol)

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
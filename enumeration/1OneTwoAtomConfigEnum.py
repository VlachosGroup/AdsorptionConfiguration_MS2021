import json
from util import SurfHelper

# Set up reactant
## Set up Surface
c = SurfHelper(10)


atom1 = ['[C]*','[C]1**1','[C]12*3*1*23']
atom1_cannonical = [c.GetCanonicalSmiles(s) for s in atom1] # load and remake smiles using rdkit
json.dump(atom1_cannonical,open('./Output/1Skeleton1.json','w'))

atom2 = ['[C]1[C]**1',
    '[C]1[C]*1',
    '[C]1[C]2*3*2*13',
    '[C]12[C]*1*2',
    '[C]1[C]23*45*2*34*15',
    '[C]1[C]23*4*2*134',
    '[C]12[C]3*45*3*14*25',
    '[C]12[C]3*4*31*24',
    '[C]12[C]3*1*23',
    '[C]12[C]34*56*3*145*26',
    '[C]12[C]34*5*13*245',
    '[C]123[C]45*167*4*256*37',
    '[C][C]*',
    '[C][C]1**1',
    '[C][C]12*3*1*23',
    '[C][C]']

atom2_cannonical = [c.GetCanonicalSmiles(s) for s in atom2[:-1]] # load and remake smiles using rdkit
atom2_cannonical += ['[C][C]']
json.dump(atom2_cannonical,open('./Output/1Skeleton2.json','w'))

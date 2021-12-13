AdsorptionConfiguration_MS2021
=========================================
In hope to understand the conversion of large adsorbates, we develop overall framework for the enumeration large adsorbate configurations. This work contains the DFT data for related manuscript to be submitted.

Guide
-----
The json file contains stable configurations with keys:
- smiles: Configuration smiles.
- smiles_2nd_layer: Configuration smiles with second layer surface atom as He to differentiate hcp and fcc sites.
- atoms: DFT relaxed structure.
- size: Number of heavy atoms.
- gas_smiles: Gas molecule smiles.
- surface: Name of the surface.
- E_f: High fidelity energy of the configuration. Calculated by subtracting the total energy with surface energy followed by referencing to CH4, H2 and H2O.

Publications
------------
If you use this data, please cite:

Geun Ho Gu, Miriam Lee, Yousung Jung, Dionisios G. Vlachos. Automated Exploitation of the Big Configuration Space of Large Adsorbates on Transition Metals Reveals Chemistry Feasibility, In preparation


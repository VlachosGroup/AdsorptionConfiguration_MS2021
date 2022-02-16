AdsorptionConfiguration_MS2021
=========================================
In hope to understand the conversion of large molecules, we develop a framework for the enumeration of large adsorbate configurations. This repository contains the DFT data and enumeration code for the related work.

Installation
------------
You can simply clone this repository

```git clone https://github.com/VlachosGroup/AdsorptionConfiguration_MS2021```

Data Guide
----------
The json file contains stable configurations with keys:
- smiles: Configuration smiles.
- smiles_2nd_layer: Configuration smiles with second layer surface atom as He to differentiate hcp and fcc sites.
- atoms: DFT relaxed structure.
- size: Number of heavy atoms.
- gas_smiles: Gas molecule smiles.
- surface: Name of the surface.
- E_f: High fidelity energy of the configuration. Calculated by subtracting the total energy with surface energy followed by referencing to CH4, H2 and H2O.

Enumeration Code
----------------
The codes are in enumeration folder, with an example output in enumeration/Output folder. The code needs to be executed in the order of first digit in file name. The enumeration time for smaller molecules are fast, but starts take several hours for 4 heteroatom molecules. As the data size get large for larger molecules, we included the result for <4 heteroatoms.

Fingerprint-like descriptor-based logistic regression
-----------------------------------------------------
The codes are in FLDLR folder, with an example output in FLDLR/Output folder. "1MLTrain.py" trains the FLDLR model, and "2MLScreen.py" make model predictions. The training and prediction time for smaller molecules are fast, but starts take several hours for 4 heteroatom molecules. As the data size get large for larger molecules, we included the result for <4 heteroatoms.

Dependencies
------------
- Numpy (1.19.5 tested)
- rdkit (2020.03.2 tested)
- Atomic Simulation Environment (3.20.1 tested)
- scipy (1.5.2 tested)
- sklearn (0.23.2 tested)
- tqdm (4.50.2 tested)
- pandas (1.1.3 tested)

Publications
------------
If you use this data, please cite:

Geun Ho Gu, Miriam Lee, Yousung Jung, Dionisios G. Vlachos. Automated Exploitation of the Big Configuration Space of Large Adsorbates on Transition Metals Reveals Chemistry Feasibility, submitted


# JANUS
Tool and Datasets for the paper "JANUS: A Difference-oriented Analyzer for Financial Backdoors in Solidity Smart Contracts"
# Requirements
1. Basic environment:
Ubuntu 20.04, Python 3.8.10

2. Packages can be installed using pip:  func_timeout==4.3.5, fuzzywuzzy==0.18.0, py_solc_x==2.0.2, slither_analyzer==0.8.3, solidity_parser==0.1.1, tqdm==4.65.0, z3==0.2.0, z3_solver==4.12.4.0, dgl==1.1.2, matplotlib==3.7.4, networkx==3.1, numpy==1.24.3, pandas==1.5.3, rapidfuzz==3.5.2, scikit_learn==1.3.2, thefuzz==0.20.0, torch==2.1.1

3. Optional:
cuda 11.8
# Usage
1. Put the contract to be analyzed in JANUS/contracts/

2. Then run 

   `cd  JANUS/validation/`   

   `python  validate.py  --fvars=var1,var2,...  --sol=filename`

     options:

     `fvars` (optional): speicifying the financial variables or other target variables

     `sol`:  speicifying the name of the contract 
# Examples
1. Examples with backdoors: `JANUS/contracts/example.sol` `JANUS/contracts/example2.sol`
2. An example without backdoors: `JANUS/contracts/safe_example.sol `
3. An example with backdoors that other tools cannot detect: `JANUS/contracts/example_pm.sol`

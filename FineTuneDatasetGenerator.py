"""
Author: Egheosa Ogbomo
Date: 21/11/2024

An implementation of a genetic algoritm using XGBoost as 
fitness function for several property predictions
"""

############### ENVIRONMENT SETUP ############
import subprocess
import sys

def runcmd(cmd, verbose = False, *args, **kwargs):
    #bascially allows python to run a bash command, and the code makes sure 
    #the error of the subproceess is communicated if it fails
    process = subprocess.run(
        cmd,
        text=True,
        shell=True)
    
    return process

sys.path.append('/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/GeneticAlgoMLRun')

################# IMPORTS ###################
import GeneticAlgorithmHelperFunction as GAF
from rdkit import Chem
from rdkit.Chem import Draw
from random import sample
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolDrawing, DrawingOptions
from random import choice as rnd
from os.path import join
from time import sleep
import sys
from copy import deepcopy
import os
import glob
import pandas as pd
import random
import numpy as np
import shutil
import traceback
from gt4sd.properties import PropertyPredictorRegistry
import csv

file_path = '/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/GeneticAlgoMLRun/ModelsandDatasets/Viscosity_40C_Test_Descriptors.csv'
dataset = pd.read_csv(file_path)

# Define initial weightings for each category and remaining scores (user-defined)
user_weights = {
    'ViscScore': 20,  # Example weights; user-defined
    'HCScore': 20,
    'TCScore': 20,
    'Toxicity': 20,
    'DVIScore': 20,
    'SCScore': 20
}

Mols = dataset['SMILES']
fragments = ['CCCC', 'CCCCC', 'C(CC)CCC', 'CCC(CCC)CC', 'CCCC(C)CC', 'CCCCCCCCC', 'CCCCCCCC', 'CCCCCC', 'C(CCCCC)C', 'CC=OOCC(CC=O)O(CO)CO']

Mols = [Chem.MolFromSmiles(x) for x in Mols]
ReplaceMols = Mols
fragments = [Chem.MolFromSmiles(x) for x in fragments]

### ATOM NUMBERS
Atoms = ['C', 'O']
AtomMolObjects = [Chem.MolFromSmiles(x) for x in Atoms]
AtomicNumbers = []

# Getting Atomic Numbers for Addable Atoms
for Object in AtomMolObjects:
     for atom in Object.GetAtoms():
          AtomicNumbers.append(atom.GetAtomicNum())         

### BOND TYPES
BondTypes = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE]

###### Implementing Genetic Algorithm Using Functions Above

Mutations = ['AddAtom', 'ReplaceAtom', 'ReplaceBond', 'RemoveAtom', 'AddFragment', 'RemoveFragment', 'Napthalenate', 'Esterify', 'Glycolate']

#GENETIC ALGORITHM HYPERPARAMETERS
CopyCommand = 'cp'
Silent = True # Edit outputs to only print if this flag is False
NumElite = 25
IDcounter = 1
FirstGenerationAttempts = 0
MasterMoleculeList = [] #Keeping track of all generated molecules
FirstGenSimList = []
MaxNumHeavyAtoms = 50
MinNumHeavyAtoms = 5
MutationRate = 0.95
showdiff = False # Whether or not to display illustration of each mutation
GenerationSize = 5000000
LOPLS = False # Whether or not to use OPLS or LOPLS, False uses OPLS
# MaxGenerations = 500
MaxMutationAttempts = 2000
Fails = 0
NumRuns = 5
NumAtoms = 10000
Agent = 'Agent1'

STARTINGDIR = deepcopy(os.getcwd())
PYTHONPATH = 'python3'
Generation = 1

Napthalenes = ['C12=CC=CC=C2C=CC=C1', 'C1CCCC2=CC=CC=C12', 'C1CCCC2CCCCC12', 'C1CCC2=CC=CC=C12']

# Master Dataframe where molecules from all generations will be stored
MoleculeDatabase = pd.DataFrame(columns=['SMILES', 'MolObject', 'MutationList', 'HeavyAtoms', 'ID', 'MolMass', 'Predecessor'])

# Function to append a single molecule to the CSV file
def append_molecule_to_csv(file_name, molecule_row):
    """
    Appends a single molecule's data to the CSV file.
    """
    try:
        # Check if the file exists to determine if we need headers
        file_exists = os.path.isfile(file_name)
        
        # Append the row to the CSV file
        with open(file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # Write the header if the file is new
            if not file_exists:
                writer.writerow(molecule_row.index)
            
            # Write the molecule's data
            writer.writerow(molecule_row.values)
    except Exception as e:
        print(f"Error appending to CSV: {e}")
        traceback.print_exc()

# Initialise population 
while len(MoleculeDatabase) < GenerationSize:

    print('\n###########################################################')
    print(f'Attempt number: {FirstGenerationAttempts}')
    StartingMolecule = rnd(Mols) #Select starting molecule

    print(len(MoleculeDatabase))

    Mutation = rnd(Mutations)
    AromaticMolecule = fragments[-1]

    # Perform mutation 
    result = GAF.Mutate(StartingMolecule, Mutation, AromaticMolecule, AtomicNumbers, BondTypes, Atoms, showdiff, Fragments=fragments, Napthalenes=Napthalenes, Mols=ReplaceMols)

    # Implement checks based on predetermined criteria (MolLength, Illegal Substructs etc.)
    if GAF.GenMolChecks(result, MasterMoleculeList, MaxNumHeavyAtoms, MinNumHeavyAtoms, MaxAromRings=2) == None:
        MutMol = None

    elif GAF.count_c_and_o(result[2]) > MaxNumHeavyAtoms:
        print('Mol too heavy')
        MutMol = None
        continue

    else:
        HeavyAtoms = result[0].GetNumHeavyAtoms() # Get number of heavy atoms in molecule
        MutMol = result[0] # Get Mol object of mutated molecule
        MolMass = GAF.GetMolMass(MutMol) # Get estimate of of molecular mass 
        MutMolSMILES = result[2] # SMILES of mutated molecule
        Predecessor = [None] # Get history of last two mutations performed on candidate
        ID = IDcounter
    
        try:
            Name = f'Generation_1_Molecule_{ID}' # Set name of Molecule as its SMILES string

            # Update Molecule database
            MoleculeDatabase = GAF.DataUpdate(MoleculeDatabase, IDCounter=IDcounter, MutMolSMILES=MutMolSMILES, MutMol='', HeavyAtoms=HeavyAtoms,
                                                MutationList='', ID=Name, MolMass=MolMass, Predecessor='')
         
            # Generate list of molecules to simulate in this generation
            FirstGenSimList.append([Name, MutMolSMILES])
            MasterMoleculeList.append(MutMolSMILES) #Keep track of already generated molecules
            print(f'Final Molecule SMILES: {MutMolSMILES}')
            IDcounter +=1 

            # Append the last row to the CSV file
            last_row = MoleculeDatabase.tail(1).iloc[0]
            append_molecule_to_csv(f'{STARTINGDIR}/MoleculeDatabase.csv', last_row)

        except Exception as E:
            print(E)
            traceback.print_exc()
            continue

    FirstGenerationAttempts += 1

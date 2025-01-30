############### ENVIRONMENT SETUP AND IMPORTS ############
import pandas as pd
import numpy as np
from random import random, choice
from copy import deepcopy
from rdkit import Chem
import traceback
import GeneticAlgorithmHelperFunction as GAF  # Assuming this contains your helper functions
import os
from random import choice as rnd
import subprocess
from pymoo.indicators.hv import HV  # For hypervolume calculation


# File paths and initialization
file_path = 'C:/Users/eeo21/Desktop/Datasets/SeedDatabase.csv'
# file_path = '/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/GeneticAlgoMLRun/ModelsandDatasets/SeedDatabase.csv'
dataset = pd.read_csv(file_path)

STARTINGDIR = deepcopy(os.getcwd())

def runcmd(cmd, verbose = False, *args, **kwargs):
    #bascially allows python to run a bash command, and the code makes sure 
    #the error of the subproceess is communicated if it fails
    process = subprocess.run(
        cmd,
        text=True,
        shell=True)
    
    return process

# Genetic algorithm parameters
GenerationSize = 5
MutationRate = 0.85
MaxGenerations = 500
MaxNumHeavyAtoms = 45
MinNumHeavyAtoms = 5
NumElite = 5
MaxMutationAttempts = 2000
showdiff = False

Napthalenes = ['C12=CC=CC=C2C=CC=C1', 'C1CCCC2=CC=CC=C12', 'C1CCCC2CCCCC12', 'C1CCC2=CC=CC=C12']
Mutations = ['AddAtom', 'ReplaceAtom', 'ReplaceBond', 'RemoveAtom', 'AddFragment', 'RemoveFragment', 'Napthalenate', 'Esterify', 'Glycolate']
fragments = ['CCCC', 'CCCCC', 'C(CC)CCC', 'CCC(CCC)CC', 'CCCC(C)CC', 'CCCCCCCCC', 'CCCCCCCC', 'CCCCCC', 'C(CCCCC)C', 'CC=OOCC(CC=O)O(CO)CO']

Mols = dataset['SMILES']
Mols = [Chem.MolFromSmiles(x) for x in Mols[:100]]
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

# Objectives for NSGA-II
objectives = ['ViscScore', 'HCScore', 'TCScore', 'Toxicity', 'DVIScore', 'SCScore']
objectives_to_invert = ['Toxicity', 'ViscScore', 'SCScore']  # Lower is better for these objectives

# Initialization
Mols = dataset['SMILES']
Mols = [Chem.MolFromSmiles(x) for x in Mols]
MasterMoleculeList = []
IDcounter = 1
Generation = 1

def evaluate_population(GenSimList, MOLSMILESList, MoleculeDatabase, GenerationDatabase):

    for Molecule, MOLSMILES in GenSimList:
        try:
            ### Similarity Scores
            Scores = GAF.TanimotoSimilarity(MOLSMILES, MOLSMILESList)
            AvScore = 1 - (sum(Scores) / GenerationSize) # The higher the score, the less similar the molecule is to others

            ## SCScore
            SCScore = GAF.SCScore(MOLSMILES)
            SCScoreNorm = SCScore/5

            ## Toxicity
            ToxNorm = GAF.Toxicity(MOLSMILES)

            ### Viscosity
            DVisc40 = GAF.Visc40ML(MOLSMILES)
            DVisc100 = GAF.Visc100ML(MOLSMILES)

            ### Density
            Dens40 = GAF.Dens40ML(MOLSMILES)
            Dens100 = GAF.Dens100ML(MOLSMILES)

            ### Heat Capacity
            HC40 = GAF.HeatCapacity40ML(MOLSMILES)
            HC100 = GAF.HeatCapacity100ML(MOLSMILES)

            ### Thermal Conductivity
            TC40 = GAF.ThermalConductivity40ML(MOLSMILES)
            TC100 = GAF.ThermalConductivity100ML(MOLSMILES)

            ## Viscosity Index
            DVI = GAF.GetDVI(DVisc40, DVisc100)

            #Update Molecule Database
            IDNumber = int(Molecule.split('_')[-1])
            MoleculeDatabase.at[IDNumber, 'Density100C'] = Dens100 #Density 40C
            MoleculeDatabase.at[IDNumber, 'Density40C'] = Dens40 #Density 100C
            MoleculeDatabase.at[IDNumber, 'DViscosity40C'] = DVisc40 #DVisc 40C
            MoleculeDatabase.at[IDNumber, 'DViscosity100C'] = DVisc100 #DVisc 100C
            MoleculeDatabase.at[IDNumber, 'HeatCapacity_40C'] = HC40
            MoleculeDatabase.at[IDNumber, 'HeatCapacity_100C'] = HC100
            MoleculeDatabase.at[IDNumber, 'ThermalConductivity_40C'] = TC40
            MoleculeDatabase.at[IDNumber, 'ThermalConductivity_100C'] = TC100
            MoleculeDatabase.at[IDNumber, 'DVI'] = DVI
            MoleculeDatabase.at[IDNumber, 'Toxicity'] = ToxNorm
            MoleculeDatabase.at[IDNumber, 'SCScore'] = SCScoreNorm
            MoleculeDatabase.at[IDNumber, 'SimilarityScore'] = AvScore

            #Update Generation Database
            GenerationDatabase.at[IDNumber, 'Density100C'] = Dens100 #Density 40C
            GenerationDatabase.at[IDNumber, 'Density40C'] = Dens40 #Density 100C
            GenerationDatabase.at[IDNumber, 'DViscosity40C'] = DVisc40 #DVisc 40C
            GenerationDatabase.at[IDNumber, 'DViscosity100C'] = DVisc100 #DVisc 100C
            GenerationDatabase.at[IDNumber, 'HeatCapacity_40C'] = HC40
            GenerationDatabase.at[IDNumber, 'HeatCapacity_100C'] = HC100
            GenerationDatabase.at[IDNumber, 'ThermalConductivity_40C'] = TC40
            GenerationDatabase.at[IDNumber, 'ThermalConductivity_100C'] = TC100
            GenerationDatabase.at[IDNumber, 'DVI'] = DVI
            GenerationDatabase.at[IDNumber, 'Toxicity'] = ToxNorm
            GenerationDatabase.at[IDNumber, 'SCScore'] = SCScoreNorm
            GenerationDatabase.at[IDNumber, 'SimilarityScore'] = AvScore

        except Exception as E:
            print(E)
            traceback.print_exc()

    #### Generate Scores
    ViscScores_40C = MoleculeDatabase['DViscosity40C'].tolist()
    ViscScores_100C = MoleculeDatabase['DViscosity100C'].tolist()

    HCScores_40C = MoleculeDatabase['HeatCapacity_40C'].tolist()
    HCScores_100C = MoleculeDatabase['HeatCapacity_100C'].tolist()

    TCScores_40C = MoleculeDatabase['DViscosity40C'].tolist()
    TCScores_100C = MoleculeDatabase['DViscosity100C'].tolist()

    SCScores = MoleculeDatabase['SCScore'].tolist()
    DVIScores = MoleculeDatabase['DVI'].tolist()
    ToxicityScores = MoleculeDatabase['Toxicity'].tolist()
    SimilarityScores = MoleculeDatabase['SimilarityScore'].tolist()
    MoleculeNames = MoleculeDatabase['ID'].tolist()

    ViscosityScore_40C  = list(zip(MoleculeNames, ViscScores_40C)) 
    ViscosityScore_100C  = list(zip(MoleculeNames, ViscScores_100C)) 

    TCScore_40C  = list(zip(MoleculeNames, TCScores_40C)) 
    TCScore_100C  = list(zip(MoleculeNames, TCScores_100C)) 

    HCScore_40C  = list(zip(MoleculeNames, HCScores_40C)) 
    HCScore_100C  = list(zip(MoleculeNames, HCScores_100C)) 

    MolecularComplexityScore  = list(zip(MoleculeNames, SCScores)) 

    DVIScore  = list(zip(MoleculeNames, DVIScores)) 

    ToxicityScore  = list(zip(MoleculeNames, ToxicityScores)) 

    # Apply the normalization function
    Viscosity_normalized_molecule_scores_40C = [(1-x[1]) for x in GAF.min_max_normalize(ViscosityScore_40C)]
    Viscosity_normalized_molecule_scores_100C = [(1-x[1]) for x in GAF.min_max_normalize(ViscosityScore_100C)]
    HC_normalized_molecule_scores_40C = [(x[1]) for x in GAF.min_max_normalize(HCScore_40C)]
    HC_normalized_molecule_scores_100C = [(x[1]) for x in GAF.min_max_normalize(HCScore_100C)]
    TC_normalized_molecule_scores_40C = [(x[1]) for x in GAF.min_max_normalize(TCScore_40C)]
    TC_normalized_molecule_scores_100C = [(x[1]) for x in GAF.min_max_normalize(TCScore_100C)]
    DVI_normalized_molecule_scores = [x[1] for x in GAF.min_max_normalize(DVIScore)]
    Tox_normalized_molecule_scores = [(1-x[1]) for x in GAF.min_max_normalize(ToxicityScore)]
    SC_normalized_molecule_scores = [(1-x[1]) for x in GAF.min_max_normalize(MolecularComplexityScore)]

    MoleculeDatabase['ViscNormalisedScore_40C'] = Viscosity_normalized_molecule_scores_40C
    MoleculeDatabase['ViscNormalisedScore_100C'] = Viscosity_normalized_molecule_scores_100C
    MoleculeDatabase['HCNormalisedScore_40C'] = HC_normalized_molecule_scores_40C
    MoleculeDatabase['HCNormalisedScore_100C'] = HC_normalized_molecule_scores_100C
    MoleculeDatabase['TCNormalisedScore_40C'] = TC_normalized_molecule_scores_40C
    MoleculeDatabase['TCNormalisedScore_100C'] = TC_normalized_molecule_scores_100C

    MoleculeDatabase['Normalized_DVIScore'] = DVI_normalized_molecule_scores
    MoleculeDatabase['Normalized_Viscosity'] = Tox_normalized_molecule_scores
    MoleculeDatabase['Normalized_SCScore'] = SC_normalized_molecule_scores

    # Combine scores into categories
    MoleculeDatabase['Normalized_ViscScore'] = (
        (MoleculeDatabase['ViscNormalisedScore_40C'] + MoleculeDatabase['ViscNormalisedScore_100C']) /2
    )
    MoleculeDatabase['Normalized_HCScore'] = (
        (MoleculeDatabase['HCNormalisedScore_40C'] + MoleculeDatabase['HCNormalisedScore_100C'])/2
    )
    MoleculeDatabase['Normalized_TCScore'] = (
        (MoleculeDatabase['TCNormalisedScore_40C'] + MoleculeDatabase['TCNormalisedScore_100C'])/2
    )

    return MoleculeDatabase, GenerationDatabase

def pareto_front_sorting(population, objectives):
    """
    Perform non-dominated sorting on the population based on the given objectives.
    """
    fronts = []
    domination_count = np.zeros(len(population), dtype=int)
    dominated_solutions = [[] for _ in range(len(population))]
    ranks = np.zeros(len(population), dtype=int)

    for i in range(len(population)):
        for j in range(len(population)):
            if i != j:
                dominates = all(population.iloc[i][obj] <= population.iloc[j][obj] for obj in objectives) and \
                            any(population.iloc[i][obj] < population.iloc[j][obj] for obj in objectives)
                if dominates:
                    dominated_solutions[i].append(j)
                elif all(population.iloc[j][obj] <= population.iloc[i][obj] for obj in objectives) and \
                     any(population.iloc[j][obj] < population.iloc[i][obj] for obj in objectives):
                    domination_count[i] += 1

        if domination_count[i] == 0:
            ranks[i] = 0
            if len(fronts) == 0:
                fronts.append([])
            fronts[0].append(i)

    front_index = 0
    while len(fronts[front_index]) > 0:
        next_front = []
        for i in fronts[front_index]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    ranks[j] = front_index + 1
                    next_front.append(j)
        front_index += 1
        fronts.append(next_front)

    return fronts[:-1], ranks

def crowding_distance_assignment(front, population, objectives):
    """
    Compute crowding distance for solutions in a single Pareto front.
    """
    crowding_distances = np.zeros(len(front))
    sorted_front = population.loc[front]
    
    for obj in objectives:
        sorted_indices = sorted_front.sort_values(by=obj).index
        sorted_values = sorted_front.loc[sorted_indices, obj].values

        try:
            crowding_distances[sorted_indices[0]] = float('inf')
            crowding_distances[sorted_indices[-1]] = float('inf')
        except IndexError:
            crowding_distances[-1] = float('inf')


        obj_range = sorted_values[-1] - sorted_values[0]
        if obj_range > 0:
            for i in range(1, len(sorted_indices) - 1):
                crowding_distances[i] += (
                    sorted_values[i + 1] - sorted_values[i - 1]
                ) / obj_range

    return crowding_distances

def select_next_generation(population, fronts, objectives, population_size):
    """
    Select the next generation based on Pareto fronts and crowding distance.
    """
    next_generation = []
    for front in fronts:
        if len(next_generation) + len(front) <= population_size:
            next_generation.extend(front)
        else:
            distances = crowding_distance_assignment(front, population, objectives)
            sorted_by_distance = np.argsort(-distances)
            remaining_slots = population_size - len(next_generation)
            next_generation.extend(np.array(front)[sorted_by_distance[:remaining_slots]])
            break

    return population.iloc[next_generation]

def calculate_hypervolume(pareto_front, objectives, reference_point):
    objective_values = pareto_front[objectives].to_numpy()
    hv_indicator = HV(ref_point=reference_point)
    return hv_indicator(objective_values)

# Initialize Population
print('Initialising Popuplation')

# Master Dataframe where molecules from all generations will be stored
MoleculeDatabase = pd.DataFrame(columns=['SMILES', 'MolObject', 'MutationList', 'HeavyAtoms', 'ID', 'MolMass', 'Predecessor', 'Score', 'Density100C', 'DViscosity40C',
                                        'DViscosity100C', 'DVI', 'Toxicity', 'SCScore', 'Density40C', 'SimilarityScore',
                                        'ThermalConductivity_40C', 'ThermalConductivity_100C', 'HeatCapacity_40C', 'HeatCapacity_100C'])

# Generation Dataframe to store molecules from each generation
GenerationDatabase = pd.DataFrame(columns=['SMILES', 'MolObject', 'MutationList', 'HeavyAtoms', 'ID', 'MolMass', 'Predecessor', 'Score', 'Density100C', 'DViscosity40C',
                                        'DViscosity100C', 'DVI', 'Toxicity', 'SCScore', 'Density40C', 'SimilarityScore',
                                        'ThermalConductivity_40C', 'ThermalConductivity_100C', 'HeatCapacity_40C', 'HeatCapacity_100C'])

GenSimList = []
while len(MoleculeDatabase) < GenerationSize:

    print('\n###########################################################')
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
            MoleculeDatabase = GAF.DataUpdate(MoleculeDatabase, IDCounter=IDcounter, MutMolSMILES=MutMolSMILES, MutMol=MutMol, HeavyAtoms=HeavyAtoms,
                                                MutationList=[None, None, Mutation], ID=Name, MolMass=MolMass, Predecessor=Predecessor)

            GenerationDatabase = GAF.DataUpdate(GenerationDatabase, IDCounter=IDcounter, MutMolSMILES=MutMolSMILES, MutMol=MutMol, HeavyAtoms=HeavyAtoms,
                                                MutationList=[None, None, Mutation], ID=Name, MolMass=MolMass, Predecessor=Predecessor)
            
            # Generate list of molecules to simulate in this generation
            GenSimList.append([Name, MutMolSMILES])
            MasterMoleculeList.append(MutMolSMILES) #Keep track of already generated molecules
            print(f'Final Molecule SMILES: {MutMolSMILES}')
            IDcounter +=1 

        except Exception as E:
            print(E)
            traceback.print_exc()
            continue

MoleculeDatabase.to_csv(f'{STARTINGDIR}/MoleculeDatabase.csv')
GenerationDatabase.to_csv(f'{STARTINGDIR}/Generation_{Generation}_Database.csv')

no_improvement_generations = 0
convergence_threshold = 1e-4
previous_hypervolume = 0
EarlyStop = 50

MOLSMILESList = [x[1] for x in GenSimList]

# Main NSGA-II Loop
for Generation in range(1, MaxGenerations + 1):
    print(f"### Generation {Generation} ###")

    # Evaluate objectives and normalize scores
    MoleculeDatabase = evaluate_population(GenSimList, MOLSMILESList, MoleculeDatabase, GenerationDatabase)

    # Perform non-dominated sorting
    pareto_fronts, ranks = pareto_front_sorting(MoleculeDatabase, [f"Normalized_{obj}" for obj in objectives])

    print(pareto_fronts)
    print(ranks)

    # Assign crowding distances
    for front in pareto_fronts:
        MoleculeDatabase.loc[front, 'CrowdingDistance'] = crowding_distance_assignment(front, MoleculeDatabase, [f"Normalized_{obj}" for obj in objectives])

    # Select the next generation
    next_generation = select_next_generation(MoleculeDatabase, pareto_fronts, [f"Normalized_{obj}" for obj in objectives], GenerationSize)

    # Save the current generation and global archive
    next_generation.to_csv(f"{STARTINGDIR}/Generation_{Generation}_Database.csv", index=False)
    combined_population = pd.concat([GlobalParetoArchive, next_generation]).reset_index(drop=True)
    combined_pareto_fronts, _ = pareto_front_sorting(combined_population, [f"Normalized_{obj}" for obj in objectives])
    GlobalParetoArchive = combined_population.iloc[combined_pareto_fronts[0]].reset_index(drop=True)
    GlobalParetoArchive.to_csv(f"{STARTINGDIR}/GlobalParetoArchive.csv", index=False)

    # Apply mutations to generate new molecules
    mutated_population = []
    for _, molecule in next_generation.iterrows():
            # Perform NSGA-II steps
        current_hypervolume = calculate_hypervolume(GlobalParetoArchive)

        if abs(current_hypervolume - previous_hypervolume) < convergence_threshold:
            no_improvement_generations += 1
            if no_improvement_generations >= EarlyStop:
                print("Converged: No significant improvement for 10 generations.")
                break
        else:
            no_improvement_generations = 0

        previous_hypervolume = current_hypervolume

        if rnd() <= MutationRate:
            try:
                StartingMolecule = Chem.MolFromSmiles(molecule['SMILES'])
                if GAF.count_c_and_o(result[2]) > 32:
                    MutationList = ['RemoveFragment','ReplaceCandidate']
                else:
                    MutationList = Mutations

                result = GAF.Mutate(
                    StartingMolecule, rnd(Mutations), None, None, None, None, showdiff,
                    Fragments=fragments, Napthalenes=Napthalenes, Mols=ReplaceMols
                )

                if result and GAF.GenMolChecks(result, MasterMoleculeList, MaxNumHeavyAtoms, MinNumHeavyAtoms):
                    HeavyAtoms = result[0].GetNumHeavyAtoms()
                    MolMass = GAF.GetMolMass(result[0])
                    SMILES = result[2]
                    mutated_population.append({
                        'SMILES': SMILES,
                        'MolObject': result[0],
                        'MutationList': molecule['MutationList'] + [Mutation] if molecule['MutationList'] else [Mutation],
                        'HeavyAtoms': HeavyAtoms,
                        'ID': f'Molecule_{IDcounter}',
                        'MolMass': MolMass,
                        **{key: None for key in objectives}
                    })

                    MasterMoleculeList.append(SMILES)
                    IDcounter += 1
                else:
                    mutated_population.append(molecule)
            except Exception as e:
                print(f"Mutation failed: {e}")
                traceback.print_exc()
                mutated_population.append(molecule)
        else:
            mutated_population.append(molecule)

    # Update MoleculeDatabase with the new generation
    MoleculeDatabase = pd.DataFrame(mutated_population)

# Save final Pareto archive
GlobalParetoArchive.to_csv(f"{STARTINGDIR}/FinalParetoArchive.csv", index=False)
print("Optimization completed. Final Pareto archive saved.")


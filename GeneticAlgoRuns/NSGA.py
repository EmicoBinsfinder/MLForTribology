import sys

sys.path.append('/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/GeneticAlgoMLRun/NSGA')

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
# file_path = 'C:/Users/eeo21/Desktop/Datasets/SeedDatabase.csv'
file_path = '/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/GeneticAlgoMLRun/ModelsandDatasets/SeedDatabase.csv'
dataset = pd.read_csv(file_path)

print(os.getcwd())

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
GenerationSize = 50
MutationRate = 0.85
MaxGenerations = 500
MaxNumHeavyAtoms = 45
MinNumHeavyAtoms = 5
MaxMutationAttempts = 2000
showdiff = False
no_improvement_generations = 0
convergence_threshold = 1e-4
previous_hypervolume = 0
EarlyStop = 50

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

# Fitness functions mapping
fitness_functions = {
    'ViscScore': lambda smiles: (GAF.Visc40ML(smiles) + GAF.Visc100ML(smiles)) / 2,
    'HCScore': lambda smiles: (GAF.HeatCapacity40ML(smiles) + GAF.HeatCapacity100ML(smiles)) / 2,
    'TCScore': lambda smiles: (GAF.ThermalConductivity40ML(smiles) + GAF.ThermalConductivity100ML(smiles)) / 2,
    'Toxicity': lambda smiles: GAF.Toxicity(smiles),
    'DVIScore': lambda smiles: GAF.GetDVI(GAF.Visc40ML(smiles), GAF.Visc100ML(smiles)),
    'SCScore': lambda smiles: GAF.SCScore(smiles) / 5,
}

# Initialization
MasterMoleculeList = []
IDcounter = 1
Generation = 1

# Global Pareto Archive
GlobalParetoArchive = pd.DataFrame(columns=['SMILES', 'MolObject', 'MutationList', 'HeavyAtoms', 'ID', 'MolMass',
                                            'Density100C', 'DViscosity40C', 'DViscosity100C', 'DVI', 'Toxicity',
                                            'SCScore', 'Density40C', 'SimilarityScore', 'ThermalConductivity_40C',
                                            'ThermalConductivity_100C', 'HeatCapacity_40C', 'HeatCapacity_100C'] + objectives)

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
    MoleculeDatabase['Normalized_Toxicity'] = Tox_normalized_molecule_scores
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
    Assigns each molecule to a Pareto front and assigns ranks.

    Returns:
        - fronts: List of Pareto fronts (each front is a list of molecule indices).
        - population_with_ranks: DataFrame with molecules and their assigned front and rank.
    """
    if population.empty:
        print("Population is empty! Returning empty results.")
        return [], population

    population_size = len(population)
    domination_count = np.zeros(population_size, dtype=int)
    dominated_solutions = [[] for x in range(population_size)]
    fronts = [[]]
    ranks = np.full(population_size, -1, dtype=int)  # Initialize all ranks as -1

    # Function to check if one solution dominates another
    def dominates(p1, p2):
        better_in_at_least_one = False
        for i in range(len(p1)):
            if p1[i] > p2[i]:  # Worse in one objective
                return False
            if p1[i] < p2[i]:  # Better in at least one
                better_in_at_least_one = True
        return better_in_at_least_one  # Must be better in at least one to dominate

    # Compare each pair of solutions to establish dominance relationships
    for i in range(population_size):
        for j in range(population_size):
            if i != j:
                p1 = population.iloc[i][objectives].values
                p2 = population.iloc[j][objectives].values
                
                if dominates(p1, p2):
                    dominated_solutions[i].append(j)
                elif dominates(p2, p1):
                    domination_count[i] += 1

        # If domination count is zero, it belongs to the first front
        if domination_count[i] == 0:
            ranks[i] = 0  # Assign to first Pareto front
            fronts[0].append(i)

    # Ensure there is at least one front before proceeding
    if not fronts[0]:  
        print("Warning: No solutions were assigned to the first Pareto front!")
        return [], population

    # Generate subsequent Pareto fronts
    front_index = 0
    while front_index < len(fronts) and fronts[front_index]:  
        next_front = []
        for i in fronts[front_index]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    ranks[j] = front_index + 1  # Assign next rank
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        front_index += 1

    # Assign Pareto front index and rank to the population DataFrame
    population_with_ranks = population.copy()
    population_with_ranks['ParetoFront'] = ranks
    population_with_ranks['Rank'] = ranks  # The rank is the front index

    return fronts, population_with_ranks

def crowding_distance_assignment(front_indices, population, objectives):
    """
    Compute the summed crowding distance for solutions in a single Pareto front.

    Args:
        front_indices (list): List of indices of molecules in the Pareto front.
        population (DataFrame): The entire population DataFrame.
        objectives (list): List of objective function names (normalized).
    
    Returns:
        dict: A dictionary mapping each molecule index to its summed crowding distance.
    """
    # Ensure front_indices is a flat list of integers
    front_indices = [idx for sublist in front_indices for idx in (sublist if isinstance(sublist, list) else [sublist])]
    
    # Ensure indices are aligned with DataFrame index
    front_indices = [idx for idx in front_indices if idx in population.index]

    if len(front_indices) == 0:
        return {}

    # Initialize crowding distances for each molecule in the front
    crowding_distances = {idx: 0 for idx in front_indices}

    for obj in objectives:
        # Sort the front based on the current objective
        sorted_indices = sorted(front_indices, key=lambda idx: population.loc[idx, obj])

        # Assign infinite distance to boundary solutions
        if len(sorted_indices) > 1:
            crowding_distances[sorted_indices[0]] = 1
            crowding_distances[sorted_indices[-1]] = 0.3

        obj_values = population.loc[sorted_indices, obj].values
        obj_range = max(obj_values) - min(obj_values) if len(obj_values) > 1 else 0

        # Compute crowding distance for intermediate solutions
        if obj_range > 0:
            for i in range(1, len(sorted_indices) - 1):
                prev_val = obj_values[i - 1]
                next_val = obj_values[i + 1]
                idx = sorted_indices[i]
                crowding_distances[idx] += (next_val - prev_val) / obj_range  # Summing across objectives

    return crowding_distances

def select_next_generation(population, fronts, population_size):
    """
    Select the next generation based on Pareto fronts and precomputed crowding distance.
    """
    next_generation = []
    
    for front in fronts:
        if len(next_generation) + len(front) <= population_size:
            next_generation.extend(front)
        else:
            front_df = population.loc[front].copy()
            
            front_df = front_df.sort_values(by="CrowdingDistance", ascending=False)

            remaining_slots = population_size - len(next_generation)
            selected_indices = front_df.index[:remaining_slots].tolist()
            
            next_generation.extend(selected_indices)
            break  # Stop after filling the population size

    return population.loc[next_generation].reset_index(drop=True)

def extract_top_candidates(population, pareto_fronts, objectives, num_candidates):
    """
    Extract the top `num_candidates` molecules based on:
    1. Pareto front rank (lower is better)
    2. Crowding distance (higher is better for diversity)
    3. Random selection if necessary

    Args:
        population (DataFrame): The full candidate population.
        pareto_fronts (list): List of Pareto fronts (sorted).
        objectives (list): List of objectives to consider.
        num_candidates (int): Number of candidates to select.

    Returns:
        DataFrame: Selected top candidates.
    """
    selected_candidates = []

    # Flatten pareto_fronts into a ranked list (front 0 first, then 1, etc.)
    sorted_candidates = []
    for rank, front in enumerate(pareto_fronts):
        for idx in front:
            sorted_candidates.append((idx, rank))  # Store index and Pareto front rank

    # Create a DataFrame with rank information
    sorted_df = population.loc[[idx for idx, _ in sorted_candidates]].copy()
    sorted_df["ParetoFront"] = [rank for _, rank in sorted_candidates]

    # Ensure 'CrowdingDistance' exists
    if "CrowdingDistance" not in sorted_df.columns:
        sorted_df["CrowdingDistance"] = 0  # Default value if not assigned

    # Sort by (1) Pareto front (ascending), (2) Crowding distance (descending)
    sorted_df = sorted_df.sort_values(by=["ParetoFront", "CrowdingDistance"], ascending=[True, False])

    # Select up to `num_candidates`
    if len(sorted_df) > num_candidates:
        # If more candidates than needed, perform selection
        selected_candidates = sorted_df.iloc[:num_candidates]
    else:
        # If we have fewer candidates than needed, add some randomly
        remaining = num_candidates - len(sorted_df)
        selected_candidates = sorted_df
        remaining_candidates = population.drop(sorted_df.index)  # Candidates not yet selected
        if len(remaining_candidates) > 0:
            random_selection = remaining_candidates.sample(min(remaining, len(remaining_candidates)))
            selected_candidates = pd.concat([selected_candidates, random_selection])

    return selected_candidates.reset_index(drop=True)


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

while len(GenSimList) < GenerationSize:

    print('\n###########################################################')
    StartingMolecule = rnd(Mols) #Select starting molecule

    print(len(MoleculeDatabase))

    Mutation = rnd(Mutations)
    AromaticMolecule = fragments[-1]
    try:
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
#         traceback.print_exc()
        continue


# Track already evaluated molecules to avoid redundant computation
# evaluated_smiles = set(MoleculeDatabase['SMILES'].tolist())
evaluated_smiles = set()

MOLSMILESList = [x[1] for x in GenSimList]
# new_molecules = [mol for mol in GenSimList if mol[1] not in evaluated_smiles]

# for Generation in range(1, MaxGenerations + 1):
for Generation in range(1, 3):
    print(f"### Generation {Generation} ###")
    print(GenSimList)

    # Evaluate only new molecules
    new_molecules = [mol for mol in GenSimList if mol[1] not in evaluated_smiles]
    
    MoleculeDatabase, GenerationDatabase = evaluate_population(new_molecules, MOLSMILESList, MoleculeDatabase, GenerationDatabase)
    
    # Add newly evaluated molecules to the set
    evaluated_smiles.update([mol[1] for mol in new_molecules])
    
    # Perform non-dominated sorting
    pareto_fronts, MoleculeDatabase = pareto_front_sorting(MoleculeDatabase, [f"Normalized_{obj}" for obj in objectives])

    pareto_fronts = [[MoleculeDatabase.index[i] for i in front] for front in pareto_fronts]

    # Assign crowding distances
    for front in pareto_fronts:
        crowding_distances = crowding_distance_assignment(front, MoleculeDatabase, [f"Normalized_{obj}" for obj in objectives])
        for idx, distance in crowding_distances.items():
            if idx in MoleculeDatabase.index:
                MoleculeDatabase.at[idx, 'CrowdingDistance'] = distance
                
    GenerationDatabase = extract_top_candidates(MoleculeDatabase, pareto_fronts, [f"Normalized_{obj}" for obj in objectives], NumElite)
    GenSimList = GenerationDatabase[['ID', 'SMILES']].values.tolist()
    print(GenSimList)
    print(GenerationDatabase)
    
    while len(GenSimList) < GenerationSize:
        MutMol = None
        # Get two parents using 3-way tournament selection
        Parent1 = choice(GenerationDatabase['SMILES'].to_list())
        Parent2 = choice(GenerationDatabase['SMILES'].to_list())

        try:
            result = GAF.Mol_Crossover(Chem.MolFromSmiles(Parent1), Chem.MolFromSmiles(Parent2))
            if result[2] == None:
                continue     
                
        except Exception as E:
            continue
        
        Predecessor = [Parent1, Parent2]
        # Molecule ID
        Name = f'Generation_{Generation}_Molecule_{IDcounter}'

        if GAF.count_c_and_o(result[2]) > 36:
            MutationList = ['RemoveFragment','ReplaceCandidate']
        else:
            MutationList = Mutations

        if random() <= MutationRate:
            Mutate = True
            print('Attempting to Mutate')
        else:
            Mutate = False    

        if Mutate:
            Mutation = rnd(MutationList)
            AromaticMolecule = fragments[-1]
            print(Mutation)

            # Perform mutation 
            result = GAF.Mutate(StartingMolecule, Mutation, AromaticMolecule, AtomicNumbers, BondTypes, Atoms, showdiff, Fragments=fragments, Napthalenes=Napthalenes, Mols=ReplaceMols)

        else:
            Mutation = None  

        # Implement checks based on predetermined criteria (MolLength, Illegal Substructs etc.)
        MutMol = GAF.GenMolChecks(result, MasterMoleculeList, MaxNumHeavyAtoms, MinNumHeavyAtoms, MaxAromRings=3)

        if MutMol == None:
            continue

        HeavyAtoms = result[0].GetNumHeavyAtoms() # Get number of heavy atoms in molecule

        if GAF.count_c_and_o(result[2]) > MaxNumHeavyAtoms:
            print('Mol too heavy')
            MutMol == None
            continue
            
        else:
            MutMol = result[0] # Get Mol object of mutated molecule
            MolMass = GAF.GetMolMass(MutMol) # Get estimate of of molecular mass 
            MutMolSMILES = result[2] # SMILES of mutated molecule
            # Update previous mutations object
            PreviousMutations = Mutation

            print(f'Final SMILES: {result[2]}')

            if GAF.count_c_and_o(result[2]) > MaxNumHeavyAtoms:
                print('Mol too heavy')
                MutMol = None
                continue

            try:
                Name = f'Generation_{Generation+1}_Molecule_{IDcounter}' # Set name of Molecule as its SMILES string

                # Update Molecule database
                MoleculeDatabase = GAF.DataUpdate(MoleculeDatabase, IDCounter=IDcounter, MutMolSMILES=MutMolSMILES, MutMol=MutMol, HeavyAtoms=HeavyAtoms,
                                                    MutationList=PreviousMutations, ID=Name, MolMass=MolMass, Predecessor=Predecessor)

                GenerationDatabase = GAF.DataUpdate(GenerationDatabase, IDCounter=IDcounter, MutMolSMILES=MutMolSMILES, MutMol=MutMol, HeavyAtoms=HeavyAtoms,
                                                    MutationList=PreviousMutations, ID=Name, MolMass=MolMass, Predecessor=Predecessor)

                # Generate list of molecules to simulate in this generation
                GenSimList.append([Name, MutMolSMILES])
                print(f'Final Molecule SMILES: {MutMolSMILES}')
                IDcounter +=1 

            except Exception as E:
                print(E)
                traceback.print_exc()
                continue
                
    MOLSMILESList = [x[1] for x in GenSimList]
    
    print(GenSimList)
            
#     sys.exit()
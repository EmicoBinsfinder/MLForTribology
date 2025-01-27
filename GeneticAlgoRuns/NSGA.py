############### ENVIRONMENT SETUP AND IMPORTS ############
import subprocess
import sys
import pandas as pd
import numpy as np
from random import sample, random, choice
from copy import deepcopy
from rdkit import Chem
import traceback
import GeneticAlgorithmHelperFunction as GAF  # Assuming this contains your helper functions

# File paths and initialization
file_path = '/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/GeneticAlgoMLRun/ModelsandDatasets/SeedDatabase.csv'
dataset = pd.read_csv(file_path)

STARTINGDIR = deepcopy(os.getcwd())
PYTHONPATH = 'python3'

# Parameters
GenerationSize = 50
MutationRate = 0.95
MaxGenerations = 500
MaxNumHeavyAtoms = 40
MinNumHeavyAtoms = 5
NumElite = 25
MaxMutationAttempts = 2000
Fails = 0

# Objectives for NSGA-II
objectives = ['ViscScore', 'HCScore', 'TCScore', 'Toxicity', 'DVIScore', 'SCScore']

# Initialization
Mols = dataset['SMILES']
Mols = [Chem.MolFromSmiles(x) for x in Mols[:200]]
MasterMoleculeList = []
IDcounter = 1
Generation = 1

# Helper Functions
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
    for obj in objectives:
        sorted_indices = np.argsort(population.iloc[front][obj].values)
        sorted_front = np.array(front)[sorted_indices]

        crowding_distances[sorted_indices[0]] = float('inf')
        crowding_distances[sorted_indices[-1]] = float('inf')

        obj_values = population.iloc[sorted_front][obj].values
        obj_range = max(obj_values) - min(obj_values)
        if obj_range > 0:
            for i in range(1, len(front) - 1):
                crowding_distances[sorted_indices[i]] += (obj_values[i + 1] - obj_values[i - 1]) / obj_range

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

# Initialize Population
MoleculeDatabase = pd.DataFrame(columns=['SMILES', 'MolObject', 'MutationList', 'HeavyAtoms', 'ID', 'MolMass', 'Predecessor', 'Score', 'Density100C',
                                         'DViscosity40C', 'DViscosity100C', 'DVI', 'Toxicity', 'SCScore', 'Density40C', 'SimilarityScore',
                                         'ThermalConductivity_40C', 'ThermalConductivity_100C', 'HeatCapacity_40C', 'HeatCapacity_100C'])

while len(MoleculeDatabase) < GenerationSize:
    StartingMolecule = choice(Mols)
    Mutation = choice(['AddAtom', 'ReplaceAtom', 'ReplaceBond', 'RemoveAtom', 'AddFragment', 'RemoveFragment'])
    result = GAF.Mutate(StartingMolecule, Mutation, None, None, None, None, False)
    if result and GAF.GenMolChecks(result, MasterMoleculeList, MaxNumHeavyAtoms, MinNumHeavyAtoms):
        HeavyAtoms = result[0].GetNumHeavyAtoms()
        MolMass = GAF.GetMolMass(result[0])
        SMILES = result[2]
        MoleculeDatabase = GAF.DataUpdate(MoleculeDatabase, IDcounter, SMILES, result[0], HeavyAtoms, [None, None, Mutation], f'Molecule_{IDcounter}', MolMass)
        MasterMoleculeList.append(SMILES)
        IDcounter += 1

# Global Pareto Archive
GlobalParetoArchive = pd.DataFrame(columns=MoleculeDatabase.columns)

# Main NSGA-II Loop
for Generation in range(1, MaxGenerations + 1):
    print(f"### Generation {Generation} ###")

    # Perform non-dominated sorting for the current population
    pareto_fronts, ranks = pareto_front_sorting(MoleculeDatabase, objectives)

    # Assign crowding distances for diversity
    for front in pareto_fronts:
        MoleculeDatabase.loc[front, 'CrowdingDistance'] = crowding_distance_assignment(front, MoleculeDatabase, objectives)

    # Select next generation
    next_generation = select_next_generation(MoleculeDatabase, pareto_fronts, objectives, GenerationSize)

    # Save the current generation to a CSV
    next_generation.to_csv(f"{STARTINGDIR}/Generation_{Generation}_Database.csv", index=False)

    # Update Global Pareto Archive
    combined_population = pd.concat([GlobalParetoArchive, next_generation]).reset_index(drop=True)
    combined_pareto_fronts, _ = pareto_front_sorting(combined_population, objectives)
    GlobalParetoArchive = combined_population.iloc[combined_pareto_fronts[0]].reset_index(drop=True)

    # Save the updated Pareto Archive to a CSV
    GlobalParetoArchive.to_csv(f"{STARTINGDIR}/GlobalParetoArchive.csv", index=False)

    # Apply mutation to molecules in the current generation
    mutated_population = []
    for index, molecule in next_generation.iterrows():
        if random() <= MutationRate:  # Apply mutation with probability = MutationRate
            try:
                StartingMolecule = Chem.MolFromSmiles(molecule['SMILES'])
                Mutation = choice(['AddAtom', 'ReplaceAtom', 'ReplaceBond', 'RemoveAtom', 'AddFragment', 'RemoveFragment'])
                result = GAF.Mutate(StartingMolecule, Mutation, None, None, None, None, False)

                if result and GAF.GenMolChecks(result, MasterMoleculeList, MaxNumHeavyAtoms, MinNumHeavyAtoms):
                    HeavyAtoms = result[0].GetNumHeavyAtoms()
                    MolMass = GAF.GetMolMass(result[0])
                    SMILES = result[2]

                    # Add mutated molecule to the list
                    mutated_population.append({
                        'SMILES': SMILES,
                        'MolObject': result[0],
                        'MutationList': molecule['MutationList'] + [Mutation],
                        'HeavyAtoms': HeavyAtoms,
                        'ID': f'Molecule_{IDcounter}',
                        'MolMass': MolMass,
                        **{key: None for key in objectives}  # Reset objectives to be recalculated
                    })
                    MasterMoleculeList.append(SMILES)
                    IDcounter += 1
                else:
                    mutated_population.append(molecule)  # Keep the original if mutation fails

            except Exception as e:
                print(f"Mutation failed for molecule {molecule['ID']}: {e}")
                traceback.print_exc()
                mutated_population.append(molecule)  # Keep the original molecule in case of failure
        else:
            mutated_population.append(molecule)  # No mutation, keep original

    # Convert mutated population to DataFrame
    MoleculeDatabase = pd.DataFrame(mutated_population)

    # Recalculate objectives for all molecules
    MoleculeDatabase = evaluate_population(MoleculeDatabase, fitness_functions)

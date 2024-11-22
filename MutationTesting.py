'''
Date 19th November 2024

Author: Egheosa Ogbomo

Script to test new mutation functions
'''

from rdkit import Chem
from rdkit.Chem import Draw
import rdkit
from random import sample
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from random import choice as rnd
import random
# from MoleculeDifferenceViewer import view_difference
from copy import deepcopy
import subprocess
import ast
import pandas as pd
import re
from math import log10
import os
# from gt4sd.properties import PropertyPredictorRegistry
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
from scipy.constants import Boltzmann
from scipy.optimize import curve_fit
import sys, argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import integrate
from tqdm import trange
from os.path import join as join
from os import chdir
from os import getcwd
from os import listdir
import os
import gzip
import shutil
import json
import fnmatch
import math
import six
from rdkit.Chem import rdFMCS

def view_difference(mol1, mol2):
    mcs = rdFMCS.FindMCS([mol1,mol2])
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    match1 = mol1.GetSubstructMatch(mcs_mol)
    target_atm1 = []
    for atom in mol1.GetAtoms():
        if atom.GetIdx() not in match1:
            target_atm1.append(atom.GetIdx())
    match2 = mol2.GetSubstructMatch(mcs_mol)
    target_atm2 = []
    for atom in mol2.GetAtoms():
        if atom.GetIdx() not in match2:
            target_atm2.append(atom.GetIdx())
    img = Draw.MolsToGridImage([mol1, mol2],highlightAtomLists=[target_atm1, target_atm2])
    img.show()

def MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff, Verbose=False):
    
    Mut_Mol = StartingMolecule.GetMol()
    MutMolSMILES = Chem.MolToSmiles(Mut_Mol)
    RI = Mut_Mol.GetRingInfo()
    
    try:
        NumRings = RI.GetNumRings()
    except:
        RI = None

    if RI != None:
        NumAromaticRings = Chem.rdMolDescriptors.CalcNumAromaticRings(StartingMolecule)
        if NumAromaticRings == 0:
            Chem.SanitizeMol(StartingMolecule, sanitizeOps=rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_ADJUSTHS, catchErrors=True)
            Chem.SanitizeMol(StartingMolecule, sanitizeOps=rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_CLEANUP, catchErrors=True)
            Chem.SanitizeMol(StartingMolecule, sanitizeOps=rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_FINDRADICALS, catchErrors=True)
            Chem.SanitizeMol(StartingMolecule, sanitizeOps=rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
            Chem.SanitizeMol(StartingMolecule, sanitizeOps=rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_SETCONJUGATION, catchErrors=True)
            Chem.SanitizeMol(StartingMolecule, sanitizeOps=rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_SETHYBRIDIZATION, catchErrors=True)
            Mut_Mol_Sanitized = Chem.SanitizeMol(StartingMolecule, sanitizeOps=rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_SYMMRINGS, catchErrors=True)
            print('Non-Aromatic Cyclic Hydrocarbon')
            print(MutMolSMILES)
        else:
            Mut_Mol_Sanitized = Chem.SanitizeMol(Mut_Mol, catchErrors=True) 
    else:
        Mut_Mol_Sanitized = Chem.SanitizeMol(Mut_Mol, catchErrors=True) 

    if len(Chem.GetMolFrags(Mut_Mol)) != 1:
        if Verbose:
            print('Fragmented result, trying new mutation')
        Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

        if showdiff:
            view_difference(StartingMoleculeUnedited, Mut_Mol)
    
    elif Mut_Mol_Sanitized != rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
        if Verbose:
            print('Validity Check Failed')
        Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

    else: 
        if Verbose:
            print('Validity Check Passed')
        if showdiff:
            view_difference(StartingMoleculeUnedited, Mut_Mol)
    
    return Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES

def Napthalenate(StartingMolecule, Napthalene, InsertStyle, showdiff=True, Verbose=False):
    """
    Function to add a fragment from a selected list of fragments to starting molecule.
    
    Args:
    - StartingMolecule: SMILES String of Starting Molecule
    - Fragments: List of fragments 
    - showdiff
    - InsertStyle: Whether to add fragment to edge or within molecule
    """
    """
    Steps:
    2. Determine whether fragment is going to be inserted within or appended to end of molecule
    
    3. If inserting fragment within molecule:
        - Combine starting molecule and fragment into single disconnected Mol object 
        - Split atom indexes of starting molecule and fragment and save in different objects
        - Check number of bonds each atom has in starting molecule and fragment, exclude any atoms that don't have 
        exactly two bonds
        - Get the atom neighbors of selected atom 
        - Remove one of selected atom's bonds with its neighbors, select which randomly but store which bond is severed 
        - Calculate terminal atoms of fragment (atoms with only one bond, will be at least two, return total number of
        fragments, store terminal atom indexes in a list)
        - Randomly select two terminal atoms from terminal atoms list of fragment
        - Create a new bond between each of the neighbors of the target atom and each of the terminal atoms on the 
        fragment, specify which type of bond, with highest likelihood that it is a single bond
        """
    
    StartingMoleculeUnedited = StartingMolecule

    try:
        #Check if molecule has rings already
        if isinstance(StartingMoleculeUnedited, str):
            StartingMoleculeUnedited = Chem.MolFromSmiles(StartingMolecule)
        if isinstance(Napthalene, str):
            Napthalene = Chem.MolFromSmiles(Napthalene)

        SMRings = StartingMoleculeUnedited.GetRingInfo() #Starting molecule rings

        if SMRings.NumRings() > 0:
            if Verbose:
                print('Starting molecule has rings, abandoning mutations')
            Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
            sys.exit()

        else:
            # Add fragment to Mol object of starting molecule
            StartingMolecule = Chem.RWMol(StartingMoleculeUnedited)
            StartingMolecule.InsertMol(Napthalene)

            # Get indexes of starting molecule and fragment, store them in separate objects 
            frags = Chem.GetMolFrags(StartingMolecule)
            StartMolIdxs = frags[0]
            FragIdxs = frags[1]

            OneBondAtomsMolecule = []
            TwoBondAtomsMolecule = []
            AromaticAtomsMolecule = []
            OneBondAtomsFragment = []
            TwoBondAtomsFragment = []

            # Getting atoms in starting molecule with different amount of bonds, storing indexes in list
            for index in StartMolIdxs:
                Atom = StartingMolecule.GetAtomWithIdx(int(index))
                if Atom.GetIsAromatic() and len(Atom.GetBonds()) == 2:
                    AromaticAtomsMolecule.append(index)
                elif len(Atom.GetBonds()) == 2:
                    TwoBondAtomsMolecule.append(index)
                elif len(Atom.GetBonds()) == 1:
                    OneBondAtomsMolecule.append(index)
                else:
                    continue

            # Getting atoms in fragment with varying bonds, storing indexes in list
            for index in FragIdxs:
                Atom = StartingMolecule.GetAtomWithIdx(int(index))
                if len(Atom.GetBonds()) == 1:
                    OneBondAtomsFragment.append(index)
                elif len(Atom.GetBonds()) == 2:
                    TwoBondAtomsFragment.append(index)

            ### INSERTING FRAGMENT AT RANDOM POINT WITHIN STARTING MOLECULE
            if InsertStyle == 'Within':
                # Select random two bonded atom, not including aromatic
                AtomRmv = rnd(TwoBondAtomsMolecule)

                # Get atom neighbor indexes, remove bonds between selected atom and neighbors 
                neighbors = [x.GetIdx() for x in StartingMolecule.GetAtomWithIdx(AtomRmv).GetNeighbors()]

                # Randomly choose which bond of target atom to sever
                SeverIdx = rnd([0,1])

                StartingMolecule.RemoveBond(neighbors[SeverIdx], AtomRmv)

                #Create a bond between the fragment and the now segmented fragments of the starting molecule
                #For situation where bond before target atom is severed
                if SeverIdx == 0:

                    StartingMolecule.AddBond(TwoBondAtomsFragment[0], AtomRmv - 1, Chem.BondType.SINGLE)
                    StartingMolecule.AddBond(TwoBondAtomsFragment[1], AtomRmv, Chem.BondType.SINGLE)

                    Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

                #For situation where bond after target atom is severed
                elif SeverIdx != 0:
                    StartingMolecule.AddBond(TwoBondAtomsFragment[0], AtomRmv + 1, Chem.BondType.SINGLE) 
                    StartingMolecule.AddBond(TwoBondAtomsFragment[1], AtomRmv, Chem.BondType.SINGLE)

                    Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

            ### INSERTING FRAGMENT AT END OF STARTING MOLECULE

            elif InsertStyle == 'Edge':
                # Choose whether fragment will be added to an aromatic or non-aromatic bond
                FragAdd = rnd(['Aromatic', 'Non-Aromatic'])

                if len(OneBondAtomsMolecule) == 0:
                    if Verbose:
                        print('No one-bonded terminal atoms in starting molecule, returning empty object')
                    Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

                elif FragAdd == 'Non-Aromatic' or len(AromaticAtomsMolecule) == 0: 
                    #Randomly choose atom from fragment (including aromatic)
                    FragAtom = rnd(FragIdxs)

                    #Select random terminal from starting molecule
                    AtomRmv = rnd(OneBondAtomsMolecule)

                    #Attach fragment to end of molecule
                    StartingMolecule.AddBond(FragAtom, AtomRmv, Chem.BondType.SINGLE)

                    Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

            else:
                if Verbose:
                    print('Edge case, returning empty objects')
                Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
    except Exception as E:
        if Verbose:
            print(E)
            print('Index error, starting molecule probably too short, trying different mutation')
        Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
      
    return Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES, StartingMoleculeUnedited

def Esterify(StartingMolecule, InsertStyle, showdiff=True, Verbose=False):
    StartingMoleculeUnedited = StartingMolecule
    EsterSMILES = 'CC(=O)OC'
    EsterMol = Chem.MolFromSmiles(EsterSMILES)
    
    if isinstance(StartingMoleculeUnedited, str):
        StartingMoleculeUnedited = Chem.MolFromSmiles(StartingMolecule)

    try:
        # Add fragment to Mol object of starting molecule
        StartingMolecule = Chem.RWMol(StartingMoleculeUnedited)
        StartingMolecule.InsertMol(EsterMol)

        # Get indexes of starting molecule and fragment, store them in separate objects 
        frags = Chem.GetMolFrags(StartingMolecule)
        StartMolIdxs = frags[0]
        FragIdxs = frags[1]

        OneBondAtomsMolecule = []
        TwoBondAtomsMolecule = []
        AromaticAtomsMolecule = []
        OneBondAtomsFragment = []
        TwoBondAtomsFragment = []

        # Getting atoms in starting molecule with different amount of bonds, storing indexes in list
        for index in StartMolIdxs:
            Atom = StartingMolecule.GetAtomWithIdx(int(index))
            if Atom.GetIsAromatic() and len(Atom.GetBonds()) == 2:
                AromaticAtomsMolecule.append(index)
            elif len(Atom.GetBonds()) == 2:
                TwoBondAtomsMolecule.append(index)
            elif len(Atom.GetBonds()) == 1:
                OneBondAtomsMolecule.append(index)
            else:
                continue

        # Getting atoms in fragment with varying bonds, storing indexes in listv
        for index in FragIdxs:
            Atom = StartingMolecule.GetAtomWithIdx(int(index))
            if len(Atom.GetBonds()) == 1:
                if Atom.GetAtomicNum() == 6:
                    OneBondAtomsFragment.append(index)
            elif len(Atom.GetBonds()) == 2:
                TwoBondAtomsFragment.append(index)
        
        ### INSERTING FRAGMENT AT RANDOM POINT WITHIN STARTING MOLECULE
        if InsertStyle == 'Within':

            if len(OneBondAtomsMolecule) == 0:
                if Verbose:
                    print('No one-bonded terminal atoms in starting molecule, returning empty object')
                Mut_Mol = None
                Mut_Mol_Sanitized = None
                MutMolSMILES = None
            
            else:
                # Select random two bonded atom, not including aromatic
                AtomRmv = rnd(TwoBondAtomsMolecule)

                # Get atom neighbor indexes, remove bonds between selected atom and neighbors 
                neighbors = [x.GetIdx() for x in StartingMolecule.GetAtomWithIdx(AtomRmv).GetNeighbors()]

                # Randomly choose which bond of target atom to sever
                SeverIdx = rnd([0,1])

                StartingMolecule.RemoveBond(neighbors[SeverIdx], AtomRmv)

                #Create a bond between the fragment and the now segmented fragments of the starting molecule
                #For situation where bond before target atom is severed
                if SeverIdx == 0:

                    StartingMolecule.AddBond(OneBondAtomsFragment[0], AtomRmv - 1, Chem.BondType.SINGLE)
                    StartingMolecule.AddBond(OneBondAtomsFragment[1], AtomRmv, Chem.BondType.SINGLE)

                    Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

                #For situation where bond after target atom is severed
                elif SeverIdx != 0:
                    StartingMolecule.AddBond(OneBondAtomsFragment[0], AtomRmv + 1, Chem.BondType.SINGLE) 
                    StartingMolecule.AddBond(OneBondAtomsFragment[1], AtomRmv, Chem.BondType.SINGLE)

                    Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

        ### INSERTING FRAGMENT AT END OF STARTING MOLECULE

        elif InsertStyle == 'Edge':
            # Choose whether fragment will be added to an aromatic or non-aromatic bond
            FragAdd = rnd(['Aromatic', 'Non-Aromatic'])

            if len(OneBondAtomsMolecule) == 0:
                if Verbose:
                    print('No one-bonded terminal atoms in starting molecule, returning empty object')
                Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

            elif FragAdd == 'Non-Aromatic' or len(AromaticAtomsMolecule) == 0: 
                #Randomly choose atom from fragment (including aromatic)
                FragAtom = rnd(FragIdxs)

                #Select random terminal from starting molecule
                AtomRmv = rnd(OneBondAtomsMolecule)

                #Attach fragment to end of molecule
                StartingMolecule.AddBond(FragAtom, AtomRmv, Chem.BondType.SINGLE)

                Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

            elif len(AromaticAtomsMolecule) > 0:
                #Randomly select 2 bonded (non-branch base) aromatic atom 
                ArmtcAtom = rnd(AromaticAtomsMolecule)

                #Randomly select terminal atom from fragment
                FragAtom = rnd(OneBondAtomsFragment)

                #Add Bond
                StartingMolecule.AddBond(ArmtcAtom, FragAtom, Chem.BondType.SINGLE)

                Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)
                
        else:
            if Verbose:
                print('Edge case, returning empty objects')
            Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
    except Exception as E:
        if Verbose:
            print(E)
            print('Index error, starting molecule probably too short, trying different mutation')
        Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
      
    return Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES, StartingMoleculeUnedited

def Generate_Glycol_Chain(ChainLength):
    if ChainLength < 1:
        raise ValueError("Repeat count must be at least 1")
    
    # Start with 'COC' and join subsequent units by removing the leading 'C' from each unit
    return "COC" + "OC" * (ChainLength - 1)

def Glycolate(StartingMolecule, InsertStyle, showdiff=True, Verbose=False):
    StartingMoleculeUnedited = StartingMolecule
    GlycolSMILES = Generate_Glycol_Chain(rnd([1,2,3]))
    GlycolMol = Chem.MolFromSmiles(GlycolSMILES)

    if isinstance(StartingMoleculeUnedited, str):
        StartingMoleculeUnedited = Chem.MolFromSmiles(StartingMolecule)

    try:
        # Add fragment to Mol object of starting molecule
        StartingMolecule = Chem.RWMol(StartingMoleculeUnedited)
        StartingMolecule.InsertMol(GlycolMol)

        # Get indexes of starting molecule and fragment, store them in separate objects 
        frags = Chem.GetMolFrags(StartingMolecule)
        StartMolIdxs = frags[0]
        FragIdxs = frags[1]

        OneBondAtomsMolecule = []
        TwoBondAtomsMolecule = []
        AromaticAtomsMolecule = []
        OneBondAtomsFragment = []
        TwoBondAtomsFragment = []

        # Getting atoms in starting molecule with different amount of bonds, storing indexes in list
        for index in StartMolIdxs:
            Atom = StartingMolecule.GetAtomWithIdx(int(index))
            if Atom.GetIsAromatic() and len(Atom.GetBonds()) == 2:
                AromaticAtomsMolecule.append(index)
            elif len(Atom.GetBonds()) == 2:
                TwoBondAtomsMolecule.append(index)
            elif len(Atom.GetBonds()) == 1:
                OneBondAtomsMolecule.append(index)
            else:
                continue

        # Getting atoms in fragment with varying bonds, storing indexes in listv
        for index in FragIdxs:
            Atom = StartingMolecule.GetAtomWithIdx(int(index))
            if len(Atom.GetBonds()) == 1:
                if Atom.GetAtomicNum() == 6:
                    OneBondAtomsFragment.append(index)
            elif len(Atom.GetBonds()) == 2:
                TwoBondAtomsFragment.append(index)
        
        ### INSERTING FRAGMENT AT RANDOM POINT WITHIN STARTING MOLECULE
        if InsertStyle == 'Within':

            if len(OneBondAtomsMolecule) == 0:
                if Verbose:
                    print('No one-bonded terminal atoms in starting molecule, returning empty object')
                Mut_Mol = None
                Mut_Mol_Sanitized = None
                MutMolSMILES = None
            
            else:
                # Select random two bonded atom, not including aromatic
                AtomRmv = rnd(TwoBondAtomsMolecule)

                # Get atom neighbor indexes, remove bonds between selected atom and neighbors 
                neighbors = [x.GetIdx() for x in StartingMolecule.GetAtomWithIdx(AtomRmv).GetNeighbors()]

                # Randomly choose which bond of target atom to sever
                SeverIdx = rnd([0,1])

                StartingMolecule.RemoveBond(neighbors[SeverIdx], AtomRmv)

                #Create a bond between the fragment and the now segmented fragments of the starting molecule
                #For situation where bond before target atom is severed
                if SeverIdx == 0:

                    StartingMolecule.AddBond(OneBondAtomsFragment[0], AtomRmv - 1, Chem.BondType.SINGLE)
                    StartingMolecule.AddBond(OneBondAtomsFragment[1], AtomRmv, Chem.BondType.SINGLE)

                    Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

                #For situation where bond after target atom is severed
                elif SeverIdx != 0:
                    StartingMolecule.AddBond(OneBondAtomsFragment[0], AtomRmv + 1, Chem.BondType.SINGLE) 
                    StartingMolecule.AddBond(OneBondAtomsFragment[1], AtomRmv, Chem.BondType.SINGLE)

                    Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

        ### INSERTING FRAGMENT AT END OF STARTING MOLECULE

        elif InsertStyle == 'Edge':
            # Choose whether fragment will be added to an aromatic or non-aromatic bond
            FragAdd = rnd(['Aromatic', 'Non-Aromatic'])

            if len(OneBondAtomsMolecule) == 0:
                if Verbose:
                    print('No one-bonded terminal atoms in starting molecule, returning empty object')
                Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

            elif FragAdd == 'Non-Aromatic' or len(AromaticAtomsMolecule) == 0: 
                #Randomly choose atom from fragment (including aromatic)
                FragAtom = rnd(FragIdxs)

                #Select random terminal from starting molecule
                AtomRmv = rnd(OneBondAtomsMolecule)

                #Attach fragment to end of molecule
                StartingMolecule.AddBond(FragAtom, AtomRmv, Chem.BondType.SINGLE)

                Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

            elif len(AromaticAtomsMolecule) > 0:
                #Randomly select 2 bonded (non-branch base) aromatic atom 
                ArmtcAtom = rnd(AromaticAtomsMolecule)

                #Randomly select terminal atom from fragment
                FragAtom = rnd(OneBondAtomsFragment)

                #Add Bond
                StartingMolecule.AddBond(ArmtcAtom, FragAtom, Chem.BondType.SINGLE)

                Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)
                
        else:
            if Verbose:
                print('Edge case, returning empty objects')
            Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
    except Exception as E:
        if Verbose:
            print(E)
            print('Index error, starting molecule probably too short, trying different mutation')
        Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
      
    return Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES, StartingMoleculeUnedited

Napthalenes = ['C12=CC=CC=C2C=CC=C1', 'C1CCCC2=CC=CC=C12', 'C1CCCC2CCCCC12']

Mols = ['CCCCCCCCCCCCCCCC', #Hexadecane
        'CC(C)CCCCCCCOC(=O)CCCCC(=O)OCCCCCCCC(C)C', #DIDA
        'CCCC(CCCCC(CCC)CCC)CCC', #4-9-di-n-propyldodecane
        'CCCCCCCC=C(CCCCCCCC)CCCCCCCC',  #9-n-octyl-8-heptadecene 
        'CC(CCCCCCCCCCCCC)C', #2-methylpentadecane 
        'CCCCC(CC)COC(=O)CCCCCCCCC(=O)OCC(CC)CCCC', # Bis2(ethylhexyl)sebacate
        'CC(C)CCCC(C)CCCC(C)CCCCC(C)CCCC(C)CCCC(C)C', #Squalane
        'CCCCCCCC(CCCCCCC)CCCCCC '] #8-n-hexylpentadecane 

for mol in Mols:
    for i in range(1, 3):
        Result = Glycolate(mol, InsertStyle = rnd(['Within', 'Edge']), showdiff=True, Verbose=True)
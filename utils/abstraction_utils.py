import sys
sys.path.append('../..')

import gzip
import json 
import logging
import networkx as nx

from rdkit import Chem
from rdchiral.main import rdchiralRun
from rdchiral.initialization import rdchiralReaction, rdchiralReactants


from utils.utils import print_if_verbose
from utils.chem_utils import get_atom_maps, get_changed_bond_atoms
from utils.electronegs import electronegs
from utils.abstraction_smarts import preprocess_smarts, abstraction_smarts

PREPROCESS_RXNS = [rdchiralReaction(smarts) for smarts in preprocess_smarts]
ABS_RXNS = [rdchiralReaction(smarts) for smarts in abstraction_smarts]
POSCHARGED_ATOM_PATTERN = Chem.MolFromSmarts("[+1,+2!h1,+3!h1!h2,+4!h1!h2!h3;!h0;!0*;!a;!$([*]~[-1,-2,-3,-4])]") 
NEGCHARGED_ATOM_PATTERN = Chem.MolFromSmarts("[-1,-2,-3,-4;!0*;!a;!$([*]~[+1,+2,+3,+4])]")
# modified to exclude aromatic atoms, any atoms without isotope tags
# +n charge, not aromatic, at least n hydrogen, and not linked to negative charged atom; or -n charge and not linked to
# positive atom
C_electroneg = electronegs[6]
MAX_ITERATIONS = 25

verbose = False

def set_verbose(v):
    global verbose
    verbose = v
        
def neutralize_abstracted_atoms(mol):
    """
    modified from http://www.rdkit.org/docs/Cookbook.html
    neutralizing a specific atom
    """
    pos_at_matches = mol.GetSubstructMatches(POSCHARGED_ATOM_PATTERN)
    neg_at_matches = mol.GetSubstructMatches(NEGCHARGED_ATOM_PATTERN)
    at_matches_list = [y[0] for y in pos_at_matches]+[y[0] for y in neg_at_matches]
    
    for atom_idx in at_matches_list:
        atom = mol.GetAtomWithIdx(atom_idx)
        chg = atom.GetFormalCharge()
        hcount = atom.GetTotalNumHs()
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(hcount - chg)
        atom.UpdatePropertyCache()


def abstract_reactant(reactant, target):

    """
    Abstract the reactant with the target as a reference
    
    Args: 
        reactant: reactant SMILES, assumes single component
        target: target SMILES
    """
    def _map_unmapped_atoms(mol):

        nonlocal next_map
            
        for atom in mol.GetAtoms():
            if not atom.GetAtomMapNum():
                atom.SetAtomMapNum(next_map)
                next_map += 1

    def _tag_core_atom(atom):

        atom_map = atom.GetAtomMapNum()

        if atom_map in new_bond_atoms and atom.GetAtomicNum()!=6:
            if atom.GetTotalNumHs() or atom.GetFormalCharge()!=0: 
                atom.SetIsotope(1)

        if atom_map in removed_bond_atoms: 
            atom.SetIsotope(100)

    def _tag_leaving_atom(atom):

        if C_electroneg < electronegs[atom.GetAtomicNum()]:
            atom.SetIsotope(201) # negative charge on the leaving atom (C+)
        elif C_electroneg > electronegs[atom.GetAtomicNum()]:
            atom.SetIsotope(202) # positive charge on the leaving atom (C-)
        else: atom.SetIsotope(200)

    target_maps = get_atom_maps(target)
    reactant_maps = get_atom_maps(reactant)
    # If no atom contribution, return None
    if not target_maps & reactant_maps:
        return None

    next_map = max(target_maps | reactant_maps) +1

    print_if_verbose(f"target_maps: {target_maps}", verbose)
    print_if_verbose(f"reactant_maps: {reactant_maps}", verbose)
    target_mol = Chem.MolFromSmiles(target)
    reactant_mol = Chem.MolFromSmiles(reactant)
    _map_unmapped_atoms(reactant_mol)

    new_bond_atoms, removed_bond_atoms = get_changed_bond_atoms(reactant_mol, target_mol)
    print_if_verbose(f"new_bond_atoms: {new_bond_atoms}", verbose)
    print_if_verbose(f"removed_bond_atoms: {removed_bond_atoms}", verbose)

    for atom in reactant_mol.GetAtoms():

        if atom.GetAtomMapNum() in target_maps:
            _tag_core_atom(atom)
        else:
            _tag_leaving_atom(atom)
            
    tagged_reactant = Chem.MolToSmiles(reactant_mol)

    print_if_verbose(f"tagged_reactant: {tagged_reactant}", verbose)

    return abstract_with_smarts(tagged_reactant)


def split_and_abstract(smiles):

    """
    Split SMILES into separate reactants and run abstraction on each.
    
    Args:
        smiles: SMILES string representing the reactants
    
    Returns:
        SMILES string with abstracted groups
    """
    
    fragments = smiles.split(".")
    new_fragments = [
        abstract_with_smarts(fragment)
        for i, fragment in enumerate(fragments)
    ]

    new_fragments = sorted([x for x in new_fragments if x])

    return '.'.join(new_fragments)


def abstract_with_smarts(smiles):
    """
    Apply abstraction SMARTS to the given SMILES.
    
    Args:
        smiles: SMILES string of the reactant
    
    Returns:
        SMILES string after applying abstraction SMARTS
    """
    new_map = max(get_atom_maps(smiles)) + 1000
    step = "preprocess"
    for i in range(MAX_ITERATIONS):
        smiles, is_done = apply_abstraction_smarts_to_smiles(smiles, new_map, step=step)
        print_if_verbose(f"{i}, {smiles}, {is_done}", verbose)
        if is_done:
            if step =="preprocess": # Move onto abstraction
                step = "abstract"
                continue
            else: break

    return finalize_reactant(smiles)


def apply_abstraction_smarts_to_smiles(smiles, new_map, step="preprocess"):
    """
    Process the reactant with SMARTS reactions and update the reactant accordingly.

    Args:
        smiles: Reactant SMILES string
        new_map: Starting atom map number
        step: Current processing step ("preprocess" or "abstract")
    
    Returns:
        Updated SMILES string and a boolean indicating if processing is done
    """
    reactant = rdchiralReactants(smiles, custom_reactant_mapping=True)
    if step == "preprocess": RXNS = PREPROCESS_RXNS
    else: RXNS = ABS_RXNS

    for rxn in RXNS:
        try:
            outcomes = rdchiralRun(rxn, reactant, keep_mapnums=True)
            if not outcomes:
                continue
            print_if_verbose(f"Reaction: {rxn.reaction_smarts}", verbose)
            outcomes = sorted(outcomes, key=len, reverse=True)
            outcome = outcomes[0].replace(':900]', f':{new_map}]')

            if "." in outcome:
                return split_and_abstract(outcome), True

            new_map += 1
            return outcome, False

        except Exception as e:
            print(f"Error with SMARTS {rxn.reaction_smarts}: {e}")

    return smiles, True


def finalize_reactant(smiles):
    """
    Finalize the reactant by neutralizing abstracted atoms and returning the SMILES string.
    
    Args:
        smiles: SMILES string of the reactant
    
    Returns:
        SMILES string of the finalized reactant
    """
    reactant_mol = Chem.MolFromSmiles(smiles)
    for atom in reactant_mol.GetAtoms():
        if atom.GetIsotope() in {10, 100}:
            atom.SetIsotope(0)

    neutralize_abstracted_atoms(reactant_mol)
    return Chem.MolToSmiles(reactant_mol)


def abstract_rsmi(rsmi):

    reactants, _ , product = rsmi.split('>')

    abs_reactants = [
        abstract_reactant(reactant, product)  
        for reactant in reactants.split('.')
    ]

    abs_reactants = sorted([x for x in abs_reactants if x])

    return '.'.join(abs_reactants) + '>>' + product



import sys
import os
import re
import collections
import itertools
import functools
import typing
import warnings
import requests
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski, rdqueries, Descriptors, Draw
from rdkit.Chem.Descriptors import ExactMolWt, qed


VERBOSE = True

isCAtomsQuerier = rdqueries.AtomNumEqualsQueryAtom(6)

def has_mapping(smiles_in):
    pattern = r"\:[0-9]+\]"
    match = re.search(pattern, smiles_in)
    return bool(match)

def has_isotopes(smiles_in):

    pattern = r"\[([1-9]+)|([1-9]+\*)"
    match = re.search(pattern, smiles_in)
    return bool(match)

def get_atom_maps(smiles_in):
    """Returns a set of all atom maps in input SMILES string"""
    if not smiles_in:
        return set()
    return set([int(x) for x in re.findall(r":(\d+)", smiles_in)])

def get_num_heavy_atoms(smiles):
    """Returns the number of heavy atoms (non-hydrogen) in a SMILES string"""
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumHeavyAtoms() if mol else 0

def get_largest_chemical(smiles):

    """Get the molecule with the highest MW in smiles (separated by '.')"""
    
    MW = 0
    large_chemical = ''
    
    for smi in smiles.split("."):
        mol = Chem.MolFromSmiles(smi)
        if Descriptors.ExactMolWt(mol) > MW:
            large_chemical = smi
            MW = Chem.Descriptors.ExactMolWt(mol)
            
    return large_chemical


def check_mol(mol):
    smi = Chem.MolToSmiles(mol, canonical=canonical)
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, f"Failed to convert {smi} back to a mol object"


CHARGED_ATOM_PATTERN = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
# i.e. +1 charge, at least one hydrogen, and not linked to negative charged atom; or -1 charge and not linked to
# positive atom

def try_neutralize_smi(smi, canonical=True, isomericSmiles=True, log=None):

    """
    Modified from https://github.com/coleygroup/react-splits/blob/main/react_splits/chem_utils.py
    Changed to only return neutralized smiles if valid (i.e., can be converted to a mol object)
    """
    mol = Chem.MolFromSmiles(smi)
    try:
        mol = neutralize_atoms(mol)
        check_mol(mol)

    except Exception as ex:
        err_str = f"Failed to neutralize {smi}"
        #warnings.warn(err_str)
        # skipping for now, can check out a few of them and see
    else: 
        smi = Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=isomericSmiles)
    return smi


def neutralize_atoms(mol):
    """
    Modified from https://github.com/coleygroup/react-splits/blob/main/react_splits/chem_utils.py
    Originally from http://www.rdkit.org/docs/Cookbook.html
    Changed so that returns a RWCopy
    """
    mol = Chem.RWMol(mol)
    at_matches = mol.GetSubstructMatches(CHARGED_ATOM_PATTERN)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol


def remove_isotope_info_from_mol_in_place(mol):
    """
    Modified from https://github.com/coleygroup/react-splits/blob/main/react_splits/chem_utils.py
    Originally adapted from https://www.rdkit.org/docs/Cookbook.html#isomeric-smiles-without-isotopes
    see limitations at link about needing to canonicalize _after_.
    """
    atom_data = [(atom, atom.GetIsotope()) for atom in mol.GetAtoms()]
    for atom, isotope in atom_data:
       if isotope:
           atom.SetIsotope(0)
    return mol

def canonicalize_route(route):
    return tuple(sorted(set([canonicalize_rsmi(rsmi) for rsmi in route])))

def canonicalize_rsmi(rsmi, include_reagents=False, **otherargs):

    try: 
        reacts, reag, prods = rsmi.split('>')

        canon_reacts = sorted([canonicalize(r, **otherargs) for r in reacts.split('.')])
        canon_prods = sorted([canonicalize(p, **otherargs) for p in prods.split('.')])

        if include_reagents:
            reag = sorted(reag.split('.'))
            return '.'.join(canon_reacts)+'>'+ '.'.join(reag) + '>' + '.'.join(canon_prods)
        else:
            return '.'.join(canon_reacts)+'>>'+ '.'.join(canon_prods)

    except Exception as e:
        err_str = f"Failed to canonicalize {rsmi}"
        #warnings.warn(err_str) 
        return rsmi



def canonicalize(smiles, remove_atm_mapping=True, remove_isotope_info=False, isomericSmiles=True, **otherargs):
    
    if not remove_isotope_info and not isomericSmiles:
        smiles = smiles.replace('@', '').replace('/', '').replace('\\', '')
        isomericSmiles = True
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None

    if remove_atm_mapping and mol is not None:
        [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]

    if remove_isotope_info and mol is not None:
        mol = remove_isotope_info_from_mol_in_place(mol)

    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomericSmiles, **otherargs)



def get_changed_bonds(reactants_mol, products_mol):
    
    """Include new or lost bonds only"""

    conserved_maps = [a.GetAtomMapNum() for a in products_mol.GetAtoms() if a.HasProp('molAtomMapNumber')]
    #print(conserved_maps)
    new_bonds = set() # keep track of formed bonds
    removed_bonds = set() # keep track of removed bonds
    
    # Look at changed bonds
    bonds_prev = {}
    for bond in reactants_mol.GetBonds():
        nums = sorted(
            [bond.GetBeginAtom().GetAtomMapNum(),
             bond.GetEndAtom().GetAtomMapNum()])
        
        if (nums[0] not in conserved_maps) and (nums[1] not in conserved_maps): continue
        bonds_prev['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()
        
    # print("bonds_prev", bonds_prev)
    bonds_new = {}
    for bond in products_mol.GetBonds():
        nums = sorted(
            [bond.GetBeginAtom().GetAtomMapNum(),
             bond.GetEndAtom().GetAtomMapNum()])
        bonds_new['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()
    # print("bonds_new", bonds_new)
    for bond in bonds_prev:
        if bond not in bonds_new: # 
            removed_bonds.add((int(bond.split('~')[0]), int(bond.split('~')[1]))) # lost bond
    for bond in bonds_new:
        if bond not in bonds_prev:
            new_bonds.add((int(bond.split('~')[0]), int(bond.split('~')[1])))  # new bond
    # print(bond_changes)
    return new_bonds, removed_bonds

def get_changed_bond_atoms(reactants_mol, products_mol):

    new_bonds, removed_bonds = get_changed_bonds(reactants_mol, products_mol)
    new_bond_atoms = set([item for t in new_bonds for item in t if item])
    removed_bond_atoms = set([item for t in removed_bonds for item in t if item])

    return new_bond_atoms, removed_bond_atoms

def get_changed_bonds_rsmi(rsmi):
    reacts, reag, prods = rsmi.split('>')
    reactants_mol = Chem.MolFromSmiles(reacts)
    products_mol = Chem.MolFromSmiles(prods)
    return get_changed_bonds(reactants_mol, products_mol)


def display_smarts(smarts):
    print(smarts)
    img = Draw.ReactionToImage(AllChem.ReactionFromSmarts(smarts), subImgSize=(300, 300))
    display(img)

def display_img(smi, size=(400,400), useSmiles=True):
        
    print("Displaying image", smi)
    if '>' in smi:
            img = Draw.ReactionToImage(
                    AllChem.ReactionFromSmarts(smi, useSmiles=useSmiles),
                    highlightByReactant=True)
    else:
            img = Draw.MolToImage(Chem.MolFromSmiles(smi), size=size)
    display(img)


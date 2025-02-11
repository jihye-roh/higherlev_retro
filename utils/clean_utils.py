import re
from rdkit import Chem
from rdkit import RDLogger  

from utils.chem_utils import canonicalize, try_neutralize_smi, get_num_heavy_atoms, has_mapping, get_atom_maps

RDLogger.DisableLog('rdApp.*') 

VERBOSE = False
MIN_NUM_PRODUCT_HEAVY_ATOMS = 5
MAX_NUM_PRODUCT_ATOMS_WO_SOURCE = 5

def set_verbose(verbose):
    global VERBOSE
    VERBOSE = verbose

def set_min_num_product_heavy_atoms(min_num_product_heavy_atoms):
    global MIN_NUM_PRODUCT_HEAVY_ATOMS
    MIN_NUM_PRODUCT_HEAVY_ATOMS = min_num_product_heavy_atoms

def set_max_num_product_atoms_wo_source(max_num_product_atoms_wo_source):
    global MAX_NUM_PRODUCT_ATOMS_WO_SOURCE
    MAX_NUM_PRODUCT_ATOMS_WO_SOURCE = max_num_product_atoms_wo_source


def reassign_atom_maps(smiles_in):
    
    """Reassigns atom mapping in canonicalized order for an input reaction SMILES string"""

    reacts, reag, prods = smiles_in.split('>')
    
    prods_mol = Chem.MolFromSmiles(prods)
    reacts_mol = Chem.MolFromSmiles(reacts)
    
    curr_map = 1
    maps_dict = {}
    
    for a in prods_mol.GetAtoms():
        maps_dict[a.GetAtomMapNum()] = curr_map
        a.SetAtomMapNum(curr_map)
        curr_map += 1
    
    for a in reacts_mol.GetAtoms():
        if a.GetAtomMapNum() in maps_dict:
            a.SetAtomMapNum(maps_dict[a.GetAtomMapNum()])
        else:
            a.SetAtomMapNum(curr_map)
            curr_map += 1
    
    return Chem.MolToSmiles(reacts_mol)+'>'+reag+'>'+Chem.MolToSmiles(prods_mol)


def get_canonicalization_dict(smiles_list):

    """
    Returns a dictionary of canonicalized smiles (with atom mapping) to 
    neutralized, canonicalized non-isomeric smiles (without atom mapping, no stereochemistry) 
    for a list of SMILES strings
    """

    return {
        canonicalize(smi, remove_atm_mapping=False): \
            canonicalize(try_neutralize_smi(smi), isomericSmiles=False)
        for smi in smiles_list
    }

    
def has_passed_product_filter(product):
    
    """Returns True if the product has at least MIN_NUM_PRODUCT_HEAVY_ATOMS, False otherwise"""

    if MIN_NUM_PRODUCT_HEAVY_ATOMS <= 1: return True
    
    return get_num_heavy_atoms(product) >= MIN_NUM_PRODUCT_HEAVY_ATOMS



def rearrange_common_structures(reactants_smi, products_smi, reagents_smi):


    """
    Takes in a list of reactant smiles and a list of product smiles
    Rearranges common canonicalized structure (after neutralizing & removing isomerism) 
        moved from reactants/products to reagents
    """

    # value: canonicalized smiles with atom mapping and isomerism removed
    reactants_dict = get_canonicalization_dict(reactants_smi)
    products_dict = get_canonicalization_dict(products_smi)
    
    # common if the structures are the same after neutralizing & removing isomerism
    can_common = set(reactants_dict.values()) & set(products_dict.values())

    common = {key for key, value in products_dict.items() if value in can_common} | \
            {key for key, value in reactants_dict.items() if value in can_common} 

    reactants_smi = [smi for smi in reactants_dict.keys() if smi not in common]
    products_smi = [smi for smi in products_dict.keys() if smi not in common]
    reagents_smi += [canonicalize(smi) for smi in common]

    if VERBOSE: 
        print("common", common)
        print("reactants_dict", reactants_dict)
        print("products_dict", products_dict)
        print("reactants_smi", reactants_smi)
        print("products_smi", products_smi)
        print("reagents", reagents_smi)

    return reactants_smi, products_smi, reagents_smi



def filter_and_recategorize_components(reactants_smi, products_smi, reagents_smi):

    """"
    Filters and recategorizes reactants, reagents, and products based on atom mapping
    Each product species must have at least one common atom mapping with the reactants
    Each reactant species must have at least one common atom mapping with the products
        if not, the species is moved to reagents
    """

    reactants_smi, products_smi, reagents_smi = \
        rearrange_common_structures(reactants_smi, products_smi, reagents_smi)

     # Filter products based on MIN_NUM_PRODUCT_HEAVY_ATOMS
    products_smi  = [smi for smi in products_smi if \
        has_mapping(smi) and \
            has_passed_product_filter(smi)]
    
    if VERBOSE: print("filtered_products", products_smi)   

    product_mapping = set()
    reactant_mapping = set()
    for smi in products_smi: product_mapping |= get_atom_maps(smi)
    for smi in reactants_smi: reactant_mapping |= get_atom_maps(smi)
    matched_mapping_set = product_mapping & reactant_mapping
    
    if VERBOSE: 
        print("product_mapping", product_mapping)
        print("reactant_mapping", reactant_mapping)
        print("matched_mapping_set", matched_mapping_set)

    final_products_smi = [
        product for product in products_smi if get_atom_maps(product) & matched_mapping_set
    ]
    # move unmapped reactants to reagents
    final_reactants_smi = []
    for reactant in reactants_smi:
        if get_atom_maps(reactant) & matched_mapping_set: final_reactants_smi.append(reactant)
        else: reagents_smi.append(canonicalize(reactant))

    return sorted(final_reactants_smi), sorted(final_products_smi), sorted(reagents_smi)


def clean_smiles(smiles_in):

    """
    Takes in an atom mapped reaction SMILES string and returns a new SMILES of the reaction 
    
    With the following cleaning steps:  
        0) Return original SMILES if no atom mapping exists
        1) Reagents moved to reactants if includes atom mapping
        2) If the number of atoms in the product without a source is greater than MAX_NUM_PRODUCT_ATOMS_WO_SOURCE,
            return canonicalized smiles (with atom mapping removed)
        3) Chemicals in both reactants and products (non_isomeric, with same atom mapping) moved to reagents 
        4) Product(s) removed if 
            4-1) it does not contain any atom mapping
            4-2) the number of heavy atoms is less than MIN_NUM_PRODUCT_HEAVY_ATOMS
        5) Reactants without atom contribution to products moved to reagents

    Then, the resulting SMILES is re-mapped to start from 1
    """
    
    # 0) If no atom mapping exists, returns original (unmapped) smiles
    if not has_mapping(smiles_in): return smiles_in
    
    # 1) move reagents to reactants if it has at least one atom mapping
    reactants, reagents, products = smiles_in.split('>')
    reactants_smi = reactants.split('.')
    products_smi = products.split('.')
    reagents_smi = []

    if reagents:
        for smi in reagents.split('.'): 
            if has_mapping(smi): 
                reactants_smi.append(smi)
            else: 
                reagents_smi.append(smi)

    joined_reactants = '.'.join(reactants_smi)
    joined_reagents = '.'.join(reagents_smi)
        
    if VERBOSE: print("reagents", reagents_smi)
    
    # 2) Filter reactions based on MAX_NUM_PRODUCT_ATOMS_WO_SOURCE
    reactant_maps = get_atom_maps(joined_reactants)
    product_maps = get_atom_maps(products)

    if len(product_maps-reactant_maps) > MAX_NUM_PRODUCT_ATOMS_WO_SOURCE:
        return canonicalize(joined_reactants)+'>'+joined_reagents+'>'+canonicalize(products)


    # 3), 4), and 5)
    final_reactants_smi, final_products_smi, reagents_smi = \
        filter_and_recategorize_components(reactants_smi, products_smi, reagents_smi)
    
    joined_smiles = \
        '.'.join(final_reactants_smi)+'>'+ '.'.join(reagents_smi) +'>' + '.'.join(final_products_smi)

    return reassign_atom_maps(joined_smiles)



def separate_and_clean_smiles(smiles_in):
    
    """
    Takes in an atom mapped reaction SMILES string, separates the reaction into single product reactions 
    Returns a list of cleaned SMILES strings (valid SMILES only)
    """
    cleaned_smiles = []

    try: 

        reactants, reagents, products = smiles_in.split('>')
        
        for p in products.split('.'):
            # Only consider products with at least MIN_NUM_PRODUCT_HEAVY_ATOMS (atom mapped)
            if get_num_heavy_atoms(p) < MIN_NUM_PRODUCT_HEAVY_ATOMS: continue 

            single_prod_smi = reactants+'>'+reagents+'>'+p
            cleaned_smi = clean_smiles(single_prod_smi)
            
            # Check if the cleaned SMILES is valid
            if has_mapping(cleaned_smi) and \
                not cleaned_smi.startswith('>') and not cleaned_smi.endswith('>'): 
                    cleaned_smiles.append(cleaned_smi)
    
    except Exception as e:
        if VERBOSE: print(f"Error {e} for smiles {smiles_in}")
    
    return cleaned_smiles
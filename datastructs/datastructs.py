import sys
sys.path.append('..')

from collections import defaultdict
from rdkit import Chem
from utils.chem_utils import canonicalize, canonicalize_rsmi

class NestedDefaultDict(defaultdict):
    def __init__(self):
        super().__init__(NestedDefaultDict._default_factory)
    
    @staticmethod
    def _default_factory():
        return defaultdict(list)
    
    def __repr__(self):
        def convert_to_dict(d):
            if isinstance(d, defaultdict):
                d = {k: convert_to_dict(v) for k, v in d.items()}
            return d
        return str(convert_to_dict(self))

class Molecule:

    def __init__(self, smiles):

        self.smiles =canonicalize(smiles)
        self.mol = Chem.MolFromSmiles(smiles)

    def add_tagged_smiles(self, tagged_smiles):
        self.tagged_smiles = tagged_smiles

    def add_smiles_as_reactant(self, smiles_as_reactant):
        self.smiles_as_reactant = smiles_as_reactant
    
    def add_smiles_as_product(self, smiles_as_product):
        self.smiles_as_product = smiles_as_product
    
    def add_map_to_tag(self, map_to_tag):
        self.map_to_tag = map_to_tag
    

class Reaction:
    
    def __init__(self, reaction_smiles, can_reaction_smiles=None):

        self.reaction_smiles = reaction_smiles
        self.can_reaction_smiles = can_reaction_smiles if \
            can_reaction_smiles else canonicalize_rsmi(reaction_smiles)
import os
import sys
import copy
import datetime

sys.path.append('..')

from rdkit import Chem, Geometry
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D


import networkx as nx
from utils.chem_utils import get_largest_chemical, canonicalize, canonicalize_rsmi, has_mapping, get_atom_maps
from datastructs.datastructs import Molecule, Reaction


class SynTree:

    """A Tree for a single synthesis pathway, with nodes as molecules and edges as reactions"""

    def __init__(self, route = None):

        self.route = route
        self.tree = nx.DiGraph()
        self.node_counter = {}
        self.product_set = set()
        self.can_route = []
        self.leaf_tags = []

        self.initialze_tree()
        self.root = next(iter(nx.topological_sort(self.tree)))
        self.root_molecule = self.tree.nodes[self.root]['molecule']
        self.depth = nx.dag_longest_path_length(self.tree)
        #max(nx.shortest_path_length(self.tree, self.root).values())

    def initialze_tree(self):

        """
        Initialize the tree with the given routes
        Nodes represent molecules and edges represent reactions
        Nodes: (canonicalized smiles, index)
        """

        for reaction in self.route:
            self.add_reaction_to_tree(reaction)

        self.add_subtrees()
        self.can_route = tuple(sorted(set(self.can_route)))


    def add_reaction_to_tree(self, reaction):
            
        """
        Update the tree with a new reaction
        """
        can_reaction_smiles = canonicalize_rsmi(reaction)
        self.can_route.append(can_reaction_smiles)

        reactants, reagents, products = reaction.split(">")

        # Add the largest product to tree
        largest_prod = get_largest_chemical(products)
        can_product = canonicalize(largest_prod)

        if can_product in self.product_set:
            raise TypeError('Cannot have multiple ways to make a product')
        
        product_node =  (can_product, 1)
        self.product_set.add(product_node)

        if not self.tree.has_node(product_node):
            self.tree.add_node(
                product_node, 
                molecule = Molecule(can_product)
            )

        self.tree.nodes[product_node]['molecule'].add_smiles_as_product(largest_prod)

        # Add reactant nodes and edges
        for reactant in reactants.split("."):

            can_reactant = canonicalize(reactant)

            self.node_counter[can_reactant] = \
                self.node_counter.get(can_reactant, 0) + 1
            
            new_idx = self.node_counter[can_reactant]
            reactant_node = (can_reactant, new_idx)

            if not self.tree.has_node(reactant_node):
                self.tree.add_node(
                    reactant_node, 
                    molecule = Molecule(can_reactant)
                )

            self.tree.nodes[reactant_node]['molecule'].add_smiles_as_reactant(reactant)

            self.tree.add_edge(
                product_node, 
                reactant_node, 
                reaction = Reaction(reaction, can_reaction_smiles)
            )

        # print(reaction)
        # for node in self.tree.nodes():
        #         print(node, self.tree.nodes[node]["molecule"].__dict__)

    def add_subtrees(self): 

        def _add_subtrees(node):

            """If duplicate nodes exist for the same molecule, add the subtrees for the duplicate nodes"""

            if node[1]>1:
                # add smiles_as_product
                dup_node = (node[0], 1)
                children = list(self.tree.successors(dup_node))

                if children:
                    self.tree.nodes[node]['molecule'].add_smiles_as_product(
                        self.tree.nodes[dup_node]['molecule'].smiles_as_product
                    )

                for child in children:

                    self.node_counter[child[0]] +=1
                    new_idx = self.node_counter[child[0]]
                    child_node = (child[0], new_idx)

                    self.tree.add_node(
                        child_node, 
                        molecule = copy.deepcopy(self.tree.nodes[child]['molecule'])
                    )
                    self.tree.add_edge(
                        node, 
                        child_node, 
                        reaction = copy.deepcopy(self.tree[dup_node][child]['reaction'])
                    )
                    _add_subtrees((child[0], new_idx))

        node_list = list(nx.topological_sort(self.tree))
        for node in node_list:
            if node[1]>1:
                _add_subtrees(node)


    def is_route_mapped(self):

        """Check all reactions are atom-mapped"""

        return all([has_mapping(rsmi) for rsmi in self.route])
        
    def is_atom_tracked(self):

        return any(self.leaf_tags)
    
    def tag_root_mol(self):

        """Tag the atoms in the root molecule with isotope labels (1~num_atoms)"""

        self.next_tag = 1
        mol = self.root_molecule.mol

        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(self.next_tag)
            self.next_tag += 1

        self.root_molecule.add_tagged_smiles(Chem.MolToSmiles(mol))
        self.root_tags = sorted([atom.GetAtomMapNum() for atom in mol.GetAtoms()])

    
    def get_map_to_tag(self, parent):

        """Map_to_tag only for product mapping to isotope tags"""
        parent_molecule = self.tree.nodes[parent]['molecule']

        try: 
            return parent_molecule.map_to_tag
        
        except AttributeError:

            map_to_tag = {}
            tagged_mol = parent_molecule.mol
            mapped_mol = Chem.MolFromSmiles(parent_molecule.smiles_as_product)
            
            sublist=list(mapped_mol.GetSubstructMatch(tagged_mol))

            # print(sublist, Chem.MolToSmiles(tagged_mol), Chem.MolToSmiles(mapped_mol))
                
            for atom in tagged_mol.GetAtoms():
                # print(atom.GetIdx())
                atom1 = mapped_mol.GetAtomWithIdx(sublist[atom.GetIdx()])
                map_to_tag[atom1.GetAtomMapNum()] = atom.GetAtomMapNum()

            parent_molecule.add_map_to_tag(map_to_tag)
            return map_to_tag

    
    def update_child_with_parent_tag(self, child, map_to_tag):

        child_molecule = self.tree.nodes[child]['molecule']
        mapped_mol = Chem.MolFromSmiles(child_molecule.smiles_as_reactant)

        for atom in mapped_mol.GetAtoms():
            tag = map_to_tag.get(atom.GetAtomMapNum(), 0) 
            if tag: 
                atom.SetAtomMapNum(tag)
            else:
                atom.SetAtomMapNum(self.next_tag)
                self.next_tag += 1
        
        child_molecule.add_tagged_smiles(Chem.MolToSmiles(mapped_mol))
        child_molecule.mol = mapped_mol

    def update_edges_with_tagged_reaction_smiles(self, node):
            
        tagged_children_smiles = sorted([self.tree.nodes[child]['molecule'].tagged_smiles for child in self.tree.successors(node)])
        tagged_reaction_smiles = ".".join(tagged_children_smiles) + ">" + self.tree.nodes[node]['molecule'].tagged_smiles

        for child in self.tree.successors(node):
            self.tree[node][child]['reaction'].tagged_reaction_smiles = tagged_reaction_smiles

    def sort_leaf_tags(self):

        # sort leaf tags by the number of common elements with the root
        self.leaf_tags = sorted(
            self.leaf_tags, 
            key = lambda x: len(set(x) & set(self.root_tags)), 
            reverse=True
        )


    def track_atoms(self):

        if not self.is_route_mapped():
            raise ValueError('All reactions must be atom-mapped')        

        for node in list(nx.topological_sort(self.tree)):

            # if leaf node, continue
            if self.tree.out_degree(node) == 0:
                self.leaf_tags.append(
                    tuple(get_atom_maps(
                        self.tree.nodes[node]['molecule'].tagged_smiles
                    ))
                )
                continue
            
            # if root node
            if node == self.root:  
                self.tag_root_mol()

            for child in self.tree.successors(node):

                # map_to_tag: atom_map_num -> isotope for map_num in the reaction_smiles
                # that the species is included as a product
                map_to_tag = self.get_map_to_tag(node)
                self.update_child_with_parent_tag(child, map_to_tag)
            
            self.update_edges_with_tagged_reaction_smiles(node)
            
            
        
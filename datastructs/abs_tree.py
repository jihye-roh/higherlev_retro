import sys
sys.path.append('..')
import os
import copy
import gzip
import json 
import logging
import networkx as nx

import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from utils.chem_utils import get_atom_maps, get_changed_bond_atoms, try_neutralize_smi, canonicalize

import PIL
from PIL import Image, ImageChops
from utils.abstraction_utils import abstract_reactant, set_verbose
from datastructs.syn_tree import SynTree
from datastructs.viz_tree import VizTree
from datastructs.datastructs import NestedDefaultDict


class AbsTree(SynTree):

    def __init__(self, reactions, route_id = -1):
        
        self.verbose = False
        self.smiles_to_reaction = {}
        self.reaction_ids = []
        self.initialize_route(reactions, route_id)

        super().__init__(self.route)
        self.track_atoms()
        self.update_reaction_data()
        self.initialize_target_smiles()
        self.map_to_label = NestedDefaultDict()
        self.abs_smiles_to_node = NestedDefaultDict()
        self.node_to_subtree_root = {}
        self.subtrees=[]
        self.subtree_reaction_smiles = {}
        
        

    # route could be a list of strings or a list of dictionaries
    def initialize_route(self, reactions, route_id):
        
        if not reactions or isinstance(reactions[0], str):
            self.route_id = f"ID_{route_id}"
            self.route = reactions
            reactions = [
                {'_id': idx,
                'reaction_smiles': reaction}
                for idx, reaction in enumerate(reactions)
            ]
        else:
            patent_id = reactions[0].get('patent_id', 'unknown')
            self.route_id = f"{patent_id}_{route_id}"
            self.route = [reaction['reaction_smiles'] for reaction in reactions]

        self.smiles_to_reaction ={
            reaction['reaction_smiles']: reaction
            for reaction in reactions
        }

        self.reaction_ids = sorted([
            reaction['_id']
            for reaction in reactions
        ])
    
    def initialize_target_smiles(self):
        
        # initial target smiles is the root smiles
        root_smiles = self.tree.nodes[self.root]['molecule'].tagged_smiles
        self.target_smiles_list = [root_smiles]

    def update_reaction_data(self):
        
        for edge_data in self.tree.edges.values():
            reaction = edge_data['reaction']
            self.smiles_to_reaction[reaction.reaction_smiles]['tagged_reaction_smiles'] = \
                reaction.tagged_reaction_smiles

    def propagate_label(self, smiles, subtree_root):

        if not smiles: return None

        mol = Chem.MolFromSmiles(smiles)

        for atom in mol.GetAtoms():
            isotope_label =  atom.GetIsotope()
            atom_map = atom.GetAtomMapNum()
            if isotope_label:
                self.map_to_label[subtree_root][atom_map] = isotope_label
            # propagate the label to the children if necessary
            elif atom_map in self.map_to_label[subtree_root]:
                if atom.GetTotalNumHs() and atom.GetAtomicNum()!=6:
                    atom.SetIsotope(self.map_to_label[subtree_root][atom_map])

        return Chem.MolToSmiles(mol)

    def get_target_node(self, node):

        current_node = node
        
        while True:
            predecessors = list(self.abs_tree.predecessors(current_node))

            # either return the current node if no predecessors or if there is an atom in the current molecule that is not in the parent molecule
            if not predecessors:
                return current_node
            
            try:
                current_atom_maps = get_atom_maps(self.abs_tree.nodes[current_node]['molecule'].abs_smiles)
                predecessor_atom_maps = get_atom_maps(self.abs_tree.nodes[predecessors[0]]['molecule'].abs_smiles)
                if current_atom_maps - predecessor_atom_maps:
                    # print(f"Non-root abstraction reference for node {node} is {current_node}")
                    return current_node
            except:
                pass
            current_node = predecessors[0]

    def get_target_smiles(self, node):

        """
        Returns the smiles of the node to use as a reference in abstraction
        """
        target_node = self.get_target_node(node)
        target_molecule = self.abs_tree.nodes[target_node]['molecule']

        return target_molecule.abs_smiles

    def find_subtree_root(self, node):
        current_node = node
        while True:
            predecessors = list(self.abs_tree.predecessors(current_node))
            if not predecessors:
                return current_node
            current_node = predecessors[0]

    def get_subtree_root(self, node):

        if node not in self.node_to_subtree_root:
            subtree_root = self.find_subtree_root(node)
            self.node_to_subtree_root[node] = subtree_root

        return self.node_to_subtree_root[node]

    def get_abs_parent_node(self, node):

        return next(iter(self.abs_tree.predecessors(node)))

    def get_parent_reaction(self, node):

        parent_node = next(self.tree.predecessors(node))
        return self.tree.edges[parent_node, node]['reaction'].reaction_smiles

    def get_nonisomeric_abs_smiles(self, abs_smiles):

        if not abs_smiles: return None
        nonisomeric_abs_smiles = canonicalize(
            try_neutralize_smi(abs_smiles), 
            isomericSmiles=False,
            remove_atm_mapping=False
        )
        return nonisomeric_abs_smiles

    def update_new_root(self, node):
        
        self.subtree_reaction_smiles[node] = set([])
        molecule = self.abs_tree.nodes[node]['molecule']
        molecule.abs_smiles = molecule.tagged_smiles
        molecule.nonisomeric_abs_smiles = \
            self.get_nonisomeric_abs_smiles(molecule.abs_smiles)

        for child in self.abs_tree.successors(node):
            self.node_to_subtree_root[child] = node
            self.abstract_single_node(child)
        
       

    def update_subtree_reaction(self, node):

        subtree_root = self.get_subtree_root(node)
        parent = next(self.tree.predecessors(node))
        self.subtree_reaction_smiles[subtree_root].add(self.get_parent_reaction(node))
        
    def check_and_split_tree(self, node):

        """
        Compare non-isomeric abs smiles of current node with parent node
        If same (but the abs_smiles are different), split tree
        """

        parent_node = self.get_abs_parent_node(node)
        parent_molecule = self.abs_tree.nodes[parent_node]['molecule']
        molecule = self.abs_tree.nodes[node]['molecule']
        # print("Nonisomeric_smiles", molecule.nonisomeric_abs_smiles, parent_molecule.nonisomeric_abs_smiles)
        # print("Abs_smiles", molecule.abs_smiles, parent_molecule.abs_smiles)
        if molecule.nonisomeric_abs_smiles == parent_molecule.nonisomeric_abs_smiles:
            if molecule.abs_smiles != parent_molecule.abs_smiles:
                # splitting the tree into two due to only stereochemistry changing in reaction
                reactants = list(self.abs_tree.successors(parent_node))
                can_parent = parent_node[0]
                # make a copy of the parent node
                self.node_counter[can_parent] = \
                    self.node_counter.get(can_parent, 0) + 1
                
                parent_copy = copy.deepcopy(self.abs_tree.nodes[parent_node])
                

                new_parent_node = (parent_node[0], self.node_counter[can_parent])
                self.abs_tree.add_node(new_parent_node, **parent_copy)

                for n in reactants:
                    # print("Removing edge", parent_node, n)
                    self.abs_tree.add_edge(new_parent_node, n, **self.abs_tree.edges[(parent_node, n)])
                    self.abs_tree.remove_edge(parent_node, n)
                    self.update_new_root(new_parent_node)
        
                return True

        return False

    def remove_cyclic_edges(self, node, subtree_root, abs_smiles):
        # print("Removing cyclic edges for node", node, abs_smiles)
        # Get the shortest path from the previous node to the current node
        prev_node = self.abs_smiles_to_node[subtree_root][abs_smiles]
        path = nx.shortest_path(
            self.abs_tree, 
            prev_node, 
            node)

        # Get the parent node
        parent = next(self.abs_tree.predecessors(prev_node))
        edge_data = self.abs_tree[parent][prev_node]

        # Remove nodes along the path except the last node
        for n in path[:-1]:
            abstracted_smi = self.abs_tree.nodes[n]['molecule'].abs_smiles
            self.abs_smiles_to_node[subtree_root].pop(abstracted_smi)
            self.abs_tree = nx.contracted_edge(self.abs_tree, (parent, n), self_loops=False)

        # Add the node to the dictionary and update the tree
        self.abs_smiles_to_node[subtree_root][abs_smiles] = node
        self.abs_tree.add_edge(parent, node, **edge_data)
        

    def abstract_single_node(self, node):

        # print("Abstracting node", node)

        molecule = self.abs_tree.nodes[node]['molecule']

        if self.abs_tree.in_degree(node) == 0:

            # print("Root node", node)

            molecule.abs_smiles = molecule.tagged_smiles
            molecule.nonisomeric_abs_smiles = \
                self.get_nonisomeric_abs_smiles(molecule.abs_smiles)
            self.node_to_subtree_root[node] = node
            self.abs_smiles_to_node[node][molecule.abs_smiles]= node
            self.subtree_reaction_smiles[node] = set([])

            return

        # print("Non-root node", node)
        reactant_smiles = molecule.tagged_smiles
        target_smiles = self.get_target_smiles(node)
        subtree_root = self.get_subtree_root(node)

        abs_smiles = abstract_reactant(reactant_smiles, target_smiles)
        molecule.abs_smiles = self.propagate_label(abs_smiles, subtree_root)
        molecule.nonisomeric_abs_smiles = \
            self.get_nonisomeric_abs_smiles(molecule.abs_smiles)

        if not molecule.abs_smiles: 
            # print("Returning due to no abs_smiles")
            self.update_subtree_reaction(node)
            return
        if self.check_and_split_tree(node): return

        self.update_subtree_reaction(node)
        # subtree_root = self.get_subtree_root(node)
        if molecule.abs_smiles not in self.abs_smiles_to_node[subtree_root]:
            self.abs_smiles_to_node[subtree_root][molecule.abs_smiles] = node

        else:
            self.remove_cyclic_edges(node, subtree_root, molecule.abs_smiles)

    def print_node_and_edge_data(self, message, print_nodes = True, print_edges = True):
            
        if self.verbose:
            print(message)
            if print_nodes:
                for node in self.abs_tree.nodes:
                    print(node)
            if print_edges:
                for edge in self.abs_tree.edges:
                    print(edge)
            print()

    def abstract_all_nodes(self):

        self.print_node_and_edge_data("Before abstracting all nodes")

        for node in nx.topological_sort(self.abs_tree):        
            self.abstract_single_node(node)
            self.print_node_and_edge_data(f"After abstracting node {node}")
        
        self.print_node_and_edge_data("After abstracting all nodes")

        self.remove_nodes_without_abs_smiles()
        
    def remove_nodes_without_abs_smiles(self):
            
        node_list = list(self.abs_tree.nodes)
        for node in node_list:
            molecule = self.abs_tree.nodes[node]['molecule']
            if not molecule.abs_smiles:
                self.abs_tree.remove_node(node)

    def collect_reactions_from_subtree(self, subtree, idx):

        subtree_root = next(iter(nx.topological_sort(subtree)))
        subtree_reactions_dict = {
            smiles: {
                '_id': self.smiles_to_reaction[smiles]['_id'],
                'reaction_smiles': self.smiles_to_reaction[smiles]['reaction_smiles'],
                'abstracted_reaction_smiles': ''
            }
            for smiles in self.subtree_reaction_smiles[subtree_root]
        }

        for node in nx.topological_sort(subtree):

            # continue if leaf node or no abs_smiles
            if subtree.out_degree(node) == 0:
                continue

            if not subtree.nodes[node]['molecule'].abs_smiles: 
                continue
                
            abs_reactants = [
                subtree.nodes[successor]['molecule'].abs_smiles
                for successor in subtree.successors(node)
            ]
            abs_reactants = [reactants for reactants in abs_reactants if reactants]
            abs_reactants = sorted(abs_reactants)

            if not abs_reactants: continue

            molecule = subtree.nodes[node]['molecule']

            abs_reactant_smi = '.'.join(abs_reactants)
            if canonicalize(abs_reactant_smi) == canonicalize(molecule.abs_smiles):
                continue
            
            successor = next(iter(subtree.successors(node)))
            original_reaction_smiles = \
                subtree.edges[node, successor]['reaction'].reaction_smiles

            subtree_reactions_dict[original_reaction_smiles]['abstracted_reaction_smiles'] = \
                abs_reactant_smi + '>>' + molecule.abs_smiles

        return list(subtree_reactions_dict.values())
    
    def contract_tree(self):

        self.subtrees_data = []

        self.subtrees = [
            self.abs_tree.subgraph(c) 
            for c in nx.weakly_connected_components(self.abs_tree)
            if len(c) > 1
        ]

        # print("Subtrees", len(subtrees), [len(subtree) for subtree in subtrees])
        self.print_node_and_edge_data("Before contracting tree")

        for idx, subtree in enumerate(self.subtrees):
            subtree_id = self.route_id + f"_{idx}"

            reactions = self.collect_reactions_from_subtree(subtree, idx)
            
            for reaction in reactions:
                reaction['_id'] = f"{subtree_id}_{reaction['_id']}"

            ori_reactions = [reaction['reaction_smiles'] for reaction in reactions]
            ori_depth = SynTree(ori_reactions).depth
            ori_num_reactions = len(ori_reactions)

            abs_depth = nx.dag_longest_path_length(subtree)
            abs_num_reactions = sum([1 for reaction in reactions if reaction['abstracted_reaction_smiles']])

            subtree_data = {
                'subtree_id': subtree_id,
                'num_reactions': (ori_num_reactions, abs_num_reactions),
                'depth': (ori_depth, abs_depth),
                'reactions': reactions  
            }

            self.subtrees_data.append(subtree_data)

    def generate_abs_tree(self):
        
        self.abs_tree = copy.deepcopy(self.tree)
        self.abstract_all_nodes()
        self.contract_tree()
        self.abstraction_data = {
            'route_id': self.route_id,  
            'original_tree': {
                            'depth': self.depth,
                            'num_reactions': len(self.reaction_ids),
                            'reaction_ids': self.reaction_ids,
                            # 'reactions': list(self.smiles_to_reaction.values())
            },
            'subtrees': self.subtrees_data
        }
            
    def get_abstraction_data(self):
        self.generate_abs_tree()
        return self.abstraction_data

    def collect_reactions(self):

        pass


    def visualize_trees(self, img_scale=1.0, img_dir = '../tree_imgs/tree_imgs_comb'):

        visualize_trees_from_data(self.abstraction_data, img_scale, img_dir)




def save_tree_img_from_route(route, img_scale = 1.0):

    viz_tree = VizTree(route)
    tree_path = viz_tree.plot_tree(
        includeHighlights = True, 
        img_scale = img_scale, 
        show=False
    )
    return tree_path


def save_tree_imgs(data, img_scale = 1.0):
    all_ori_reactions = []
    abs_subtree_paths = []
    ori_subtree_paths = []
    for i, subtree in enumerate(data['subtrees']):
        print(f"Abstracted tree {i}")

        ori_route = [
            reaction['reaction_smiles']
            for reaction in subtree['reactions']
        ]

        all_ori_reactions.extend(ori_route)

        abs_route = [
            reaction['abstracted_reaction_smiles'] 
            for reaction in subtree['reactions']
            if reaction['abstracted_reaction_smiles']    
        ]

        path = save_tree_img_from_route(ori_route, img_scale)
        if path: 
            ori_subtree_paths.append(path)

        path = save_tree_img_from_route(abs_route, img_scale)
        if path: 
            abs_subtree_paths.append(path)

    print("Original tree")
    all_ori_reactions = list(set(all_ori_reactions))
    ori_tree_path = save_tree_img_from_route(all_ori_reactions, img_scale)

    return ori_tree_path, ori_subtree_paths, abs_subtree_paths

def visualize_trees_from_data(data, img_scale=1.0, img_dir = '../tree_imgs/tree_imgs_comb'):

    ori_tree_path, ori_subtree_paths, abs_subtree_paths = save_tree_imgs(data, img_scale)

    images = [plt.imread(ori_tree_path)]
    titles = ["Original tree"]
    for i, paths in enumerate(zip(ori_subtree_paths, abs_subtree_paths)):
        p1, p2 = paths
        #images.append(plt.imread(p1))
        images.append(plt.imread(p2))
        #titles.append(f"Split tree {i}")
        titles.append(f"Abstracted (Higher-level) tree")

    images = adjust_images(images)

    # Calculate the figure size and subplot dimensions
    fig_height = 2.5*data['original_tree']['depth']
    img_height, img_width, _ = images[0].shape
    scale = fig_height / img_height
    
    # Calculate the total width of all subplots including spacing
    spacing = 0.1
    total_width = sum([img.shape[1] * scale for img in images]) + spacing*len(images)
    total_height = max([img.shape[0] * scale for img in images])

    fig, ax = plt.subplots(1, len(images), figsize=(total_width, total_height))

    # Calculate normalized subplot widths and heights
    subplot_widths = [(img.shape[1] * scale) / total_width for img in images]
    subplot_heights = [(img.shape[0] * scale) / total_height for img in images]

    next_x = 0.5*spacing
    for i, img in enumerate(images):
        
        imgheight, imgwidth, _ = img.shape
        
        # Set the position of each subplot
        left = next_x
        bottom = 1.0 - subplot_heights[i]  # Align at the top

        ax[i].set_position([left, bottom, subplot_widths[i], subplot_heights[i]])

        # Display the image
        ax[i].imshow(img, extent=[0, imgwidth, 0, imgheight])
        ax[i].axis('off')
        ax[i].set_title(titles[i])
        ax[i].set_anchor('N')
        

        bbox = ax[i].get_position()
        next_x = bbox.x0 + subplot_widths[i] + spacing

    # save
    os.makedirs(img_dir, exist_ok=True)
    plt.savefig(img_dir + f"/{data['route_id']}.png", bbox_inches='tight', pad_inches=0)

    os.remove(ori_tree_path)
    for path in ori_subtree_paths+abs_subtree_paths:
        os.remove(path)

def collect_reactions(route_data):
    original_route = [
        r['reaction_smiles']
        for r in route_data['subtrees'][0]['reactions']
    ]
    higherlev_route = [
        r['abstracted_reaction_smiles']
        for r in route_data['subtrees'][0]['reactions']
        if r['abstracted_reaction_smiles']
    ]

    return original_route, higherlev_route

def adjust_single_image(image):
    image = Image.fromarray((image* 255).astype(np.uint8))
    image = image.crop((2, 2, image.width-2, image.height - 2))

    grayscale = image.convert('L')
    inverted = ImageChops.invert(grayscale)

    # Get bounding box of non-zero regions in the image
    bbox = inverted.getbbox()
    cropped_image = image.crop(bbox)
    return np.array(cropped_image)
    
def adjust_images(images):
    new_images = [adjust_single_image(image) for image in images]
    return new_images

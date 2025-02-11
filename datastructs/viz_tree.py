import os
import datetime

from rdkit import Chem, Geometry
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from collections import Counter

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from math import sqrt
import numpy as np

from datastructs.syn_tree import SynTree

HIGHLIGHT_COLORS = [(1, 1, 0.6, 0.75), (1, 0.7, 1, 0.75), (0.7, 1, 1, 0.75), 
                   (1, 0.7, 0.7, 0.75), (0.7, 1, 0.7, 0.75), (0.75, 0.75, 1, 0.75),
                   (0.8, 0.9, 0.7, 0.75), (0.8, 0.7, 0.9, 0.75), (0.9, 0.8, 0.7, 0.75), 
                   (0.9, 0.6, 0.8, 0.75), (0.6, 0.8, 0.9, 0.75), (0.6, 0.9,0.8, 0.75)]



def save_img(mol, img_path, highlight_atoms = {}):
    
    """
    Save individual molecule images as png with the same bond length
    Can also highlight specific atoms with specific colors

    Args:
    mol: rdkit mol object
    img_path: path to save image
    highlight_atoms: dictionary of atom indices to colors for atoms to highlight

    """

    highlight_bonds = {}
    highlight_atom_radii = {idx: 0.5 for idx in highlight_atoms}

    mol=Chem.Mol(mol)

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    if highlight_atoms:
        for bond in mol.GetBonds():
            beg = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            if beg in highlight_atoms and end in highlight_atoms:
                if highlight_atoms[beg] == highlight_atoms[end]:
                    highlight_bonds[bond.GetIdx()] = highlight_atoms[end]
            
    mc = Chem.Mol(mol.ToBinary())
    AllChem.Compute2DCoords(mc)
    
    coords = mc.GetConformer(-1).GetPositions()
    
    # if the margin +/-1 is removed linear molecules are broken
    min_p = Geometry.Point2D(*coords.min(0)[:2] - 1)
    max_p = Geometry.Point2D(*coords.max(0)[:2] + 1)
    
    dpa = 50
    # try to catch 0's with max()
    w = int(dpa * (max_p.x - min_p.x)) + 1
    h = int(dpa * (max_p.y - min_p.y)) + 1

    #drawer = rdMolDraw2D.MolDraw2DSVG(max(w, dpa), max(h, dpa))
    drawer = rdMolDraw2D.MolDraw2DCairo(max(w, dpa), max(h, dpa))
    dopts = drawer.drawOptions()
    dopts.bondLineWidth = 5
    
    if highlight_atoms: 
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mc, highlightAtoms=highlight_atoms.keys(),
                                            highlightAtomColors=highlight_atoms,
                                            highlightAtomRadii=highlight_atom_radii,                                     
                                            highlightBonds=highlight_bonds.keys(),
                                            highlightBondColors=highlight_bonds)
    else: 
        drawer.DrawMolecule(mc)
        
    drawer.FinishDrawing()
    drawer.WriteDrawingText(img_path+'.png')



class VizTree(SynTree):

    """Visuzalizing synthesis pathways, with nodes as molecules and edges as reactions"""

    def __init__(self, route = None, width = 2.0, img_dir = '../tree_imgs'):

        super().__init__(route)
        self.level = nx.shortest_path_length(self.tree, self.root)
        # counting the number of nodes at each level
        #self.level_count = [sum(1 for key in self.level if self.level[key] == i) for i in range(max(self.level.values())+1)] 
        self.level_count = Counter(self.level.values())
        self.img_dir = img_dir
        self.positions = {}
        self.subtree_widths = {}
        self.width = width

        self.make_img_dir()
    
    def make_img_dir(self):
        os.makedirs(os.path.join(self.img_dir, 'temp'), exist_ok=True)
        os.makedirs(os.path.join(self.img_dir, 'tree_imgs'), exist_ok=True)
                        
    def save_node_imgs(self, includeHighlights = False):

        if includeHighlights:
            self.add_map_to_highlight()
            self.add_highlight_colors()

        for node, data in self.tree.nodes(data=True):

            path_name = f"{node[0][:200].replace('/', '_')}_{node[1]}"
            img_path = os.path.join(self.img_dir, 'temp', path_name)
            
            molecule=data["molecule"]
            
            if includeHighlights: 
                current_time = datetime.datetime.now()
                img_path+=str(current_time)[-6:]
                save_img(molecule.mol, img_path, highlight_atoms = molecule.highlight_atoms)
                self.tree.nodes[node]["highlighted_image_path"] = img_path+'.png'

            else:
                save_img(molecule.mol, img_path)
                self.tree.nodes[node]["image_path"] = img_path+'.png'
            
    def add_map_to_highlight(self):

        if not self.is_atom_tracked():
            self.track_atoms()
            self.sort_leaf_tags()
        
        map_to_highlight = {}
        
        for i, maps in enumerate(self.leaf_tags):
            for atom_map in maps:
                if atom_map in self.root_tags:
                    map_to_highlight[atom_map] = i % len(HIGHLIGHT_COLORS)

        # add highlight for atoms in target molecule without source
        for i, atom_map in enumerate(self.root_tags):
            if atom_map not in map_to_highlight:
                map_to_highlight[atom_map] = i % len(HIGHLIGHT_COLORS)

        self.map_to_highlight = map_to_highlight

        
    def add_highlight_colors(self):

        for node, data in self.tree.nodes(data=True):

            highlight_atoms = {}
            molecule = data["molecule"]

            for atom in molecule.mol.GetAtoms():
                atom_map = atom.GetAtomMapNum()
                if atom_map in self.map_to_highlight:
                    highlight_atoms[atom.GetIdx()] = HIGHLIGHT_COLORS[self.map_to_highlight[atom_map]]

            molecule.highlight_atoms = highlight_atoms


    def topo_pos(self, width =2., pos = None):

        if not nx.is_tree(self.tree):
            print('cannot use topo_pos on a graph that is not a tree')
            raise TypeError('Not a tree')
            
        root_loc = (width/2, 0)
        
        n_max = max(self.level_count.values())
        dy = 1.
        
        
        def _topo_pos(root, dy, width, root_loc, pos = None):

            if pos is None:
                pos = {root: root_loc}
            else:
                pos[root] = root_loc

            children = list(self.tree.neighbors(root))
            
            root_level = self.level[root]
            
            if len(children)!=0:
                
                if len(children) > self.level_count[root_level+1]-3:
                    width = 1.0
                    
                if self.level_count[root_level] == 1 or len(children) == self.level_count[root_level+1]:
                    width =2.0
                
                dx = width/len(children)     
                nextx = root_loc[0] - width/2 - dx/2
                
                reverse = (root_loc[0]>1.8 or (0.2<root_loc[0] and root_loc[0] < 1.))

                children = sorted(children, key=lambda x: len(list(self.tree.neighbors(x))), reverse = reverse)
                
                width = dx
                
                for child in children:
                    nextx += dx
                    pos = _topo_pos(child, dy, width = width, root_loc = (nextx, root_loc[1]-dy), pos=pos)

            return pos

        return _topo_pos(self.root, dy, width, root_loc, pos)


    def plot_tree(self, img_scale=1, includeHighlights=False, save_file=True, show=True):
        #pos = self.get_positions(self.width)

        pos = self.topo_pos()

        root = self.root

        img_path = 'highlighted_image_path' if includeHighlights else 'image_path'

        try:
            img = mpimg.imread(self.tree.nodes[root][img_path])
        except:
            self.save_node_imgs(includeHighlights=includeHighlights)
            img = mpimg.imread(self.tree.nodes[root][img_path])

        root_height, root_width, _ = img.shape

        delx = max(pos.values(), key=lambda p: p[0])[0] - min(pos.values(), key=lambda p: p[0])[0]
        dely = max(pos.values(), key=lambda p: p[1])[1] - min(pos.values(), key=lambda p: p[1])[1]

        scale = 0.5*sqrt(1 / root_height / root_width) * img_scale

        delx = delx+root_width*scale
        dely = dely+root_height*scale

        fig, ax = plt.subplots(figsize=(5 * delx, 3* dely))
        ori_bounds = ax.get_position().bounds

        for node, data in self.tree.nodes(data=True):
            img = mpimg.imread(data[img_path])
            node_height, node_width, _ = img.shape
            scale = 0.5 * sqrt(1 / root_height / root_width) * img_scale

            imgheight = node_height * scale
            imgwidth = node_width * scale

            ax.imshow(img, extent=[pos[node][0] - imgwidth, pos[node][0] + imgwidth,
                                   pos[node][1] - imgheight, pos[node][1] + imgheight])
            ax.axis('off')
            # show the positon of images
            # how to get what the position 
        
        node_size = min(self.depth+1, 4)*5000
        margin = min(self.depth, 2)*50

        nx.draw_networkx_edges(self.tree, pos, width=2.0, edge_color='gray', arrowsize=15,
                               node_shape='X', node_size=node_size, alpha=1,
                               ax=ax, min_source_margin=margin, min_target_margin=margin)

        w,h = fig.get_size_inches()
        x1,x2 = ax.get_xlim()
        y1,y2 = ax.get_ylim()
        fig.set_size_inches(w, ax.get_aspect()*(y2-y1)/(x2-x1)*w)

        if save_file:
            path_name = self.root[0].replace('/', '_').replace("\\", "__")
            path_name = path_name[:200]
            curr_time = datetime.datetime.now()
            path_name += f"_{str(curr_time)[-6:]}"
            img_path = os.path.join(self.img_dir, 'temp', path_name)
            file_path = os.path.join(self.img_dir, 'tree_imgs', path_name + "_tree.png")
            plt.savefig(file_path, bbox_inches='tight', pad_inches=0)


        if show: plt.show()
        plt.close(fig)

        self.remove_node_images()

        if save_file:
            return file_path


    def remove_node_images(self):

        path = os.path.join(self.img_dir, 'temp')
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))

            
    
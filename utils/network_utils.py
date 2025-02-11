import sys
sys.path.append('../')

import os
import re
import gc
import gzip
import json
import requests
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw
from ASKCOSv2.tree_search.mcts.pathway_utils import prune, nx_graph_to_paths


color = (1, 0.5, 0, 0.5)

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://0.0.0.0:9100")
pricer_url=f"{GATEWAY_URL}/api/pricer/lookup-smarts" 
smiles_to_smarts_url=f"{GATEWAY_URL}/api/pricer/smiles-to-lookup-smarts" 
session = requests.Session()

KEYS_TO_KEEP = ["id", "type", "terminal", "iteration"]

def extract_min_info(data, keys_to_keep = KEYS_TO_KEEP):
    # Extract the 'nodes' list from the dictionary
    nodes = data.get('nodes', [])
    
    # Filter each dictionary in 'nodes' to keep only the specified keys
    filtered_nodes = [
        {
            key: node[key]
            for key in keys_to_keep if key in node
        }
        for node in nodes
    ]
    
    return {**data, 'nodes': filtered_nodes}


def load_graph_as_nx(
    graph_file: str, 
    root: str = None, 
    use_min_info: bool=False, 
    keys_to_keep=KEYS_TO_KEEP
): 
    with gzip.open(graph_file, 'rt') as f: 
        data = json.load(f)

    if root is None: 
        root = data["result"]["graph"]["nodes"][0]["id"]

    graph_data = data["result"]["graph"]
    if use_min_info:
        graph_data = extract_min_info(graph_data, keys_to_keep)

    G = nx.node_link_graph(
        graph_data
    )

    for node in G.nodes():
        G.nodes[node]["depth"] = \
            int(
                (nx.shortest_path_length(G, root, node)+1)/2
            )
    
    del graph_data
    del data
    gc.collect()

    return G, root
    

def nx_graph_to_rxn_smiles(
    path=nx.DiGraph(),
):
    return [
        d["smiles"] for n, d in path.nodes(data=True)
        if ">>" in d["smiles"]
    ]

def get_pathway_stats(path: nx.DiGraph):
    # Initialize counters
    num_rxn_nodes = 0
    num_chemical_nodes = 0
    depth = 0
    max_iteration = 0

    # Iterate over all nodes once
    for _, d in path.nodes(data=True):
        # Count reactions and chemicals based on "smiles" key
        if ">>" in d.get("smiles", ""):
            num_rxn_nodes += 1
        else:
            num_chemical_nodes += 1
        
        # Track maximum depth and iteration
        depth = max(depth, d.get("depth", 0))
        max_iteration = max(max_iteration, d.get("iteration", 0))

    # Return the results as a dictionary
    return {
        "num_chemical_nodes": num_chemical_nodes,
        "num_rxn_nodes": num_rxn_nodes,
        "depth": depth,
        "max_iteration": max_iteration
    }

def get_matched_buyables(
    smiles, 
    max_buyables=10,
    display_buyables=False
):

    """Returns a list of buyable SMILES for a given input higher-level molecule SMILES"""

    smiles_all = []
    mols_all = []

    smarts_list = session.post(
        url=smiles_to_smarts_url, 
        params={"smiles": smiles}, 
        verify=False
    ).json()

    for smarts in smarts_list:

        try:
            params = {
                "smarts": smarts, 
                "limit": max_buyables,
                "version": "preloaded_vec",
                "max_ppg": 100.0,
            }
            response = session.post(
                url=pricer_url, 
                params=params, 
                verify=False
            ).json()
            
        except:
            response = []

        buyables_smi = [
            r["smiles"] for r in response 
            if Chem.MolFromSmiles(r["smiles"])
        ]

        buyables_smi = buyables_smi[:max_buyables]
    
        if display_buyables:
            _display_buyables(buyables_smi, query_smiles = smiles, query_smarts=smarts)

    return buyables_smi


def _display_buyables(buyables_smi, query_smiles, query_smarts):

    buyable_mols = [
        Chem.MolFromSmiles(smi) for smi in buyables_smi
    ]
    pattern = Chem.MolFromSmarts(query_smarts)

    if buyable_mols:

        print("Query Molecule")
        img = Draw.MolToImage(
            Chem.MolFromSmiles(query_smiles), 
            size=(250, 250), 
            legend=query_smiles
        )
        display(img)

        print("Matched Buyable Molecules")
        highlight_atoms = [
            mol.GetSubstructMatch(pattern) for mol in buyable_mols
        ]
        highlight_atom_radii = [
            {idx: 0.4 for idx in mol}
            for mol in highlight_atoms
        ]
        highlight_atom_colors = [
            {atom: color for atom in atoms} 
            for atoms in highlight_atoms
        ]
        img = Draw.MolsToGridImage(
            buyable_mols, 
            molsPerRow=5, 
            subImgSize=(250, 250), 
            legends=buyables_smi, 
            highlightAtomLists=highlight_atoms, 
            highlightAtomColors=highlight_atom_colors, 
            highlightAtomRadii =highlight_atom_radii
        )
        display(img)

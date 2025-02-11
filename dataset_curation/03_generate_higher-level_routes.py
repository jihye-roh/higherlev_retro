import sys
sys.path.append('..')
import json 
import gzip
import time
import logging
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

from utils.utils import setup_logger, load_data
from datastructs.abs_tree import AbsTree


def _setup(data_dir="./data"):
    """Set up directories and logging for patent processing"""
    data_dir = Path(data_dir)
    route_dir = data_dir / "routes"
    route_dir.mkdir(parents=True, exist_ok=True)

    input_file = route_dir / "uspto.extracted.routes.jsonl.gz"
    output_file = route_dir / "uspto.routes.jsonl.gz"
    
    log_dir = Path("./logs/generate_higher-level_routes")
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger("log", log_dir)
    
    return input_file, output_file


def generate_abstraction_data(input_data):
    """Generate abstraction data for a single route"""
    i, route = input_data
    syn_tree = AbsTree(route, i)
    data = syn_tree.get_abstraction_data()
    return data

def process_data(data, exclude_patents = [], num_workers = 16):

    all_routes = [
        route for patent_id, routes in data
        for route in routes
        if patent_id not in exclude_patents
    ]

    logging.info(f"Loaded {len(all_routes)} routes")

    with Pool(num_workers) as p:
        results = list(tqdm(p.imap(generate_abstraction_data, enumerate(all_routes)),
                    total=len(all_routes), desc='Generating abstraction data'))

    return results

def save_as_csv(routes, data_dir="./data"):

    # Prepare output file paths
    data_dir = Path(data_dir)
    ofn1 = data_dir / "reactions" / "uspto_original.csv"
    ofn2 = data_dir / "reactions" / "uspto_higher-level.csv"
    ofn1.parent.mkdir(parents=True, exist_ok=True)

    # Lists to collect data for pandas DataFrame
    original_reactions = []
    abstracted_reactions = []

    logging.info(f"Saving reactions from {len(routes)} routes to csv")
    logging.info(f"Saving original reactions to {ofn1} "
                f"and abstracted reactions to {ofn2}")

    # Use a set for faster membership checking
    reaction_ids = set()

    # Iterate through data and gather reaction data
    for d in tqdm(routes, total=len(routes), desc="Saving reactions to csv"):
        for subtree in d['subtrees']:
            for reaction in subtree['reactions']:
                # Get the unique ID and check if we've already processed it
                idx = "_".join(reaction['_id'].split('_')[-2:])
                if idx not in reaction_ids:
                    reaction_ids.add(idx)
                    r, _, p = reaction['reaction_smiles'].split('>')
                    original_reactions.append([f"uspto_{idx}", f"{r}>>{p}"])

                # Check if abstracted reaction exists
                if reaction['abstracted_reaction_smiles']:
                    r, _, p = reaction['abstracted_reaction_smiles'].split('>')
                    abstracted_reactions.append([f"uspto_{reaction['_id']}", f"{r}>>{p}"])

    # Convert collected data to DataFrames
    original_df = pd.DataFrame(original_reactions, columns=['id', 'rxn_smiles'])
    abstracted_df = pd.DataFrame(abstracted_reactions, columns=['id', 'rxn_smiles'])

    # Write to CSV files using pandas (handles large datasets efficiently)
    original_df.to_csv(ofn1, index=False)
    abstracted_df.to_csv(ofn2, index=False)

    logging.info(f"Done saving reactions to csv")
    logging.info(f"No of original reactions: {len(original_reactions)}")
    logging.info(f"No of abstracted reactions: {len(abstracted_reactions)}")

if __name__ == "__main__":
    # Configuration
    data_dir="./data"
    input_file, output_file = _setup(data_dir=data_dir)

    # Patents to exclude (throw errors in later steps)
    exclude_patents = ['US08440628B2', 'US08518893B2']

    data = load_data(input_file)
    results = process_data(data, exclude_patents)
    logging.info(f"Generated {len(results)} abstracted routes")
    
    # Save
    with gzip.open(output_file, "wt") as f:
        [f.write(json.dumps(item) + '\n') for item in results]

    logging.info(f"Saved results to {output_file}")

    save_as_csv(results, data_dir=data_dir)
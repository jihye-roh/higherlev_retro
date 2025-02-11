import sys
sys.path.append('..')
import json
import gzip
import logging
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

from utils.extract import extract_one_patent
from utils.utils import load_data, setup_logger

def _setup(data_dir="./data"):
    """Set up directories and logging for patent processing"""
    data_dir = Path(data_dir)
    reaction_dir = data_dir / "reactions"
    route_dir = data_dir / "routes"
    reaction_dir.mkdir(parents=True, exist_ok=True)
    route_dir.mkdir(parents=True, exist_ok=True)

    input_file = reaction_dir / "uspto.processed.reactions.data.pkl.gz"
    output_file = route_dir / "uspto.extracted.routes.jsonl.gz"
    
    log_dir = Path("./logs/extract_multistep_routes")
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger("log", log_dir)

    return input_file, output_file

def extract_reactions_from_tree(node, reactions=None):
    """
    Recursively extract reactions from a tree structure
    
    Args:
        node: Current tree node
        reactions: List to accumulate reactions
        
    Returns:
        List of extracted reactions
    """
    if reactions is None:
        reactions = []
        
    if node.get('child'):
        reaction = node['record_data']
        reactions.append(reaction)
        for child in node['child']:
            extract_reactions_from_tree(child, reactions)
    
    return reactions

def process_single_patent(input_data):
    """
    Process reaction routes for a single patent
    
    Args:
        input_data: Tuple of (patent_id, trees, reactions)
        
    Returns:
        Tuple of (patent_id, routes)
    """
    patent_id, trees, reactions = input_data
    
    # Extract trees
    trees = [tree['tree'] for tree in trees]
    
    # Calculate routes for the current patent_id
    routes = sorted(
        (extract_reactions_from_tree(tree) for tree in trees),
        key=len, reverse=True
    )
    
    # Track included reactions
    included_reaction_ids = {
        reaction['_id'] 
        for route in routes 
        for reaction in route
    }
    
    # Add single-step routes for reactions not in extracted trees
    single_step_routes = [
        [reaction] for reaction in reactions 
        if reaction['_id'] not in included_reaction_ids
    ]
    
    routes.extend(single_step_routes)
    return patent_id, routes

def extract_pathways(reactions_by_patent, num_workers=16):
    """
    Extract pathways for all patents in parallel
    
    Args:
        reactions_by_patent: Dictionary mapping patent IDs to reactions
        num_workers: Number of parallel workers
        
    Returns:
        Dictionary mapping patent IDs to extracted pathways
    """
    extracted = Parallel(n_jobs=num_workers, verbose=1)(
        delayed(extract_one_patent)(reactions_by_patent[key], key) 
        for key in reactions_by_patent
    )
    
    pathways = {
        data["patent_id"]: data["trees"] 
        for data in extracted
    }
    
    logging.info(f"Patents with extracted pathways: {len(pathways)}")
    return pathways

def process_all_patents(reactions_by_patent, num_workers=16):
    """
    Process routes for all patents
    
    Args:
        reactions_by_patent: Dictionary mapping patent IDs to reactions
        num_workers: Number of parallel workers
        
    Returns:
        List of (patent_id, routes) tuples
    """
    pathways = extract_pathways(reactions_by_patent, num_workers)
    
    input_data = [
        (patent_id, pathways[patent_id], reactions_by_patent[patent_id])
        for patent_id in reactions_by_patent
    ]
    
    with Pool(num_workers) as p:
        results = list(tqdm(
            p.imap(process_single_patent, input_data),
            total=len(input_data)
        ))
        
    return results
    
if __name__ == "__main__":
    # Setup directories and logging
    input_file, output_file = _setup()
    
    # Load reaction data
    reactions_by_patent = load_data(input_file)
    
    # Process all patents
    results = process_all_patents(reactions_by_patent, num_workers=5)
    logging.info(f"Processed {len(results)} patents")

    # Save
    with gzip.open(output_file, "wt") as f:
        [f.write(json.dumps(item) + '\n') for item in results]

    logging.info(f"Saved results to {output_file}")

import sys
sys.path.append('..')
import gzip
import pickle
import logging
from pathlib import Path
from tqdm import tqdm
from itertools import chain
from multiprocessing import Pool
from collections import defaultdict

from utils.chem_utils import canonicalize_rsmi
from utils.clean_utils import separate_and_clean_smiles
from utils.utils import load_data, setup_logger

DEFAULT_ID = "USPTO"

def _setup(data_dir="./data"):
    """Set up directories and logging for patent processing"""
    data_dir = Path(data_dir)
    reaction_dir = data_dir / "reactions"
    reaction_dir.mkdir(parents=True, exist_ok=True)

    # A subset of uspto reactions to be used as an example ONLY
    input_file = f"{reaction_dir}/uspto.reactions.example.json.gz"
    output_file = f"{reaction_dir}/uspto.processed.reactions.data.pkl.gz"
    testset_file = f"{reaction_dir}/testset/uspto190_canon_reactions.smi"
    
    log_dir = Path("./logs/process_uspto_reactions")
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger("log", log_dir)

    return input_file, output_file, testset_file

def process_and_clean_single_reaction(input_data):
    """
    Process and clean a single reaction entry.
    Reactions must contain 'reaction_smiles', may have '_id' and 'patent_id'.
    Returns list of processed reactions split by products if multiple.
    """
    idx, reaction = input_data

    # Set default IDs if not present
    reaction.setdefault("_id", f"{DEFAULT_ID}_{idx}")
    reaction.setdefault("patent_id", DEFAULT_ID)

    rsmi = reaction["reaction_smiles"]
    cleaned_smiles = separate_and_clean_smiles(rsmi)

    return [
        {
            **reaction,
            "_id": f"{reaction['_id']}_{i}",
            "reaction_smiles": cleaned_smi,
            "canonical_smiles": canonicalize_rsmi(cleaned_smi),
            "original_smiles": rsmi
        }
        for i, cleaned_smi in enumerate(cleaned_smiles)
    ]

def load_testset_reactions(testset_file=None):
    """
    Load reactions to exclude from the dataset.
    Returns set of canonicalized SMILES strings.
    """
    if testset_file is None:
        logging.info("No testset file provided")
        return set()

    with open(testset_file, 'r') as f:
        testset_smiles = {line.strip() for line in f}

    logging.info(f"Loaded {len(testset_smiles)} reactions to exclude")
    return testset_smiles

def separate_by_source(reactions):
    """
    Group reactions by their patent_id.
    Returns dict mapping patent_ids to reaction lists.
    """
    reactions_by_patent = defaultdict(list)
    
    for reaction in tqdm(reactions, desc="Grouping by Patent ID"):
        reactions_by_patent[reaction['patent_id']].append(reaction)

    # Sort by number of reactions per source
    reactions_by_patent = dict(sorted(
        reactions_by_patent.items(), 
        key=lambda x: len(x[1]), 
        reverse=True
    ))
    logging.info(f"Found {len(reactions_by_patent)} unique patents")
    
    # Log stats about the most frequent patent
    most_frequent_patent = next(iter(reactions_by_patent.items()))
    logging.info(
        f"Max # rxn: patent - {most_frequent_patent[0]}, "
        f"# rxn - {len(most_frequent_patent[1])}"
    )

    return reactions_by_patent

def process_reactions(reactions, testset_smiles=[], num_workers=16):
    """
    Process and clean reaction data from input file.
    Saves processed reactions to output file.
    """

    # Process reactions in parallel
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(
                process_and_clean_single_reaction, 
                enumerate(reactions)
            ),
            desc="Processing reactions",
            total=len(reactions)
        ))
    results = list(chain.from_iterable(results))
    logging.info(f"Generated {len(results)} cleaned reactions")

    # Filter out testset reactions
    if testset_smiles:
        results = [
            r for r in results 
            if r["canonical_smiles"] not in testset_smiles
        ]
        logging.info(f"Retained {len(results)} reactions after excluding testset")
    

    return results


if __name__ == "__main__":
    # Setup paths and logging
    input_file, output_file, testset_file = _setup()

    # Load input data
    reactions = load_data(input_file)
    logging.info(f"Loaded {len(reactions)} reactions from {input_file}")

    testset_smiles = load_testset_reactions(testset_file)
    logging.info(f"Loaded {len(testset_smiles)} reactions from {testset_file}")

    # Process reactions
    results = process_reactions(
        reactions,
        testset_smiles
    )

    # Separate by source (patent)
    results = separate_by_source(results)

    # Save results
    with gzip.open(output_file, 'wb') as f:
        pickle.dump(results, f)
    logging.info(f"Saved processed reactions to {output_file}")

import os
import gzip
import json
import pickle
import torch
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions as Reactions
from rdchiral.main import rdchiralRunText

def print_if_verbose(msg, verbose):
    if verbose:
        print(msg)

def load_data(file_path):
    files_to_func = {
        ".jsonl.gz": load_jsonl_gz,
        ".json.gz": load_json_gz,
        ".txt.gz": load_txt_gz,
        ".pkl.gz": load_pickle_gz,
        ".jsonl": load_jsonl,
        ".json": load_json,
    }
    file_path = str(file_path) 
    for ext, func in files_to_func.items():
        if file_path.endswith(ext):
            return func(file_path)

    raise ValueError(f"Cannot be used for this file type: {file_path}")

def load_jsonl_gz(file_path):
    with gzip.open(file_path, 'rt') as f:
        return [json.loads(line) for line in f]

def load_json_gz(file_path):
    with gzip.open(file_path, 'rt') as f:
        return json.load(f)

def load_txt_gz(file_path):
    with gzip.open(file_path, 'rt') as f:
        return [line.strip() for line in f]

def load_pickle_gz(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)



def setup_logger(filename, log_dir = '../logs'):
    os.makedirs(log_dir, exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    log_file = f"{log_dir}/{filename}.{dt}"
    logging.basicConfig(
        filename=log_file, level=logging.DEBUG,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p'
    )
    print("Log file:", log_file)


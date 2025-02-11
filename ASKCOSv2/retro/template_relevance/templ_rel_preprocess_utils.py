import argparse
import csv
import json
import logging
import misc
import multiprocessing
import numpy as np
import os
import random
import templ_rel_parser
import time
from datetime import datetime
from utils import canonicalize_smarts, canonicalize_smiles
from concurrent.futures import TimeoutError
from pebble import ProcessPool
from rdchiral.template_extractor import extract_from_reaction
from rdchiral.initialization import rdchiralReaction, rdchiralReactants
from rdchiral.main import rdchiralRun
from rdkit import Chem, RDLogger
from scipy import sparse
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Tuple, Callable
from utils import load_templates_as_dict, save_templates_from_dict, mol_smi_to_count_fp
from multiprocessing import Pool
from collections import Counter

global TEMPLATES
global TEMPLATE_COUNTER

def get_tpl(task: Tuple[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
    i, rxn = task
    rxn_id = rxn["id"]
    r_smi, _, p_smi = rxn["rxn_smiles"].strip().split(">")

    reaction = {'_id': rxn_id, 'reactants': r_smi, 'products': p_smi}
    try:
        with misc.BlockPrint():
            template = extract_from_reaction(reaction)
        p_templ = canonicalize_smarts(template["products"])
        r_templ = canonicalize_smarts(template["reactants"])

        # Note: "reaction_smarts" is actually: p_temp >> r_temp!
        canon_templ = p_templ + '>>' + r_templ
    except:
        template = {}
        canon_templ = ""
    
    rxn_with_template = rxn
    rxn_with_template["canon_reaction_smarts"] = canon_templ

    rxn_with_template["intra_only"] = template.get("intra_only", False)
    rxn_with_template["dimer_only"] = template.get("dimer_only", False)

    return i, rxn_with_template

def dep(args):
    i, rxn = args
    if i % 10_000 == 0:
        print(" {} reactions processed/deduplicated.".format(i))
    r_smi, _, p_smi = rxn["rxn_smiles"].strip().split(">")
    canon_r_smi = canonicalize_smiles(r_smi, remove_atom_number=True, remove_isotope=False)
    canon_p_smi = canonicalize_smiles(p_smi, remove_atom_number=True, remove_isotope=False)
    canon_rxn_smi = f"{canon_r_smi}>>{canon_p_smi}"
    rxn["canon_reaction_smiles"] = canon_rxn_smi
    return rxn, canon_rxn_smi


def _deduplicate(rxns: Iterable[Dict[str, Any]], num_workers: int) -> List[Dict[str, Any]]:
    p = Pool(num_workers)

    dedupped_rxns = {}
    for rxn, canon_rxn_smi in tqdm(p.imap(dep, ((args) for args in enumerate(rxns)))):
        if canon_rxn_smi in dedupped_rxns:

            dedupped_rxns[canon_rxn_smi]['deduped_rxn_ids'].append(rxn['id'])
            continue
        else:
            rxn['deduped_rxn_ids'] = [rxn['id']]
            dedupped_rxns[canon_rxn_smi] = rxn

    return list(dedupped_rxns.values())

def _process_templates(rxns: List[Dict[str, Any]], 
                        func: Callable[Tuple[int, Dict[str, Any]], Tuple[int, Dict[str, Any]]],
                        max_workers: int=1, 
                       ) -> Tuple[List[Dict[str, Any]],
                                  Dict[str, Dict[str, Any]],
                                  int]:

    _start = time.time()
    rxns_with_template = []
    templates = {}
    failed_count = 0

    with ProcessPool(max_workers=max_workers) as pool:

        # Using pebble to add timeout, as rdchiral could hang
        future = pool.map(func, enumerate(rxns), timeout=10)
        iterator = future.result()

        # The while True - try/except/StopIteration is just pebble signature
        while True:
            try:
                i, rxn_with_template = next(iterator)
                if i > 0 and i % 10000 == 0:
                    logging.info(f"Processing {i}th reaction, "
                                 f"elapsed time: {time.time() - _start: .0f} s")

                rxn_id = rxn_with_template["id"]
                canon_reaction_smarts = rxn_with_template["canon_reaction_smarts"]

                if canon_reaction_smarts:
                        
                    intra_only = rxn_with_template["intra_only"]
                    dimer_only = rxn_with_template["dimer_only"]
                    references = rxn_with_template["deduped_rxn_ids"]

                    if canon_reaction_smarts in templates:
                        templates[canon_reaction_smarts]["count"] += 1 
                        templates[canon_reaction_smarts]["num_references"] += \
                            len(rxn_with_template["deduped_rxn_ids"])
                        templates[canon_reaction_smarts]["references"].extend(
                            rxn_with_template["deduped_rxn_ids"]
                        )
                        templates[canon_reaction_smarts]["intra_only"] &= intra_only
                        templates[canon_reaction_smarts]["dimer_only"] &= dimer_only
                        
                    else:
                        templates[canon_reaction_smarts] = {
                            "index": -1,    # placeholder, to be reset after sorting
                            "reaction_smarts": canon_reaction_smarts,
                            "count": 1,
                            "num_references": len(references),
                            "necessary_reagent": "",
                            "intra_only": intra_only,
                            "dimer_only": dimer_only,
                            "template_set": "",
                            "references": references,
                            "attributes": {
                                "ring_delta": 1.0,
                                "chiral_delta": 0
                            },
                            "_id": "-1"     # placeholder, to be reset after sorting
                        }
                else:
                    failed_count += 1
            except StopIteration:
                break
            except TimeoutError as error:
                logging.info(f"function call took more than {error.args} seconds.")
                failed_count += 1
                rxn_with_template = rxns[i]
                rxn_with_template["canon_reaction_smarts"] = ""
                rxn_with_template["intra_only"] = False
                rxn_with_template["dimer_only"] = False
            except Exception as e:
                logging.info(f"Unknown error for getting template. "
                            f"Error {e} for reaction {rxn_with_template}")
                failed_count += 1
                rxn_with_template = rxns[i]
                rxn_with_template["canon_reaction_smarts"] = ""
                rxn_with_template["intra_only"] = False
                rxn_with_template["dimer_only"] = False

            rxns_with_template.append(rxn_with_template)

    # pool.close()
    # pool.join()
    return rxns_with_template, templates, failed_count
                                

def _extract_templates(rxns: List[Dict[str, Any]],
                       max_workers: int=1
                       ) -> Tuple[List[Dict[str, Any]],
                                  Dict[str, Dict[str, Any]],
                                  int]:
    
    return _process_templates(rxns, get_tpl, max_workers)


def _sort_and_filter_templates(templates: Dict[str, Dict[str, Any]], 
                               template_set: str, min_freq: int
                               ) -> Dict[str, Dict[str, Any]]:
    sorted_templates = sorted(
        templates.items(),
        key=lambda _tup: _tup[1]["count"],
        reverse=True
    )
    filtered_templates = {}
    for i, (canon_templ, metadata) in enumerate(sorted_templates):
        if metadata["count"] < min_freq:
            break
        metadata["index"] = i
        metadata["_id"] = f"{template_set}_{i}"
        metadata["template_set"] = template_set
        filtered_templates[canon_templ] = metadata

    return filtered_templates
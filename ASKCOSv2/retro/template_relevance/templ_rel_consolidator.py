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
import shutil
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
from typing import Any, Dict, Iterable, List, Tuple
from utils import load_templates_as_dict, save_templates_from_dict, mol_smi_to_count_fp, save_reactions_from_list, load_reactions_as_list
from templ_rel_preprocess_utils import _deduplicate, _process_templates, _extract_templates, _sort_and_filter_templates
from multiprocessing import Pool
from collections import Counter

global SMARTS, SMARTS_COUNTER

RDLogger.DisableLog("rdApp.warning")

def get_consol_tpl(task: Tuple[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:

    i, rxn_with_template = task

    global SMARTS_COUNTER

    rxn_with_template["rdchiral_reaction_smarts"] = rxn_with_template["canon_reaction_smarts"]
    
    for consolidated_smarts, count in SMARTS_COUNTER.items():
        if consolidated_smarts: 
            # keep the most general template that gives the reactant
            if consolidated_smarts in rxn_with_template["smarts_recovering_reactant"]:
                rxn_with_template["canon_reaction_smarts"] = consolidated_smarts
                break
    else: 
        rxn_with_template["canon_reaction_smarts"] = ""

    return i, rxn_with_template


def _get_all_smarts_recovering_reactant(task: Tuple[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
    i, reaction = task

    canon_r_smi, _, canon_p_smi = reaction["canon_reaction_smiles"].split(">")
    can_r_set = set(canon_r_smi.split("."))
    prod = rdchiralReactants(canon_p_smi)

    smarts_recovering_reactant = [] # all smarts that can recover the reactant
    
    for template in SMARTS:
        try:
            outcomes = rdchiralRun(template, prod)
            for outcome in outcomes:
                can_outcome = canonicalize_smiles(outcome, remove_atom_number=True, remove_isotope=False)
                can_outcome_set = set(can_outcome.split("."))

                if can_outcome_set == can_r_set:
                    smarts = template.reaction_smarts[1:-1].replace(")>>(", ">>")
                    smarts_recovering_reactant.append(smarts)

        except: pass

    reaction["smarts_recovering_reactant"] = smarts_recovering_reactant
    
    return i, reaction


def _save_reactions_with_all_smarts(rxns_with_template: List[Dict[str, Any]], templates: Dict[str, Dict[str, Any]], 
                                reaction_data_path: str, max_workers: int=1, 
                                ) -> None:
    
    _start = time.time()

    global SMARTS

    SMARTS =  [t["reaction_smarts"] for t in templates.values()]
    logging.info(f"No of templates before consolidation: {len(SMARTS)}")
    SMARTS = [rdchiralReaction(
                    "("+smarts.replace(">>", ")>>(")+")"
                ) for smarts in SMARTS]
    
    logging.info("rdchiralReactions created, starting template consolidation, "
                f"time: {time.time() - _start: .2f} s")

    
    # load partially saved reactions
    saved_rxns_with_template = []
    if os.path.exists(reaction_data_path):
        with open(reaction_data_path, 'r') as f:
            saved_rxns_with_template = [json.loads(line.strip()) for line in f]

        logging.info(f"{reaction_data_path} exists")
        logging.info(f"Loaded {len(saved_rxns_with_template)} reactions with templates from {reaction_data_path}")
    else:
        logging.info(f"{reaction_data_path} does not exist, starting from scratch")
    
    num_processed=len(saved_rxns_with_template)

    with open(reaction_data_path, 'w') as f:

        for rxn in tqdm(saved_rxns_with_template, total=num_processed, desc="Saving reactions"):
            f.write(json.dumps(rxn) + "\n")
        
        if saved_rxns_with_template:
            logging.info(f"Saved {num_processed} loaded reactions to {reaction_data_path}")
            logging.info(f"{len(rxns_with_template) - num_processed} reactions left to process")
            logging.info(f"Last saved reaction {saved_rxns_with_template[-1]}")
        
        if len(rxns_with_template) - num_processed:
            logging.info(f"Starting processing from {num_processed}th reaction, {rxns_with_template[num_processed]}")

            with Pool(max_workers) as pool:
                for result in tqdm(pool.imap(_get_all_smarts_recovering_reactant, 
                                            enumerate(rxns_with_template[num_processed:],start=num_processed)), 
                                            total=len(rxns_with_template) - num_processed, desc="Applying all templates"):

                    try:
                        # get a list of the elements of results["reactant_recovery_templates"]
                        i, rxn_with_template = result
                        if i % 2000 == 0:
                            logging.info(f"Applying all smarts to {i}th reaction, "
                                        f"elapsed time: {time.time() - _start: .0f} s")
                    except StopIteration:
                        break
                    except Exception as e:
                        logging.info(f"Unknown error {e} for template consolidation.")
                        rxn_with_template = rxns_with_template[i]
                        rxn_with_template["smarts_recovering_reactant"] = []

            
                    f.write(json.dumps(rxn_with_template) + "\n")

    logging.info(f"Done applying all templates and saving to {reaction_data_path}, "
                f"time: {time.time() - _start: .2f} s")

    SMARTS = []


def _load_reactions_and_consolidated_templates(reaction_data_path: str, 
                                              max_workers: int=1,
                                              ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], int]:

    rxns_with_template = []

    templates = {}
    failed_count = 0 

    global SMARTS_COUNTER

    with open(reaction_data_path, 'r') as f:
        rxns_with_template = [json.loads(line.strip()) for line in f]
    
    logging.info(f"Loaded {len(rxns_with_template)} reactions with templates from {reaction_data_path}")

    all_smarts_recovering_reactant = [
        smarts for r in rxns_with_template for smarts in r["smarts_recovering_reactant"] if smarts
    ]
    all_smarts_recovering_reactant = sorted(all_smarts_recovering_reactant, key=lambda x: len(x))
    SMARTS_COUNTER = Counter(all_smarts_recovering_reactant)

    logging.info(f"No of unique smarts before consolidation: {len(SMARTS_COUNTER)}")

    rxns_with_template, templates, failed_count = \
        _process_templates(rxns_with_template, get_consol_tpl, max_workers=max_workers)

    return rxns_with_template, templates, failed_count


def _consolidate_templates(rxns_with_template: List[Dict[str, Any]], templates: Dict[str, Dict[str, Any]], 
                          reaction_data_path: str, max_workers: int=1, 
                          ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], int]:

    _start = time.time()

    _save_reactions_with_all_smarts(rxns_with_template, templates, reaction_data_path, max_workers=max_workers)
    rxns_with_template, templates, failed_count = \
        _load_reactions_and_consolidated_templates(reaction_data_path, max_workers=max_workers)

    logging.info(f"Done template consolidation, {len(templates)} templates, "
                f"time: {time.time() - _start: .2f} s")

    return rxns_with_template, templates, failed_count


class TemplRelConsolidator:
    """Class for Template Relevance template extraction & consolidation"""

    def __init__(self, args):
        self.args = args

        self.model_name = args.model_name
        self.data_name = args.data_name
        self.log_file = args.log_file
        self.all_reaction_file = args.all_reaction_file
        self.train_file = args.train_file
        self.val_file = args.val_file
        self.test_file = args.test_file
        self.processed_data_path = args.processed_data_path
        self.num_cores = args.num_cores
        self.min_freq = args.min_freq

        os.makedirs(self.processed_data_path, exist_ok=True)

        self.is_data_presplit = None

    def preprocess(self) -> None:
        self.check_data_format()
        self.extract_and_consolidate()
        
        assert all(os.path.exists(file) for file in [
            os.path.join(self.processed_data_path, "train_rxns_with_template.jsonl"),
            os.path.join(self.processed_data_path, "val_rxns_with_template.jsonl"),
            os.path.join(self.processed_data_path, "test_rxns_with_template.jsonl"),
            os.path.join(self.processed_data_path, "templates.jsonl"),
        ])

        self.save_nonconsol_data()

    def check_data_format(self) -> None:
        """
        Check that all files exists and the data format is correct for the
        first few lines
        """
        check_count = 100

        logging.info(f"Checking the first {check_count} entries for each file")
        assert os.path.exists(self.all_reaction_file), \
               f"The file with all reactions ({self.all_reaction_file}) " \
               f"needs to be supplied for template consolidation!"

        fn = self.all_reaction_file
        assert os.path.exists(fn), \
            f"{fn} does not exist, skipping format check"


        with open(fn, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for i, row in enumerate(csv_reader):
                if i > check_count:
                    break

                assert (c in row for c in ["id", "rxn_smiles"]), \
                    f"Error processing file {fn} line {i}, ensure columns 'id' " \
                    f"and 'rxn_smiles' are included!"

                reactants, reagents, products = row["rxn_smiles"].split(">")
                # simply ensures that SMILES can be parsed
                Chem.MolFromSmiles(reactants)
                Chem.MolFromSmiles(products)

        logging.info("Data format check passed")


    def extract_and_consolidate(self):
        _start = time.time()
        logging.info(f"Extracting templates from "
                     f"{self.all_reaction_file}..")
        
        template_file_path = os.path.join(self.processed_data_path, "templates_before_consol.jsonl")
        reaction_file_path = os.path.join(self.processed_data_path, "rxns_with_template_before_consol.jsonl")

        if os.path.exists(reaction_file_path) and os.path.exists(template_file_path):
            logging.info(f"Skipping template extraction, loading from {reaction_file_path}")
            templates = load_templates_as_dict(template_file_path) 
            logging.info(f"Loaded {len(templates)} templates from {template_file_path}")

            rxns_with_template = load_reactions_as_list(reaction_file_path)

            logging.info(f"Loaded {len(rxns_with_template)} reactions with templates from {reaction_file_path}")

        else:
            logging.info(f"Loading all reaction SMILES from {self.all_reaction_file}")
            with open(self.all_reaction_file, "r") as csv_file:
                csv_reader = csv.DictReader(csv_file)
                dedupped_rxns = _deduplicate(csv_reader, self.num_cores)

                # extract templates from reactions
            logging.info(f"Loaded all reaction SMILES and deduplicated. "
                        f"No of rxn after deduplication: {len(dedupped_rxns)}. "
                        f"Parallelizing extraction over {self.num_cores} cores")

            # save first reaction
            logging.info(f"First reaction: {dedupped_rxns[0]}")
            rxns_with_template, templates, failed_count = _extract_templates(
                dedupped_rxns, max_workers=self.num_cores)
            logging.info(f'No of rxn where template extraction failed: {failed_count}')

            # save templates 
            save_templates_from_dict(templates, template_file_path)

            # save reactions with templates
            save_reactions_from_list(rxns_with_template, reaction_file_path)


        # TODO: add template consolidation
        rxns_with_template, templates, failed_count = _consolidate_templates(
            rxns_with_template, templates, 
            os.path.join(self.processed_data_path, "all_rxns_with_template.jsonl"), 
            max_workers=self.num_cores)


        # filter templates by min_freq and save
        filtered_templates = _sort_and_filter_templates(
            templates, template_set = self.data_name, min_freq=self.min_freq)
        template_file = os.path.join(self.processed_data_path, "templates.jsonl")
        save_templates_from_dict(filtered_templates, template_file)

        # filter reactions by templates
        rxns_with_template = [
            rxn_with_template for rxn_with_template in rxns_with_template
            if rxn_with_template["canon_reaction_smarts"] and
            rxn_with_template["canon_reaction_smarts"] in filtered_templates
        ]

        # split
        split_ratio = [float(r) for r in self.args.split_ratio.split(":")]
        split_ratio = [val/sum(split_ratio) for val in split_ratio]
        assert len(split_ratio) == 3
        random.shuffle(rxns_with_template)


        train_count = int(len(rxns_with_template) * split_ratio[0])
        val_count = int(len(rxns_with_template) * split_ratio[1])
        train_rxns = rxns_with_template[:train_count]
        val_rxns = rxns_with_template[train_count:train_count+val_count]
        test_rxns = rxns_with_template[train_count+val_count:]

        for rxns, phase, file_name in [(train_rxns, "train", self.train_file),
                        (val_rxns, "val", self.val_file),
                        (test_rxns, "test", self.test_file)]:
            ofn = os.path.join(self.processed_data_path, f"{phase}_rxns_with_template.jsonl")
            
            with open(ofn, "w") as of:
                for rxn in rxns:
                    of.write(f"{json.dumps(rxn)}\n")

            logging.info(f"Saving {len(rxns)} {phase} reactions to {file_name}")
            with open(file_name, "w") as of:
                writer = csv.DictWriter(of, fieldnames=["id", "rxn_smiles"])
                writer.writeheader()
                for rxn in rxns:
                    writer.writerow({
                        "id": rxn["id"], 
                        "rxn_smiles": rxn["rxn_smiles"]
                    })
        

        logging.info(f"Done template extraction, filtering, and splitting, "
                    f"time: {time.time() - _start: .2f} s")

            

    def save_nonconsol_data(self):
        
        _start = time.time()
        nonconsol_data_name = self.data_name.replace("consol", "nonconsol")
        nonconsol_processed_data_path = self.processed_data_path.replace('consol', 'nonconsol')
        os.makedirs(nonconsol_processed_data_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.train_file.replace('consol', 'nonconsol')), exist_ok=True)

        # copy the template files
        template_file = os.path.join(self.processed_data_path, "templates_before_consol.jsonl")
        output_template_file = os.path.join(nonconsol_processed_data_path, "templates.jsonl")

        templates = load_templates_as_dict(template_file)
        filtered_templates = _sort_and_filter_templates(
                templates, template_set = nonconsol_data_name, min_freq=self.min_freq)
        save_templates_from_dict(filtered_templates, output_template_file)


        logging.info(f"Copied & sorted templates from {template_file} to {output_template_file}")


        for phase, file_name in [("train", self.train_file),
                                    ("val", self.val_file),
                                    ("test", self.test_file)]:

            # copy the raw csv files
            shutil.copy(file_name, file_name.replace('consol', 'nonconsol'))
            logging.info(f"Coppied {file_name} to {file_name.replace('consol', 'nonconsol')}")
            
            # copy the jsonl files
            ifn = os.path.join(self.processed_data_path, f"{phase}_rxns_with_template.jsonl")
            ofn = os.path.join(nonconsol_processed_data_path, f"{phase}_rxns_with_template.jsonl")
            with open(ifn, "r") as ifile, open(ofn, "w") as ofile:
                for line in tqdm(ifile, desc=f"Copying reactions with templates for {phase}"):
                    reaction = json.loads(line)
                    reaction["canon_reaction_smarts"] = reaction["rdchiral_reaction_smarts"]
                    ofile.write(json.dumps(reaction) + "\n")

            logging.info(f"Saved reactions from {ifn} to {ofn}")

        logging.info("Done saving non-consolidated data, "
                    f"time: {time.time() - _start: .2f} s")



if __name__ == "__main__":
    parser = argparse.ArgumentParser("template_relevance")
    templ_rel_parser.add_model_opts(parser)
    templ_rel_parser.add_preprocess_opts(parser)
    args, unknown = parser.parse_known_args()

    # logger setup
    RDLogger.DisableLog("rdApp.warning")
    os.makedirs("./logs/consolidation", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    args.log_file = f"./logs/consolidation/{args.log_file}.{dt}"
    logger = misc.setup_logger(args.log_file)
    misc.log_args(args, message="Logging arguments")

    start = time.time()
    random.seed(args.seed)

    processor = TemplRelConsolidator(args)
    processor.preprocess()


    logging.info(f"Consolidation done, "
                f"time: {time.time() - start: .2f} s")

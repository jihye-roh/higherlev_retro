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
from rdkit import Chem, RDLogger
from scipy import sparse
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Tuple
from utils import load_templates_as_dict, save_templates_from_dict, mol_smi_to_count_fp
from templ_rel_preprocess_utils import _deduplicate, _process_templates, _extract_templates, _sort_and_filter_templates
from multiprocessing import Pool


def _gen_product_fp(task: Tuple[str, int, int]) -> Tuple[sparse.csr_matrix, str]:
    line, radius, fp_size = task
    rxn_with_template = json.loads(line.strip())
    p_smi = rxn_with_template["rxn_smiles"].split(">")[-1]

    p_smi = canonicalize_smiles(p_smi, remove_atom_number=True, remove_isotope=False)
    try:
        product_fp = mol_smi_to_count_fp(mol_smi=p_smi, radius=radius, fp_size=fp_size)
    except:
        logging.info(f"Error when converting smi to count fingerprint. "
                     f"Setting it to zero vector.")
        count_fp = np.zeros((1, fp_size), dtype=np.int32)
        product_fp = sparse.csr_matrix(count_fp, dtype="int32")

    canon_reaction_smarts = rxn_with_template["canon_reaction_smarts"]

    return product_fp, canon_reaction_smarts


class TemplRelProcessor:
    """Class for Template Relevance Preprocessing"""

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
        self.is_data_processed = None

    def preprocess(self) -> None:
        
        # If self.use_processed_data, then use processed files 
        # "train/val/test_rxns_with_template.jsonl" and "templates.jsonl" exist
        # in the processed_data_path directory
        
        self.check_processed_data()

        if self.is_data_processed:
            logging.info(f"Processed data exists in {self.processed_data_path}, skipping template extraction")
        else:
            self.check_data_format()
            if self.is_data_presplit:
                logging.info("Processed data does not exist. Data is prespplit. Extracting templates")
                self.extract_templates_for_all_split()
            else:
                logging.info("Processed data does not exist. Data is prespit. Extracting templates and splitting data")
                self.extract_templates_and_split()
            
        assert all(os.path.exists(file) for file in [
            os.path.join(self.processed_data_path, "train_rxns_with_template.jsonl"),
            os.path.join(self.processed_data_path, "val_rxns_with_template.jsonl"),
            os.path.join(self.processed_data_path, "test_rxns_with_template.jsonl"),
            os.path.join(self.processed_data_path, "templates.jsonl"),
        ])
        self.featurize()

    def check_processed_data(self) -> None:

        logging.info(f"Checking if processed data exists in {self.processed_data_path}")
        
        self.is_data_processed = all(os.path.exists(file) for file in [
            os.path.join(self.processed_data_path, "train_rxns_with_template.jsonl"),
            os.path.join(self.processed_data_path, "val_rxns_with_template.jsonl"),
            os.path.join(self.processed_data_path, "test_rxns_with_template.jsonl"),
            os.path.join(self.processed_data_path, "templates.jsonl"),
        ])



    def check_data_format(self) -> None:
        """
        Check that all files exists and the data format is correct for the
        first few lines
        """
        check_count = 100

        logging.info(f"Checking the first {check_count} entries for each file")
        assert os.path.exists(self.all_reaction_file) or \
               os.path.exists(self.train_file), \
               f"Either the train file ({self.train_file}) " \
               f"or the file with all reactions ({self.all_reaction_file}) " \
               f"needs to be supplied!"

        for fn in [self.train_file, self.val_file, self.test_file,
                   self.all_reaction_file]:
            if not os.path.exists(fn):
                logging.info(f"{fn} does not exist, skipping format check")
                continue

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

        self.is_data_presplit = os.path.isfile(self.train_file)

    def extract_templates_and_split(self):
        _start = time.time()
        logging.info(f"Data is not presplit. Extracting templates from "
                     f"{self.all_reaction_file}..")

        with open(self.all_reaction_file, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            dedupped_rxns = _deduplicate(csv_reader, self.num_cores)

            # extract templates from reactions
            logging.info(f"Loaded all reaction SMILES and deduplicated."
                        f"Parallelizing extraction over {self.num_cores} cores")
            rxns_with_template, templates, failed_count = _extract_templates(
                dedupped_rxns, max_workers=self.num_cores)
            logging.info(f'No of rxn where template extraction failed: {failed_count}')

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

            for rxns, phase in [(train_rxns, "train"),
                                (val_rxns, "val"),
                                (test_rxns, "test")]:
                ofn = os.path.join(self.processed_data_path, f"{phase}_rxns_with_template.jsonl")
                with open(ofn, "w") as of:
                    for rxn in rxns:
                        of.write(f"{json.dumps(rxn)}\n")

            logging.info(f"Done template extraction, filtering and splitting, "
                        f"time: {time.time() - _start: .2f} s")

    def extract_templates_for_all_split(self):
        _start = time.time()
        logging.info(f"Data is presplit. Extracting templates from "
                     f"{self.train_file}..")

        with open(self.train_file, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            dedupped_rxns = _deduplicate(csv_reader, args.num_cores)

            # extract templates from train reactions
            logging.info(f"Loaded all train reaction SMILES and deduplicated."
                        f"Parallelizing extraction over {self.num_cores} cores")
            train_rxns_with_template, templates, failed_count = _extract_templates(
                dedupped_rxns, max_workers=self.num_cores)
            logging.info(f'No of rxn where template extraction failed: {failed_count}')

            # filter templates by min_freq and save
            filtered_templates = _sort_and_filter_templates(
                templates, min_freq=self.min_freq)
            template_file = os.path.join(self.processed_data_path, "templates.jsonl")
            save_templates_from_dict(filtered_templates, template_file)

            # filter reactions by templates
            train_rxns_with_template = [
                rxn_with_template for rxn_with_template in train_rxns_with_template
                if rxn_with_template["canon_reaction_smarts"] and
                rxn_with_template["canon_reaction_smarts"] in filtered_templates
            ]

            # save filtered train reactions
            ofn = os.path.join(self.processed_data_path, "train_rxns_with_template.jsonl")
            with open(ofn, "w") as of:
                for rxn in train_rxns_with_template:
                    of.write(f"{json.dumps(rxn)}\n")

            # for val and test we'll keep all reactions
            for file, phase in [(self.val_file, "val"),
                                (self.test_file, "test")]:
                with open(file, "r") as csv_file:
                    csv_reader = csv.DictReader(csv_file)

                    # extract templates from train reactions
                    logging.info(f"Loaded all reaction SMILES from {file}."
                                f"Parallelizing extraction over {self.num_cores} cores")
                    rxns_with_template, _, failed_count = _extract_templates(
                        list(csv_reader), max_workers=self.num_cores)
                    logging.info(f'No of rxn where template extraction failed: {failed_count}')

                    ofn = os.path.join(self.processed_data_path, f"{phase}_rxns_with_template.jsonl")
                    with open(ofn, "w") as of:
                        for rxn in rxns_with_template:
                            of.write(f"{json.dumps(rxn)}\n")

            logging.info(f"Done template extraction, time: {time.time() - _start: .2f} s")

    def featurize(self):
        logging.info("(Re-)loading templates for featurization")
        templates = load_templates_as_dict(
            template_file=os.path.join(self.processed_data_path, "templates.jsonl")
        )
        for phase in ["train", "val", "test"]:
            fn = os.path.join(self.processed_data_path,
                              f"{phase}_rxns_with_template.jsonl")
            logging.info(f"Loading rxns_with_template from {fn} "
                         f"and featurizing over {self.num_cores} cores")
            pool = multiprocessing.Pool(self.num_cores)

            product_fps = []
            labels = []

            with open(fn, "r") as f:
                lines = f.readlines()
            tasks = [(line, self.args.radius, self.args.fp_size)
                     for line in lines]
            for result in tqdm(pool.imap(_gen_product_fp, tasks),
                               total=len(tasks),
                               desc="Processing line "):
                product_fp, canon_reaction_smarts = result
                product_fps.append(product_fp)

                if canon_reaction_smarts and canon_reaction_smarts in templates:
                    label = templates[canon_reaction_smarts]["index"]
                else:
                    label = -1
                labels.append(label)

            pool.close()
            pool.join()

            product_fps = sparse.vstack(product_fps)
            sparse.save_npz(
                os.path.join(self.processed_data_path, f"product_fps_{phase}.npz"),
                product_fps
            )
            np.save(
                os.path.join(self.processed_data_path, f"labels_{phase}.npy"),
                np.asarray(labels)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("template_relevance")
    templ_rel_parser.add_model_opts(parser)
    templ_rel_parser.add_preprocess_opts(parser)
    args, unknown = parser.parse_known_args()

    # logger setup
    RDLogger.DisableLog("rdApp.warning")
    os.makedirs("./logs/preprocess", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    args.log_file = f"./logs/preprocess/{args.log_file}.{dt}"
    logger = misc.setup_logger(args.log_file)
    misc.log_args(args, message="Logging arguments")

    start = time.time()
    random.seed(args.seed)

    processor = TemplRelProcessor(args)
    processor.preprocess()

    logging.info(f"Preprocessing done, total time: {time.time() - start: .2f} s")

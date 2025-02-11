import csv
import misc
import utils
import logging
import multiprocessing
import numpy as np
import os
import pandas as pd
import random
import templ_rel_parser
from datetime import datetime
import argparse
import torch
import torch.nn as nn
from collections import Counter
from dataset import FingerprintDataset, init_loader
from rdchiral.main import rdchiralReaction, rdchiralReactants, rdchiralRun
from rdkit import Chem
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List
import templ_rel_scorer
from utils import canonicalize_smiles

global G_templates_filtered, G_preds

def gen_precs(task):
    global G_templates_filtered, G_preds

    i, prod_smi_nomap, phase_topk = task

    templates_filtered = G_templates_filtered
    preds = G_preds

    # generate predictions from templates
    precursors, dup_count = [], 0
    pred_temp_idxs = preds[i]
    applied_temps = 0

    seen, seen_per_temp = [], []
    for k, idx in enumerate(pred_temp_idxs):
        template_k_precursors=[]
        template = templates_filtered[idx]["reaction_smarts"]
        try:
            rxn = rdchiralReaction('('+template.replace('>>', ')>>(')+')')
            prod = rdchiralReactants(prod_smi_nomap)
            prod_mapped = Chem.MolToSmiles(prod.reactants)
            #precs = rdchiralRun(rxn, prod)
            precs = rdchiralRun(rxn, prod)
            if precs:
                applied_temps += 1
           

            precursors.extend(precs)
            for prec in precs:
                prec = canonicalize_smiles(prec, remove_atom_number = True, remove_isotope = False)
                if prec not in seen:
                    seen.append(prec)
                    template_k_precursors.append(prec)
                else:
                    dup_count += 1
        
        except KeyError as e:
            logging.info(f'Key error {e} applying template {template} to product {prod_smi_nomap}')
            logging.info(f"precs: {precs}")
            # logging.info(f"mapped_precs: {mapped_precs}")
        except Exception as e: 
            logging.info(f'Error {e} applying template {template} to product {prod_smi_nomap}')
        if template_k_precursors: 
            seen_per_temp.append(f"{k+1};"+"_".join(template_k_precursors))
        if len(seen)>=phase_topk: break
    
    seen_per_temp.extend([None] * max(0, phase_topk - len(seen_per_temp)))
    
    return precursors, seen_per_temp, dup_count


def analyse_proposed(
    prod_smiles_phase: List[str],
    prod_smiles_mapped_phase: List[str],
    proposals_phase: Dict[str, List[str]],
):
    proposed_counter = Counter()
    total_proposed, min_proposed, max_proposed = 0, float('+inf'), float('-inf')
    key_count = 0
    for key, mapped_key in zip(prod_smiles_phase, prod_smiles_mapped_phase):
        precursors = proposals_phase[mapped_key]
        precursors_count = len(precursors)
        total_proposed += precursors_count
        if precursors_count > max_proposed:
            max_proposed = precursors_count
            prod_smi_max = key
        if precursors_count < min_proposed:
            min_proposed = precursors_count
            prod_smi_min = key

        proposed_counter[key] = precursors_count
        key_count += 1

    logging.info(f'Average precursors proposed per prod_smi (dups removed): {total_proposed / key_count}')
    logging.info(f'Min precursors: {min_proposed} for {prod_smi_min}')
    logging.info(f'Max precursors: {max_proposed} for {prod_smi_max})')

    logging.info(f'\nMost common 20:')
    for i in proposed_counter.most_common(20):
        logging.info(f'{i}')
    logging.info(f'\nLeast common 20:')
    for i in proposed_counter.most_common()[-20:]:
        logging.info(f'{i}')
    return


class TemplRelPredictor:
    """Class for TemplRel Predicting"""

    def __init__(self, args):
        self.args = args

        self.model_name = args.model_name
        self.data_name = args.data_name
        self.log_file = args.log_file
        self.processed_data_path = args.processed_data_path
        self.model_path = args.model_path
        self.test_output_path = args.test_output_path
        self.test_file = args.test_file
        os.makedirs(self.test_output_path, exist_ok=True)

        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.templates_filtered = []
        self.init_templates()
        self.build_predict_model()

    def init_templates(self):
        templates_file = os.path.join(self.processed_data_path, "templates.jsonl")
        logging.info(f'Loading templates from file: {templates_file}')
        self.templates_filtered, _ = utils.load_templates_as_list(templates_file)
        logging.info(f"Loaded {len(self.templates_filtered)} templates")
        if len(self.templates_filtered)<self.args.max_num_templ:
            self.args.max_num_templ = len(self.templates_filtered)

    def build_predict_model(self):
        # --------------------#
        checkpoint_file = os.path.join(self.model_path, "model_latest.pt")
        print(f"Building model and loading from {checkpoint_file}")

        self.args.load_from = checkpoint_file
        self.args.local_rank = -1
        # Note: the model will be built using pretraining args
        self.model, _ = utils.get_model(self.args, device=self.device)
        self.model.eval()

    def predict(self):

        if os.path.exists(os.path.join(self.test_output_path, "predictions.csv")):
            logging.info("predictions.csv already exist, skipping predicting")
        else:
            logging.info("Predicting on test set")
            self.infer_all()
            self.raw_to_processed()
            self.compile_into_csv()
        templ_rel_scorer.score_main(self.args)

    def infer_all(self):
        """Actual file-based predicting, adapted from infer_all.py"""
        dataset = FingerprintDataset(
            os.path.join(self.processed_data_path, "product_fps_test.npz"),
            os.path.join(self.processed_data_path, "labels_test.npy")
        )
        loader = DataLoader(dataset, batch_size=self.args.test_batch_size, shuffle=False)
        del dataset

        preds = []
        # loader = tqdm(loader, desc="Predicting on test")
        self.model.eval()
        with torch.no_grad():
            for data in loader:
                inputs, labels = data             # we don't need labels & idxs
                inputs = inputs.to(self.device).float()
                labels = labels.to(self.device)

                outputs = self.model(inputs).squeeze()
                _, acc = self.model.get_loss(logits=outputs, target=labels)
                # print("top 1",acc.item())
                outputs = nn.Softmax(dim=1)(outputs)

                preds.append(torch.topk(outputs, k=self.args.max_num_templ, dim=1)[1])
                # print("preds", preds[-1])

            preds = torch.cat(preds, dim=0).squeeze(dim=-1).cpu().numpy()
        logging.info(f'preds.shape: {preds.shape}')
        np.save(os.path.join(self.test_output_path, "raw_outputs_on_test.npy"), preds)
        logging.info(f'Saved preds of test as npy!')

    def raw_to_processed(self):
        processed_test_file = os.path.join(self.processed_data_path, "processed_test.csv")
        if not os.path.exists(processed_test_file): 
            logging.info(f"Raw to Processed, "
            "reading in {self.test_file} and saving to {processed_test_file}")
            test_raw_csv = pd.read_csv(self.test_file)
            new_ = test_raw_csv["rxn_smiles"].str.strip().str.split(">>", expand=True)
            new_[0] = new_[0].apply(
                lambda x: utils.canonicalize_smiles(
                    x, remove_atom_number=True, remove_isotope = False
                )
            )
            new_[1] = new_[1].apply(
                lambda x: utils.canonicalize_smiles(
                    x, remove_atom_number=True, remove_isotope = False
                )
            )
            # assert utils.canonicalize_smiles(test_raw_csv.loc[0, "rxn_smiles"].strip().split(">>")[0]) == new_.iloc[0, 1], "Did not flip"
            new_.to_csv(processed_test_file, index=False)
        else:
            logging.info("\nFile processed_test.csv already existed!\n")

    def compile_into_csv(self):
        global G_templates_filtered, G_preds

        logging.info("Compiling into predictions.csv")

        preds = np.load(os.path.join(self.test_output_path, "raw_outputs_on_test.npy"))

        # load mapped_rxn_smi
        with open(self.test_file, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            clean_rxnsmi_phase = [row["rxn_smiles"].strip()
                                  for row in csv_reader]

        proposals_data = pd.read_csv(
            os.path.join(self.processed_data_path, "processed_test.csv"),
            index_col=None, dtype='str'
        )

        phase_topk = self.args.topk
        tasks = []
        for i in range(len(clean_rxnsmi_phase)):        # build tasks
            tasks.append((i, proposals_data.iloc[i, 1], phase_topk))

        proposals_phase = {}
        proposed_precs_phase, prod_smiles_phase, rcts_smiles_phase = [], [], []
        proposed_precs_phase_withdups = []              # true representation of model predictions, for calc_accs()
        prod_smiles_mapped_phase = []                   # helper for analyse_proposed()
        dup_count = 0

        G_templates_filtered = self.templates_filtered
        G_preds = preds

        num_cores = self.args.num_cores
        logging.info(f'Parallelizing over {num_cores} cores')
        pool = multiprocessing.Pool(num_cores)

        for i, result in enumerate(tqdm(pool.imap(gen_precs, tasks),
                                        total=len(clean_rxnsmi_phase),
                                        desc='Generating predicted reactants')):
            precursors, seen, this_dup = result
            dup_count += this_dup

            prod_smi = clean_rxnsmi_phase[i].split('>>')[-1]
            prod_smiles_mapped_phase.append(prod_smi)

            prod_smi_nomap = proposals_data.iloc[i, 1]
            prod_smiles_phase.append(prod_smi_nomap)

            rcts_smi_nomap = proposals_data.iloc[i, 0]
            rcts_smiles_phase.append(rcts_smi_nomap)

            proposals_phase[prod_smi] = precursors
            proposed_precs_phase.append(seen)
            proposed_precs_phase_withdups.append(precursors)

        pool.close()
        pool.join()

        dup_count /= len(clean_rxnsmi_phase)
        logging.info(f'Avg # duplicates per product: {dup_count}')

        analyse_proposed(
            prod_smiles_phase,
            prod_smiles_mapped_phase,
            proposals_phase,        # this func needs this to be a dict {mapped_prod_smi: proposals}
        )

        zipped = []
        for rxn_smi, prod_smi, rcts_smi, proposed_rcts_smi in zip(
                clean_rxnsmi_phase,
                prod_smiles_phase,
                rcts_smiles_phase,
                proposed_precs_phase,
        ):
            result = [prod_smi]
            result.extend(proposed_rcts_smi)
            zipped.append(result)

        logging.info('Zipped all info for each rxn_smi into a list for dataframe creation!')

        temp_dataframe = pd.DataFrame(data={'zipped': zipped})
        phase_dataframe = pd.DataFrame(
            temp_dataframe['zipped'].to_list(),
            index=temp_dataframe.index
        )

        proposed_col_names = [f'cand_precursor_{i}' for i in range(1, self.args.topk + 1)]
        col_names = ['prod_smi']
        col_names.extend(proposed_col_names)
        phase_dataframe.columns = col_names

        phase_dataframe.to_csv(os.path.join(self.test_output_path, "predictions.csv"), index=False)
        logging.info(f'Saved proposals of test as CSV!')



if __name__ == "__main__":
    parser = argparse.ArgumentParser("template_relevance")
    templ_rel_parser.add_model_opts(parser)
    templ_rel_parser.add_preprocess_opts(parser)
    templ_rel_parser.add_train_opts(parser)
    templ_rel_parser.add_predict_opts(parser)
    args, unknown = parser.parse_known_args()

    # logger setup
    os.makedirs("./logs/predict", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    args.log_file = f"./logs/predict/{args.log_file}.{dt}.log"
    logger = misc.setup_logger(args.log_file)

    utils.set_seed(args.seed)
    misc.log_args(args, message="Logging predicting args")

    predictor = TemplRelPredictor(args)
    predictor.predict()

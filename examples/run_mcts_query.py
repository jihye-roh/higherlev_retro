import sys
sys.path.append("..")

import copy
import numpy as np
import requests
import json
import os
import logging
import time
import gzip
import argparse
from datetime import datetime
from multiprocessing import Pool

from misc import setup_logger   

QUERY_TEMPLATE = {
    "smiles": "",
    "expand_one_options": {
        "template_count": 25,
        "max_cum_template_prob": 1.0,
        "banned_chemicals": [],
        "banned_reactions": [],
        "retro_backend_options": [
            {
                "retro_backend": "template_relevance",
                "retro_model_name": "",
                "max_num_templates": 25,
                "max_cum_prob": 1.0,
                "attribute_filter": []
            }
        ],
        "use_fast_filter": False,
        "retro_rerank_backend": "model_score", 
        "cluster_precursors": False,
        "extract_template": False,
        "return_reacting_atoms": False,
        "selectivity_check": False
    },
    "build_tree_options": {
        "expansion_time": 30,
        "max_iterations": 500,
        "max_branching": 25,
        "max_depth": 8,
        "exploration_weight": 1,
        "return_first": False,
        "max_trees": 500,
        "buyable_logic": "and",
        "max_ppg_logic": "and",
        "max_ppg": 100.0,
        "max_scscore_logic": "none",
        "max_scscore": 0,
        "chemical_property_logic": "none",
        "max_chemprop_c": 0,
        "max_chemprop_n": 0,
        "max_chemprop_o": 0,
        "max_chemprop_h": 0,
        "custom_buyables": []
    },
    "enumerate_paths_options": {
        "path_format": "json",
        "json_format": "nodelink",
        "sorting_metric": "number_of_reactions",
        "validate_paths": True,
        "score_trees": False,
        "cluster_trees": False,
        "cluster_method": "hdbscan",
        "paths_only": False,
        "max_paths": 200
    },
}

def get_query_template(model_name = "reaxys", max_depth = 5, exp_weight = 1, max_num_templates=25):

    template = QUERY_TEMPLATE.copy()
    template["expand_one_options"]["retro_backend_options"][0]["retro_model_name"] = model_name
    template["expand_one_options"]["template_count"] = max_num_templates
    template["expand_one_options"]["retro_backend_options"][0]["max_num_templates"] = max_num_templates
    template["build_tree_options"]["max_branching"] = max_num_templates
    template["build_tree_options"]["max_depth"] = max_depth
    template["build_tree_options"]["exploration_weight"] = exp_weight
    

    return template

def run_mcts_for_single_smiles(input_data):

    HOST = "0.0.0.0"
    PORT = "9100"

    i, data = input_data

    smiles = data["smiles"]
    start = time.time()
    try:
        resp = requests.post(
            url=f"http://{HOST}:{PORT}/api/tree-search/mcts/call-sync-without-token",
            json=data
        ).json()
        return resp, i, smiles
    except requests.exceptions.RequestException as e:
        logging.info(f"Error in request. SMILES: {smiles}, error: {e}, time for running: {time.time()-start: .2f} s")
        print(f"Error in request. SMILES: {smiles}, error: {e}")

    return {}, i, smiles

def get_args():

    args = argparse.ArgumentParser()
    args.add_argument("--data", type=str, default="example")
    args.add_argument("--model_name", type=str, default="uspto_higher-level_consol")
    args.add_argument("--max_depth", type=int, default=8)
    args.add_argument("--exp_weight", type=int, default=1)
    args.add_argument("--num_workers", type=int, default=1) 
    args.add_argument("--max_num_templates", type=int, default=25)
    args = args.parse_args()

    return args


def get_data_list(args):

    with open(f"./data/targets/{args.data}.txt", "r") as f:
        smiles = f.readlines()
    smiles = [smi.split(",")[0].strip() for smi in smiles]

    data_list = []
    query_template = get_query_template(args.model_name, args.max_depth, args.exp_weight)
    for i, smi in enumerate(smiles):
        data = copy.deepcopy(query_template)
        data["smiles"] = smi
        data_list.append((i, data))

    print(f"{len(data_list)} molecules to query, "
          f"first molecule {data_list[0][1]['smiles']}")
    
    return data_list



def main():

    logs_dir = "./logs/mcts"
    mcts_results_dir = "./results/mcts/results_by_target"
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(mcts_results_dir, exist_ok=True)
    

    args = get_args()
    data_list = get_data_list(args)

    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    ofn = f"{args.data}_{args.model_name}_depth{args.max_depth}_{args.max_num_templates}"
    log_file = f"{logs_dir}/{ofn}.{dt}"
    setup_logger(log_file)

    i_result_dir = f"{mcts_results_dir}/{ofn}/"
    os.makedirs(i_result_dir, exist_ok=True)

    query_template = get_query_template(
        args.model_name, 
        args.max_depth, 
        args.exp_weight, 
        args.max_num_templates
    )

    logging.info(f"Querying MCTS for {args.data}")
    logging.info(json.dumps(query_template))        

    with Pool(args.num_workers) as p:
        for result in p.imap_unordered(run_mcts_for_single_smiles, data_list):
            resp, i, smi = result
            with gzip.open(os.path.join(i_result_dir, f"target_{i}_result.json.gz"), "wt") as f1:
                f1.write(json.dumps(resp))

            try:
                first_path_time = resp["result"]["stats"]["first_path_time"]
                total_iterations = resp["result"]["stats"]["total_iterations"]
                build_time = resp["result"]["stats"]["build_time"]
                logging.info(f"i: {i}, SMILES: {smi}, total_iterations: {total_iterations}, "
                    f"first_path_time: {first_path_time: .2f} s,  build_time: {build_time: .2f} s")

            except KeyError:
                logging.info(f"i: {i}, SMILES: {smi}, no path found")

    logging.info("Done running MCTS")


if __name__ == "__main__":
    main()
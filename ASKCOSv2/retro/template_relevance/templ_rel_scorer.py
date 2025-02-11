import argparse
import csv
import logging
import multiprocessing
import numpy as np
import os
import sys
from datetime import datetime
from rdkit import RDLogger
from tqdm import tqdm
from rdkit import Chem
from utils import misc, canonicalize_smiles

global G_predictions


def csv2kv(_args):
    prediction_row, n_best = _args
    k = canonicalize_smiles(prediction_row["prod_smi"], remove_atom_number=True, remove_isotope=False)
    v = []

    for i in range(n_best):
        try:
            prediction = prediction_row[f"cand_precursor_{i + 1}"]
        except KeyError:
            break

        if not prediction or prediction == "9999":          # padding
            break

        template_rank, prediction = prediction.split(";")
        prediction = '_'.join([
            canonicalize_smiles(p, remove_atom_number=True, remove_isotope=False) 
            for p in prediction.split('_')])
        v.append(prediction)

    return k, v


# def match_results(_args):
#     global G_predictions
#     test_row, n_best = _args
#     predictions = G_predictions

#     accuracy = np.zeros(n_best, dtype=np.float32)

#     gt, reagent, prod = test_row["rxn_smiles"].strip().split(">")
#     k = canonicalize_smiles(prod)

#     if k not in predictions:
#         logging.info(f"Product {prod} not found in predictions (after canonicalization), skipping")
#         return accuracy

#     gt = canonicalize_smiles(gt)
#     for j, prediction in enumerate(predictions[k]):
#         if prediction == gt:
#             accuracy[j:] = 1.0
#             break

    # return accuracy

def match_results(_args):
    global G_predictions
    test_row, n_best, is_optimistic_ranking = _args
    #print("Scorer, optimistic ranking", is_optimistic_ranking)
    predictions = G_predictions

    accuracy = np.zeros(n_best, dtype=np.float32)

    gt, reagent, prod = test_row["rxn_smiles"].strip().split(">")
    k = canonicalize_smiles(prod, remove_atom_number=True, remove_isotope=False)

    if k not in predictions:
        logging.info(f"Product {prod} not found in predictions (after canonicalization), skipping")
        return accuracy

    gt = canonicalize_smiles(gt, remove_atom_number=True, remove_isotope=False)
    seen_precursors = []
    for j, prediction in enumerate(predictions[k]):
        prediction_list = prediction.split('_') if prediction != "9999" else []
        if gt in prediction_list:

            if is_optimistic_ranking:
                correct_index = len(seen_precursors)
            else:
                correct_index = len(seen_precursors)+len(prediction_list)-1
                
            accuracy[correct_index:] = 1.0
            return accuracy
        seen_precursors.extend(prediction_list)
        if len(seen_precursors) >= n_best: break
        
    return accuracy


def score_main(args):
    """
        Adapted from Molecular Transformer
        Parallelized (210826 by ztu)
    """
    global G_predictions
    n_best = args.topk
    logging.info(f"Scoring predictions with model: {args.model_name}")

    # Load predictions and transform into a huge table {cano_prod: [cano_cand, ...]}
    args.prediction_file = os.path.join(args.test_output_path, "predictions.csv")
    logging.info(f"Loading predictions from {args.prediction_file}")
    predictions = {}
    p = multiprocessing.Pool(args.num_cores)

    with open(args.prediction_file, "r") as prediction_csv:
        prediction_reader = csv.DictReader(prediction_csv)
        for result in tqdm(p.imap(csv2kv,
                                  ((prediction_row, n_best) for prediction_row in prediction_reader))):
            k, v = result
            predictions[k] = v

    G_predictions = predictions

    p.close()
    p.join()
    p = multiprocessing.Pool(args.num_cores)        # re-initialize to see the global variable

    # Results matching
    logging.info(f"Matching against ground truth from {args.test_file}")
    logging.info(f"Optimistic ranking: {args.is_optimistic_ranking}")
    with open(args.test_file, "r") as test_csv:
        test_reader = csv.DictReader(test_csv)
        accuracies = p.imap(match_results,
                            ((test_row, n_best, args.is_optimistic_ranking) 
                            for test_row in test_reader))
        accuracies = np.stack(list(accuracies))

    p.close()
    p.join()

    # Log statistics
    mean_accuracies = np.mean(accuracies, axis=0)
    for n in range(n_best):
        logging.info(f"Top {n+1} accuracy: {mean_accuracies[n]}")

    # save mean_accuracies as a list
    if args.is_optimistic_ranking:
        output_file = os.path.join(args.test_output_path, "topk_accuracies_optimistic.csv")
    else:
        output_file = os.path.join(args.test_output_path, "topk_accuracies_pessimistic.csv")
    
    with open(output_file, "w") as output_csv:
        output_csv.write("topk,accuracy\n")
        for n in range(n_best):
            output_csv.write(f"{n+1},{mean_accuracies[n]}\n")

    

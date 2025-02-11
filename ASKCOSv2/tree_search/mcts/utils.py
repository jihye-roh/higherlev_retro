import sys
import time
import uuid
import itertools
import operator

import networkx as nx
import numpy as np

from api.pathway_ranker_api import PathwayRankerAPI
from api.scscorer_api import SCScorerAPI
from collections import defaultdict
from collections.abc import Iterator
from rdkit import Chem
from typing import Any, Dict, List, Tuple
from pathway_utils import * 


def is_terminal(
    smiles: str,
    build_tree_options=None,
    scscorer: SCScorerAPI=None,
    ppg: float = None,
    hist: dict = None,
    properties: list = None
) -> bool:
    """
    Determine if the specified chemical is a terminal node in the tree based
    on pre-specified criteria.

    Criteria to be considered are specified via ``self.termination_logic``,
    and the thresholds for each criterion are specified separately.

    If no criteria are specified, will always return ``False``.

    Args:
        smiles (str): smiles string of the chemical
        build_tree_options (BuildTreeOptions object): options for tree builder
        scscorer (SCScorerAPI): API to be used as an scscorer
        ppg (float): cost of the chemical
        hist (dict): historian data for the chemical
        properties (list): properties of the chemical
    """

    def buyable() -> bool:
        return bool(ppg) or (build_tree_options.custom_buyables and
                             smiles in build_tree_options.custom_buyables)

    def max_ppg() -> bool:
        if build_tree_options.max_ppg is not None:
            # ppg of 0 means not buyable
            return ppg is not None and 0 < ppg <= build_tree_options.max_ppg
        return True

    def max_scscore() -> bool:
        if build_tree_options.max_scscore is not None:
            scscore = scscorer(smiles=smiles)
            return scscore <= build_tree_options.max_scscore
        return True

    def max_elements() -> bool:
        if build_tree_options.max_elements is not None:
            # Get structural properties
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                elem_dict = defaultdict(int)
                for a in mol.GetAtoms():
                    elem_dict[a.GetSymbol()] += 1
                elem_dict["H"] = sum(a.GetTotalNumHs() for a in mol.GetAtoms())

                return all(elem_dict[k] <= v for k, v
                           in build_tree_options.max_elements.items())
        return True

    def min_history() -> bool:
        if build_tree_options.min_history is not None:
            return hist is not None and (
                hist["as_reactant"] >=
                build_tree_options.min_history["as_reactant"]
                or hist["as_product"] >=
                build_tree_options.min_history["as_product"]
            )
        return True

    def property_criteria() -> bool:
        if build_tree_options.property_criteria:
            results = check_property_criteria(
                properties=properties,
                criteria=build_tree_options.property_criteria
            )
            if "property_criteria" in or_criteria:
                return any(results)
            elif "property_criteria" in and_criteria:
                return all(results)
        return True

    local_dict = locals()
    or_criteria = build_tree_options.termination_logic.get("or")
    and_criteria = build_tree_options.termination_logic.get("and")

    return (
        bool(or_criteria)
        and any(local_dict[criteria]() for criteria in or_criteria)
        or bool(and_criteria)
        and all(local_dict[criteria]() for criteria in and_criteria)
    )
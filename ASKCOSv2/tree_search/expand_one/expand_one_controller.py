import numpy as np
import os
import time
from api.atom_map_api import AtomMapAPI
from api.cluster_api import ClusterAPI, ClusterSetting
from api.fast_filter_api import FastFilterAPI
from api.fast_filter_batch_api import FastFilterBatchAPI
from api.pricer_api import PricerAPI
from api.retro_api import RetroAPI
from api.scscorer_api import SCScorerAPI
from api.scscorer_batch_api import SCScorerBatchAPI
from descriptors_util import number_of_rings, rms_molecular_weight
from multiprocessing import Pool
from pydantic import BaseModel, Field
from rdchiral.template_extractor import extract_from_reaction
from rdchiral_util import apply_one_template_to_precursors, get_reacting_atoms
from rdkit import Chem
from typing import Any, Dict, List, Optional, Tuple

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://0.0.0.0:9100")

atom_mapper = AtomMapAPI(
    url=f"{GATEWAY_URL}/api/atom-map/call-sync",
    backend="rxnmapper"
)
clusterer = ClusterAPI(
    default_url=f"{GATEWAY_URL}/api/cluster/call-sync"
)
fast_filter = FastFilterAPI(
    default_url=f"{GATEWAY_URL}/api/fast-filter/call-sync"
)
fast_filter_batch = FastFilterBatchAPI(
    default_url=f"{GATEWAY_URL}/api/fast-filter/batch/call-sync"
)
pricer = PricerAPI(
    default_url=f"{GATEWAY_URL}/api/pricer/lookup-smiles"
)
scscorer = SCScorerAPI(
    default_url=f"{GATEWAY_URL}/api/scscore/call-sync"
)
scscorer_batch = SCScorerBatchAPI(
    default_url=f"{GATEWAY_URL}/api/scscore/batch/call-sync"
)
retro_controller = RetroAPI(
    default_url=f"{GATEWAY_URL}/api/retro/call-sync",
    default_backend="template_relevance"
)


class RetroBackendOption(BaseModel):
    retro_backend: str = "template_relevance"
    retro_model_name: str = "reaxys"
    max_num_templates: int = 100
    max_cum_prob: float = 0.995
    attribute_filter: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

    threshold: float = 0.3
    top_k: int = 10


def _get_relevance_new(reactant_smiles: str) -> float:
    scores = []
    for smiles in reactant_smiles.split("."):

        mol = Chem.MolFromSmiles(smiles)
        total_atoms = mol.GetNumHeavyAtoms()
        ring_bonds = sum(b.IsInRing() - b.GetIsAromatic() for b in mol.GetBonds())
        chiral_centers = len(Chem.FindMolChiralCenters(mol))

        scores.append(
            -2.00 * np.power(total_atoms, 1.5)
            - 1.00 * np.power(ring_bonds, 1.5)
            - 2.00 * np.power(chiral_centers, 2.0)
        )

    score = np.mean(scores)

    return score

def _get_relevance(_args: Tuple[str, str, float]) -> float:
    reactant_smiles, necessary_reagent, template_score = _args
    necessary_reagent_atoms = necessary_reagent.count("[") / 2.0
    scores = []
    for smiles in reactant_smiles.split("."):
        ppg = pricer(smiles=smiles, canonicalize=False)
        # If buyable, basically free
        if ppg:
            scores.append(-ppg / 1000.0)
            continue

        # Else, use heuristic
        mol = Chem.MolFromSmiles(smiles)
        total_atoms = mol.GetNumHeavyAtoms()
        ring_bonds = sum(b.IsInRing() - b.GetIsAromatic() for b in mol.GetBonds())
        chiral_centers = len(Chem.FindMolChiralCenters(mol))

        scores.append(
            -2.00 * np.power(total_atoms, 1.5)
            - 1.00 * np.power(ring_bonds, 1.5)
            - 2.00 * np.power(chiral_centers, 2.0)
        )

    score = np.sum(scores) - 4.00 * np.power(necessary_reagent_atoms, 2.0)
    score = score / template_score

    return score


def print_if_debug(m: str, debug: bool = False):
    if debug:
        print(m)


class ExpandOneController:
    def __init__(self):
        self.atom_mapper = atom_mapper
        self.clusterer = clusterer
        # self.fast_filter = fast_filter
        self.fast_filter_batch = fast_filter_batch
        self.pricer = pricer
        # self.scscorer = scscorer
        self.scscorer_batch = scscorer_batch
        self.retro_controller = retro_controller
        self.p = Pool()

    def merge_data(self, 
                result_1: Dict[str, Any],
                result_2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merges the data from two result dictionaries.

        Args:
            result_1: The first result dictionary.
            result_2: The second result dictionary.

        Returns:
            A dictionary containing the merged data.
        """
        if result_1 == result_2:
            return result_1

        if result_1["normalized_model_score"] > result_2["normalized_model_score"]:
            new_result, old_result = result_1, result_2
        else:
            new_result, old_result = result_2, result_1

        # Currently keeping the highest score (score, template score, etc)
        new_result = self.merge_template_data(new_result, old_result)
        new_result = self.merge_retrosim_data(new_result, old_result)
        
        new_result["models_predicted_by"] += old_result["models_predicted_by"]
        new_result["models_predicted_by"].sort(key=lambda x: x[2], reverse=True)

        return new_result

    def merge_template_data(self, 
                new_result: Dict[str, Any],
                old_result: Dict[str, Any]) -> Dict[str, Any]:
        # Merge template information
        if old_result.get("template"):
            if new_result.get("template"):
                new_result["template"]["tforms"].extend(old_result["template"]["tforms"])
                new_result["template"]["tsources"].extend(old_result["template"]["tsources"])
                new_result["template"]["num_examples"] += old_result["template"]["num_examples"]
            else:
                new_result["template"] = old_result["template"]

        return new_result
        
    def merge_retrosim_data(self,
                new_result: Dict[str, Any],
                old_result: Dict[str, Any]) -> Dict[str, Any]:
        
        # Merge reaction data information
        if old_result.get("reaction_data"):
            if new_result.get("reaction_data"):

                for field in ["reference_url", "patent_number"]:
                    new_result["reaction_data"][field] = (
                        new_result["reaction_data"].get(field) or 
                        old_result["reaction_data"].get(field)
                    )
            else:
                new_result["reaction_data"] = old_result["reaction_data"]

        return new_result

    def get_outcomes(
        self,
        smiles: str,
        retro_backend_options: List[RetroBackendOption],
        banned_chemicals: List[str] = None,
        banned_reactions: List[str] = None,
        use_fast_filter: bool = False,
        fast_filter_threshold: float = 0.75,
        retro_rerank_backend: str = "relevance_heuristic",
        cluster_precursors: bool = False,
        cluster_setting: ClusterSetting = None,
        extract_template: bool = False,
        return_reacting_atoms: bool = False,
        selectivity_check: bool = False,
        debug: bool = False
    ) -> List[Dict[str, any]]:
        if not banned_chemicals:
            banned_chemicals = []
        if not banned_reactions:
            banned_reactions = []

        # retro_controller takes in list[str], here we only pass in one smiles
        # retro_results is list[dict]
        start = time.time()
        retro_results = []
        for option in retro_backend_options:

            retro_result = self.retro_controller(
                smiles=[smiles],
                backend=option.retro_backend,
                model_name=option.retro_model_name,
                max_num_templates=option.max_num_templates,
                max_cum_prob=option.max_cum_prob,
                attribute_filter=option.attribute_filter,
                threshold=option.threshold,
                top_k=option.top_k
            )[0]
            for result in retro_result:
                result["retro_backend"] = option.retro_backend
                result["retro_model_name"] = option.retro_model_name
                result["models_predicted_by"] = \
                    [(
                        option.retro_backend, 
                        option.retro_model_name, 
                        result["normalized_model_score"]
                    )]
 
                if result.get("template"):
                    result["template"]["tsources"] = \
                        [option.retro_model_name]*len(result["template"]["tforms"])


            retro_results.extend(retro_result)
        print_if_debug(f"retro: {time.time() - start}", debug)

        # A number of postprocessing steps
        # <deduplication>
        start = time.time()
        mol = Chem.MolFromSmiles(smiles)
        cano_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        
        unique_results_dict = {}

        for result in retro_results:
            reactants_split = result["outcome"].split(".")
            if any(smi in banned_chemicals for smi in reactants_split):
                continue

            reaction_smi = result["outcome"] + ">>" + smiles
            cano_outcome = Chem.MolToSmiles(
                Chem.MolFromSmiles(result["outcome"]), isomericSmiles=True
            )
            result["outcome"] = cano_outcome
            cano_rxn_smi = f"{cano_outcome}>>{cano_smiles}"
            if reaction_smi in banned_reactions or cano_rxn_smi in banned_reactions:
                continue

            if cano_outcome == cano_smiles:
                continue

            if cano_outcome in unique_results_dict:
                
                result = self.merge_data(result, unique_results_dict[cano_outcome])

            #cano_outcomes.append(cano_outcome)
            #reaction_smis.append(reaction_smi)
            unique_results_dict[cano_outcome] = result

        unique_results = list(unique_results_dict.values())
        reaction_smis = [result["outcome"] + ">>" + smiles for result in unique_results]
        print_if_debug(f"dedup: {time.time() - start}", debug)
        # </deduplication>

        # <filtering>
        if use_fast_filter:
            start = time.time()
            plausibilities = self.fast_filter_batch(rxn_smiles=reaction_smis)
            print_if_debug(f"fast_filter: {time.time() - start}", debug)
        else:
            # hardcode to 1.0, since tree analysis still relies on these
            plausibilities = [1.0 for _ in reaction_smis]
        # something weird happening for fast_filter
        if not plausibilities:
            plausibilities = [0.5 for _ in reaction_smis]

        filtered_results = []
        for result, plausibility in zip(unique_results, plausibilities):
            if plausibility < fast_filter_threshold:
                continue
            result["plausibility"] = plausibility
            filtered_results.append(result)
        # </filtering>

        # <post-filter computation>
        start = time.time()
        smiles_list = [result["outcome"] for result in filtered_results]
        #scscore_batch_result = self.scscorer_batch(smiles_list=smiles_list)

        for result in filtered_results:
            # start = time.time()
            result["rms_molwt"] = rms_molecular_weight(result["outcome"])
            # print_if_debug(f"rms_molecular_weight: {time.time() - start}", debug)

            # start = time.time()
            result["num_rings"] = number_of_rings(result["outcome"])
            # print_if_debug(f"number_of_rings: {time.time() - start}", debug)

            # start = time.time()
            # result["scscore"] = scscore_batch_result[result["outcome"]]
            result["scscore"] = 5 # Placeholder
            # print_if_debug(f"scscorer: {time.time() - start}\n", debug)
            # scscore is the culprit; 50x the time of the other two
        print_if_debug(f"post-filter computation: {time.time() - start}", debug)
        # </post-filter computation>

        # <rerank>
        start = time.time()
        if retro_rerank_backend == "relevance_heuristic":
            reranked_results = self._rerank_by_relevance_heuristic_new(filtered_results)
        elif retro_rerank_backend == "scscore":
            reranked_results = self._rerank_by_scscore(filtered_results)
        elif retro_rerank_backend == "model_score":
            reranked_results = self._rerank_default(filtered_results)
        else:
            print(f"retro_rerank_backend: {retro_rerank_backend} not supported! "
                  f"Returning results based on normalized_model_score")
            reranked_results = self._rerank_default(filtered_results)
        print_if_debug(f"rerank: {time.time() - start}", debug)
        # </rerank>

        # <cluster>
        start = time.time()
        if cluster_precursors:
            reranked_results = reranked_results[:100]
            cluster_ids, names = self.clusterer(
                original=smiles,
                outcomes=[result["outcome"] for result in reranked_results],
                scores=[result["score"] for result in reranked_results],
                cluster_setting=cluster_setting
            )
            for result, cluster_id in zip(reranked_results, cluster_ids):
                result["group_id"] = cluster_id
                try:
                    result["group_name"] = names[str(cluster_id)]
                except KeyError:
                    result["group_name"] = names[cluster_id]
        print_if_debug(f"cluster: {time.time() - start}", debug)
        # </cluster>

        # <template extraction>
        start = time.time()
        if extract_template or selectivity_check:
            for result in reranked_results:
                if "template" in result and result["template"]:
                    continue

                if "mapped_smiles" not in result:
                    rxn_smi = result["outcome"] + ">>" + smiles
                    res_atom_mapper = self.atom_mapper(smiles=[rxn_smi])
                    mapped_rxn_smi = res_atom_mapper[0] if res_atom_mapper else ""
                    result["mapped_smiles"] = mapped_rxn_smi

                reactants, _, products = result["mapped_smiles"].split(">")
                reaction = {
                    '_id': -1,
                    'reactants': reactants,
                    'products': products
                }
                try:
                    template = extract_from_reaction(reaction)
                except:
                    template = {}
                if (
                    "reaction_smarts" not in template
                    or not template["reaction_smarts"]
                ):
                    template["reaction_smarts"] = "failed_extraction"

                for k in [
                    "reactants_smarts",
                    "products_smarts",
                    "reaction_smarts_forward",
                    "reaction_smarts_retro",
                    "reactants",
                    "products"
                ]:
                    template.pop(k, None)
                result["template"] = template
        print_if_debug(f"extract: {time.time() - start}", debug)
        # </template extraction>

        # <reacting atoms computation>
        start = time.time()
        if return_reacting_atoms:
            if all("reacting_atoms" in result for result in reranked_results):
                pass
            else:
                if all("mapped_smiles" in result for result in reranked_results):
                    pass
                else:
                    # force remap all if not all mapped, with a batch call
                    rxn_smis_to_map = [
                        result["outcome"] + ">>" + smiles
                        for result in reranked_results
                    ]
                    mapped_rxn_smis = self.atom_mapper(smiles=rxn_smis_to_map)
                    for result, mapped_rxn_smi in zip(
                        reranked_results, mapped_rxn_smis
                    ):
                        if mapped_rxn_smi:
                            result["mapped_smiles"] = mapped_rxn_smi
                        else:
                            result["mapped_smiles"] = ""

                for result in reranked_results:
                    if result["mapped_smiles"]:
                        # print("----------mapped_smiles----------")
                        # print(result["mapped_smiles"])
                        reacting_atoms = get_reacting_atoms(result["mapped_smiles"])
                        # print("----------reacting_atoms----------")
                        # print(reacting_atoms)

                        # Reverse mapping as atom_mapper will canonicalize the product SMILES
                        # The reacting_atoms returned correspond to the indices in the
                        # canonical SMILES; need to map them back into the original SMILES

                        # DO NOT use CanonicalRankAtoms; this is something different
                        # canonical_rank = tuple(Chem.CanonicalRankAtoms(mol))
                        # canonical_rank: (1, 2, 6, 10, 7, 8, 3, 5, 9, 4, 0)
                        # canonical_rank_inverted = tuple(zip(
                        #     *sorted((j, i) for i, j in enumerate(canonical_rank))
                        # ))[1]
                        # canonical_rank_inverted: (10, 0, 1, 6, 9, 7, 2, 4, 5, 8, 3)
                        output_order = mol.GetProp('_smilesAtomOutputOrder')
                        # output_order: something like '[0,1,2,3,]'
                        # output_order = eval(output_order)

                        # This is very stupid but Snyk doesn't like eval()
                        # had to switch to the below logic, essentially equivalent to eval
                        output_order = [
                            int(c)
                            for c in output_order.lstrip("[").rstrip("]").split(",")
                            if c
                        ]

                        # print("----------output_order----------")
                        # print(output_order)

                        # SMILES: C1=CC=C2C=C(C=CC2=C1)OCC
                        # canonical SMILES: CCOc1ccc2ccccc2c1
                        # output_order: [12,11,10,5,6,7,8,9,0,1,2,3,4,]

                        # Raw reacting_atom 3 corresponds to the O
                        # (with an atom mapping no. of 3 in canonical SMILES),
                        # which has a no. of 11 in the original SMILES.
                        # So we need to convert "3" to "11"

                        reacting_atoms = [output_order[a-1] + 1 for a in reacting_atoms]
                        # The -1 and +1 is just converting between 0- and 1- indexed

                        # print("----------reacting_atoms----------")
                        # print(reacting_atoms)

                        result["reacting_atoms"] = reacting_atoms

                        # FIXME: temporary hardcode to be consistent with V1;
                        #  really hate the naming as only a reaction SMILES can be "mapped"
                        result["mapped_smiles"] = result["mapped_smiles"].split(">")[0]
                        # Reset the mappings for the reactant SMILES too
                        r_mol = Chem.MolFromSmiles(result["mapped_smiles"])
                        for a in r_mol.GetAtoms():
                            atom_map_num = a.GetAtomMapNum()
                            if atom_map_num > 0:
                                a.SetAtomMapNum(output_order[atom_map_num-1] + 1)
                        result["mapped_smiles"] = Chem.MolToSmiles(r_mol)
                        # print("----------mapped_smiles----------")
                        # print(result["mapped_smiles"])

                    else:
                        result["reacting_atoms"] = []
        print_if_debug(f"return react: {time.time() - start}", debug)
        # </reacting atoms computation>

        # <selectivity check>
        start = time.time()
        if selectivity_check:
            for result in reranked_results:
                template = result["template"]["reaction_smarts"]
                if template == "failed_extraction":
                    continue

                mapped_products, mapped_precursors = apply_one_template_to_precursors(
                    precursors=result["outcome"],
                    template=template
                )
                if mapped_products and cano_smiles not in mapped_products:
                    # We couldn't recover the original product for some reason
                    result["selec_error"] = True
                    continue

                # Look for other products besides the target with the same number of heavy atoms
                product_atom_count = Chem.MolFromSmiles(smiles).GetNumHeavyAtoms()
                other_products = [
                    x for x in mapped_products
                    if x != cano_smiles
                    and Chem.MolFromSmiles(x).GetNumHeavyAtoms() == product_atom_count
                ]

                if len(other_products) > 0:
                    result["outcomes"] = ".".join(
                        [smiles] + [x for x in other_products]
                    )
                    result["mapped_outcomes"] = ".".join(
                        [mapped_products[smiles]]
                        + [mapped_products[x] for x in other_products]
                    )
                    result["mapped_precursors"] = mapped_precursors
        print_if_debug(f"selec check: {time.time() - start}", debug)
        # </selectivity check>

        return reranked_results

    @staticmethod
    def _rerank_default(filtered_results: List[Dict[str, Any]]
                           ) -> List[Dict[str, Any]]:
        for result in filtered_results:
            result["score"] = result["normalized_model_score"]

        reranked_results = sorted(
            filtered_results,
            key=lambda d: d["score"],
            reverse=True
        )

        for rank, result in enumerate(reranked_results, start=1):
            result["rank"] = rank

        return reranked_results

    def _rerank_by_relevance_heuristic(self, filtered_results: List[Dict[str, Any]]
                                       ) -> List[Dict[str, Any]]:
        tasks = []
        for result in filtered_results:
            try:
                necessary_reagent = result["template"]["necessary_reagent"]
            except (KeyError, TypeError):
                necessary_reagent = ""

            try:
                template_score = result["template"]["template_score"]
            except (KeyError, TypeError):
                template_score = result["normalized_model_score"]

            tasks.append((result["outcome"], necessary_reagent, template_score))

        scores = self.p.imap(_get_relevance, tasks)
        for result, score in zip(filtered_results, scores):
            result["score"] = score

        reranked_results = sorted(
            filtered_results,
            key=lambda d: d["score"],
            reverse=True
        )

        for rank, result in enumerate(reranked_results, start=1):
            result["rank"] = rank

        return reranked_results

    @staticmethod
    def _rerank_by_scscore(filtered_results: List[Dict[str, Any]]
                           ) -> List[Dict[str, Any]]:
        for result in filtered_results:
            result["score"] = result["scscore"]

        reranked_results = sorted(
            filtered_results,
            key=lambda d: d["score"],
            reverse=True
        )

        for rank, result in enumerate(reranked_results, start=1):
            result["rank"] = rank

        return reranked_results

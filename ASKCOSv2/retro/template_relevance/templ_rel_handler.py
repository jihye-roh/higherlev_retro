import argparse
import glob
import numpy as np
import os
import templ_rel_parser
import torch
import torch.nn.functional as F
from utils import canonicalize_smiles
from rdchiral.initialization import rdchiralReactants, rdchiralReaction
from rdchiral.main import rdchiralRun
from typing import Any, Dict, List, Tuple
from utils import get_model, load_templates_as_list, mol_smi_to_count_fp


class TemplRelHandler:
    """TemplRel Handler for use with torchserve"""

    def __init__(self):
        self._context = None
        self.manifest = None
        self.initialized = False

        self.args = None
        self.templates = None
        self.template_attributes = None
        self.model = None
        self.indices = None
        self.device = None

    def initialize(self, context):
        self._context = context
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        print(glob.glob(f"{model_dir}/*"))
        self.device = torch.device("cuda:" + str(properties.get("gpu_id"))
                                   if torch.cuda.is_available() else "cpu")

        serve_parser = argparse.ArgumentParser("serve")
        templ_rel_parser.add_predict_opts(serve_parser)
        self.args, _ = serve_parser.parse_known_args()

        template_file = os.path.join(model_dir, "templates.jsonl")
        print(f"Loading templates and attributes from {template_file}")
        self.templates, self.template_attributes = \
            load_templates_as_list(template_file=template_file)
        print(f'Total number of templates: {len(self.templates)}')

        checkpoint_file = os.path.join(model_dir, "model_latest.pt")
        print(f"Building model and loading from {checkpoint_file}")

        self.args.load_from = checkpoint_file
        self.args.local_rank = -1
        # Note: the model will be built using pretraining args
        self.model, _ = get_model(self.args, device=self.device)
        self.model.eval()

        self.initialized = True

    def preprocess(self, data: List[dict]
                   ) -> Tuple[List[str], int, float, List[Dict[str, Any]]]:
        print(f"input: {data}")
        canonical_smiles = [canonicalize_smiles(smi, remove_atom_number=True, remove_isotope=False)
                            for smi in data[0]["body"]["smiles"]]
        max_num_templates = data[0]["body"].get("max_num_templates", 1000)
        max_cum_prob = data[0]["body"].get("max_cum_prob", 0.999)
        attribute_filter = data[0]["body"].get("attribute_filter", [])

        return canonical_smiles, max_num_templates, max_cum_prob, attribute_filter

    def inference(self, inputs: Tuple[List[str], int, float, List[Dict[str, Any]]]
                  ) -> List[Dict[str, Any]]:
        canonical_smiles, max_num_templates, max_cum_prob, attribute_filter = inputs
        filters = [x for x in attribute_filter
                   if x.get("name") in self.template_attributes]

        if filters:
            filter_query = " and ".join(
                [f"({q['name']} {q['logic']} {q['value']})" for q in filters]
            )
            filtered_indices = self.template_attributes.query(filter_query).index.values

        results = []

        # maybe not the most efficient way, but should have been faster than v1 already
        with torch.no_grad():
            for smi in canonical_smiles:
                prod_fp = mol_smi_to_count_fp(smi, self.args.radius, self.args.fp_size)
                final_fp = torch.as_tensor(prod_fp.toarray()).float().to(self.device)

                # template_prioritizer.predict()
                logits = self.model(final_fp)
                scores = F.softmax(logits, dim=1)
                scores = scores.squeeze(dim=0).cpu().numpy()
                indices = np.argsort(-scores)
                scores = scores[indices]

                # retro_transformer.filter_by_attributes()
                if filters:
                    bool_mask = np.isin(indices, filtered_indices)
                    indices = indices[bool_mask]
                    scores = scores[bool_mask]

                # template_prioritizer.truncate()
                if max_num_templates:
                    indices = indices[:max_num_templates]
                    scores = scores[:max_num_templates]

                if max_cum_prob:
                    exceeds = np.nonzero(np.cumsum(scores) >= max_cum_prob)[0]
                    if exceeds.size:
                        # Include the prediction which exceeds max_cum_prob
                        max_index = exceeds[0] + 1
                        scores = scores[:max_index]
                        indices = indices[:max_index]

                smiles_to_index = {}
                result = {
                    "templates": [],
                    "reactants": [],
                    "scores": []
                }
                for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
                    template = self.templates[idx]
                    # IMPORTANT MAGIC from v1. DO NOT TOUCH
                    # Force reactants and products to be one pseudo-molecule (bookkeeping)
                    reaction_smarts = template["reaction_smarts"]
                    reaction_smarts_one = "(" + reaction_smarts.replace(">>", ")>>(") + ")"
                    rxn = rdchiralReaction(str(reaction_smarts_one))
                    prod = rdchiralReactants(smi)
                    try:
                        # New: set return_mapped to False. Remap after in expand_one
                        reactants = rdchiralRun(rxn, prod, return_mapped=False)
                    except:
                        continue                # unknown error in rdchiral

                    if not reactants:           # empty precursors
                        continue

                    template = {k: v for k, v in template.items()
                                if k not in ["references", "rxn"]}
                    template["template_score"] = score.item()
                    template["template_rank"] = rank

                    for reactant in reactants:
                        # here we just return the full template with metadata
                        # in theory it should also be fetchable from the DB
                        smiles_list = reactant.split(".")
                        if template["intra_only"] and len(smiles_list) > 1:
                            # Disallowed intermolecular reaction
                            continue
                        if template["dimer_only"] and (
                            len(set(smiles_list)) != 1 or len(smiles_list) != 2
                        ):
                            # Not a dimer
                            continue
                        if smi in smiles_list:
                            # Skip if no transformation happened
                            continue

                        joined_smiles = ".".join(sorted(smiles_list))
                        if joined_smiles in smiles_to_index:
                            # Precursor was already generated by another template
                            # -> update metadata. Templates are ordered by score, so
                            # no need to update template score or rank
                            res = result["templates"][
                                smiles_to_index[joined_smiles]
                            ]
                            if template["_id"] not in res["tforms"]:
                                res["tforms"].append(template["_id"])
                            res["num_examples"] += template["count"]
                        else:
                            # New precursor -> generate metadata
                            template["tforms"] = [template["_id"]]
                            template["num_examples"] = template["count"]
                            smiles_to_index[joined_smiles] = len(result["templates"])

                            result["templates"].append(template)
                            result["reactants"].append(reactant)
                            result["scores"].append(score.item())

                    del rxn, prod, reactants

                results.append(result)

        return results

    def postprocess(self, data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        return [data]

    def handle(self, data: List, context) -> List[List[Dict[str, Any]]]:
        self._context = context

        inputs = self.preprocess(data)
        output = self.inference(inputs)
        output = self.postprocess(output)

        # print(output)     # turn off (debug) printing to speed up

        return output

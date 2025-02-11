import itertools
import networkx as nx
import numpy as np
import os
import re
import random
import time
from api.expand_one_api import ExpandOneAPI
from api.historian_api import HistorianAPI
from api.pathway_ranker_api import PathwayRankerAPI
from api.pricer_api import PricerAPI, SmartsPricerAPI
from api.reaction_classification_api import ReactionClassificationAPI
from api.scscorer_api import SCScorerAPI
from options import ExpandOneOptions, BuildTreeOptions, EnumeratePathsOptions
from rdkit import Chem
from typing import List, Optional, Set, Tuple
from utils import get_graph_from_tree, is_terminal, nx_graph_to_paths, nx_paths_to_json, canonicalize

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://0.0.0.0:9100")
expand_one = ExpandOneAPI(
    default_url=f"{GATEWAY_URL}/api/tree-search/expand-one/call-sync-without-token",
)
historian = HistorianAPI(
    default_url=f"{GATEWAY_URL}/api/historian/lookup-smiles"
)
pathway_ranker = PathwayRankerAPI(
    url=f"{GATEWAY_URL}/api/pathway-ranker/call-sync"
)
pricer = PricerAPI(
    default_url=f"{GATEWAY_URL}/api/pricer/lookup-smiles" 
)
smarts_pricer = SmartsPricerAPI(
    default_url=f"{GATEWAY_URL}/api/pricer/lookup-smarts"
)
reaction_classifier = ReactionClassificationAPI(
    url=f"{GATEWAY_URL}/api/get-top-class-batch/call-sync"
)
scscorer = SCScorerAPI(
    default_url=f"{GATEWAY_URL}/api/scscore/call-sync"
)


class MCTS:
    def __init__(self):
        self.expand_one_options = None
        self.build_tree_options = None
        self.enumerate_paths_options = None

        self.tree = nx.DiGraph()        # directed graph
        self.target = None              # the target compound
        self.target_uuid = None         # unique identifier for the target in paths
        self.paths = None               # pathway results as nx graphs
        self.chemicals = []
        self.reactions = []
        self.iterations = 0
        self.time_to_solve = 0
        self.path_time = 0
        self.enumerated_paths = []

        self.expand_one = expand_one
        self.historian = historian
        self.pathway_ranker = pathway_ranker
        self.pricer = pricer
        self.smarts_pricer = smarts_pricer
        self.scscorer = scscorer
        self.reaction_classifier = reaction_classifier

    @property
    def num_unique_chemicals(self):
        """Number of unique chemicals explored."""
        return len(self.chemicals)

    @property
    def num_unique_reactions(self):
        """Number of unique reactions explored."""
        return len(self.reactions)

    @property
    def num_total_reactions(self):
        """Total number of reactions explored."""
        return sum(self.tree.out_degree(chem) for chem in self.chemicals)

    @property
    def done(self):
        """Determine if we're done expanding the tree."""
        return (
            self.tree.nodes[self.target]["done"]
            or (
                self.build_tree_options.max_iterations is not None
                and self.iterations >= self.build_tree_options.max_iterations
            )
            or (
                self.build_tree_options.max_chemicals is not None
                and self.num_unique_chemicals >= self.build_tree_options.max_chemicals
            )
            or (
                self.build_tree_options.max_reactions is not None
                and self.num_unique_reactions >= self.build_tree_options.max_reactions
            )
            or (
                self.build_tree_options.max_templates is not None
                and self.num_total_reactions >= self.build_tree_options.max_templates
            )
        )

    def get_buyable_paths(
        self,
        target: str,
        expand_one_options: ExpandOneOptions = ExpandOneOptions(),
        build_tree_options: BuildTreeOptions = BuildTreeOptions(),
        enumerate_paths_options: EnumeratePathsOptions = EnumeratePathsOptions(),
    ) -> Tuple[List[dict], dict, dict]:
        """
        Build retrosynthesis tree and return paths to buyable precursors.
        *Based on v2 tree builder*
        Args:
            target (str): SMILES of target chemical
            expand_one_options (ExpandOneOptions object): options for one-step retro
            build_tree_options (BuildTreeOptions object): options for build_tree
            enumerate_paths_options (EnumeratePathsOptions object):
                options for enumerate_paths

        Returns:
            trees (list of dict): List of synthetic routes as networkx json
            stats (dict): Various statistics about the expansion
            graph (dict): Full explored graph as networkx node link json
        """
        self.expand_one_options = expand_one_options
        self.build_tree_options = build_tree_options
        self.enumerate_paths_options = enumerate_paths_options

        start = time.time()
        self.build_tree(target=target)
        build_time = time.time() - start

        # Note: Skipping pathway enumeration 
        # - done as post processing due to the need for multiple sorting metrics
        start = time.time()
        # paths = []
        # path_time = 0
        paths = self.enumerate_paths()
        path_time = time.time() - start

        graph = nx.node_link_data(get_graph_from_tree(self.tree))
        stats = {
            "total_iterations": self.iterations,
            "total_chemicals": self.num_unique_chemicals,
            "total_reactions": self.num_unique_reactions,
            "total_templates": self.num_total_reactions,
            "total_paths": len(paths),
            "first_path_time": self.time_to_solve,
            "build_time": build_time,
            "path_time": path_time,
        }

        return paths, stats, graph

    def _initialize(self, target: str) -> None:
        """
        Initialize the tree by with the target chemical.
        """
        self.target = Chem.MolToSmiles(
            Chem.MolFromSmiles(target),
            isomericSmiles=True
        )           # Canonicalize SMILES
        self.create_chemical_node(smiles=self.target)
        self.tree.nodes[self.target]["terminal"] = False
        self.tree.nodes[self.target]["solved"] = False
        self.tree.nodes[self.target]["done"] = False
        self.tree.nodes[self.target]["min_depth"] = 0

    def create_chemical_node(self, smiles: str) -> None:
        """
        Create a new chemical node from the provided SMILES and populate node
        properties with chemical data.

        Includes purchase price and *no* template info
        """

        template_sets = [option.retro_model_name for option
                         in self.expand_one_options.retro_backend_options]
        # if abstracted group, use smarts_pricer
        _st_chem = time.time()
        if re.findall(r"\[\d+", smiles):
            start = time.time()
            purchase_price, properties, buyable_data = self.smarts_pricer(
                smarts=smiles,
                source=self.build_tree_options.buyables_source, 
                max_ppg=self.build_tree_options.max_ppg, 
                convert_smiles=True,
                version='preloaded_vec'
            )

            hist = {"as_reactant":0, "as_product":0}

        else: 
            start = time.time()
            purchase_price, properties = self.pricer(
                smiles=smiles,
                source=self.build_tree_options.buyables_source,
                canonicalize=False
            )
            buyable_data = None
            hist = self.historian(
                smiles=smiles,
                template_sets=template_sets,
                canonicalize=False
            )
        

        if self.build_tree_options.max_ppg and purchase_price > self.build_tree_options.max_ppg: 
            purchase_price = 0 

        terminal = is_terminal(
            smiles=smiles,
            build_tree_options=self.build_tree_options,
            scscorer=self.scscorer,
            ppg=purchase_price,
            hist=hist,
            properties=properties
        )
        est_value_sum = 1.0 if terminal else 0.0
        est_value_cnt = 1
        est_value = est_value_sum / est_value_cnt

        self.chemicals.append(smiles)
        # *terminal* is like a static property, e.g., buyable
        # *expanded* indicates whether _expand() has been called on this node
        # *solved* indicates whether this node falls on *a* buyable path
        # *done* is similar to "proven", indicating whether all subtrees have been expanded
        # Of course, a node can only be "done" if it has been "expanded",
        # unless it's "terminal", in which case it'd been "done" at creation
        self.tree.add_node(
            smiles,
            as_reactant=hist["as_reactant"],
            as_product=hist["as_product"],
            est_value_sum=est_value_sum,        # total value of node
            est_value_cnt=est_value_cnt,        # total visit count of node
            est_value=est_value,                # average value of node
            min_depth=None,         # minimum depth at which this chemical appears
            properties=properties,  # properties from buyables database if any
            purchase_price=purchase_price,
            buyable_data=buyable_data,
            solved=terminal,        # whether a path to terminal leaves has been found from this node
            terminal=terminal,      # whether this chemical meets terminal criteria
            done=terminal,          # simplified update logic from is_chemical_done()
            expanded=False,
            type="chemical",
            visit_count=1, 
            iteration = self.iterations # iteration at which this node was created
        )

    def create_reaction_node(
        self,
        smiles: str,
        precursor_smiles: str,
        tforms: Optional[List[str]],
        tsources: Optional[List[str]],
        necessary_reagent: Optional[str],
        template_tuple: Optional[Tuple[str, str]],
        rxn_score_from_model: float,
        plausibility: float,
        num_examples: int,
        forward_score: Optional[float],
        rms_molwt: Optional[float],
        num_rings: Optional[int],
        scscore: Optional[float],
        rank: Optional[int],
        score: Optional[float],
        class_num,
        class_name,
        template,
        retro_backend,
        retro_model_name, 
        models_predicted_by
    ):
        """Create a new reaction node from the provided smiles and data."""
        self.reactions.append(smiles)
        self.tree.add_node(
            smiles,
            est_value=0.0,      # score for how feasible a route is, based on whether precursors are terminal
            plausibility=plausibility,
            solved=False,       # whether a path to terminal leaves has been found from this node
            rxn_score_from_model=rxn_score_from_model,
            num_examples=num_examples,
            forward_score=forward_score,
            rms_molwt=rms_molwt,
            num_rings=num_rings,
            scscore=scscore,
            rank=rank,
            score=score,
            class_num=class_num,
            class_name=class_name,
            template_tuples=[template_tuple] if template_tuple is not None else [],
            precursor_smiles=precursor_smiles,
            tforms=tforms,
            tsources=tsources,
            necessary_reagent=necessary_reagent,
            type="reaction",
            visit_count=1,
            template=template,
            retro_backend=retro_backend,
            retro_model_name=retro_model_name,
            models_predicted_by=models_predicted_by,
            iteration = self.iterations # iteration at which this node was created
        )

    def is_reaction_done(self, smiles: str) -> bool:
        """
        Determine if the specified reaction node should be expanded further.

        Reaction nodes are done when all of its children chemicals are done.
        """
        return all(self.tree.nodes[c]["done"] for c in self.tree.successors(smiles))

    def build_tree(
        self,
        target: str
    ) -> None:
        """
        Build retrosynthesis tree by iterative expansion of precursor nodes.
        """
        print(f"Initializing tree with target {target}...")
        self._initialize(target)

        print("Starting tree expansion...")
        start_time = time.time()
        elapsed_time = time.time() - start_time

        while elapsed_time < self.build_tree_options.expansion_time and not self.done:
            chem_path, rxn_path = self._select()
            if not chem_path and not rxn_path:
                # backtracked to the root, which means no path was found
                break

            self.iterations += 1
            self._expand(chem_path)
            self._update(chem_path, rxn_path)

            elapsed_time = time.time() - start_time

            if self.iterations % 20 == 0:
                print(f"Iteration {self.iterations} ({elapsed_time: .2f}s): "
                      f"|C| = {len(self.chemicals)} "
                      f"|R| = {len(self.reactions)}")
            
            if not self.time_to_solve and self.tree.nodes[self.target]["solved"]:
                self.time_to_solve = elapsed_time
                print(f"Found first pathway after {elapsed_time:.2f} seconds.")
                if self.build_tree_options.return_first:
                    print("Stopping expansion to return first pathway.")
                    break

        print("Tree expansion complete.")
        self.print_stats()

    def _select(self) -> tuple[list[str], list[str]]:
        """
        Select next unexpanded leaf node to be expanded.

        This starts at the root node (target chemical), and at each level,
        use UCB to score each of the "reaction" options which can be taken.
        It will take the optimal option, which will now be an already explored
        reaction. It will descend to the next level and repeat the process until
        reaching an unexpanded node.
        """
        _st = time.time()
        chem_path = [self.target]
        rxn_path = []
        invalid_options = set()

        while True:
            leaf = chem_path[-1]

            self.tree.nodes[leaf]["min_depth"] = \
                min(self.tree.nodes[leaf]["min_depth"], len(chem_path)-1) # update min_depth

            if len(chem_path) <= self.build_tree_options.max_depth \
                and not self.tree.nodes[leaf]["expanded"]:

                break
            

            elif len(chem_path) >= self.build_tree_options.max_depth:

                invalid_options.add(leaf)
                del chem_path[-1]
                try:
                    del rxn_path[-1]
                except IndexError: # no more options at the root; terminate
                    return [], []
                continue
            
            options = self.ucb(
                node=leaf,
                chem_path=chem_path,
                invalid_options=invalid_options,
                exploration_weight=self.build_tree_options.exploration_weight
            )
            if not options:
                # There are no valid options from this chemical node, need to backtrack
                invalid_options.add(leaf)
                del chem_path[-1]
                try:
                    del rxn_path[-1]
                except IndexError:      # no more options at the root; terminate
                    return [], []
                continue

            score, reaction = options[0]
            # With ASKCOSv2 refactor, a reaction would always have been *explored*
            # If there are multiple reactants, pick the one with the lower visit count
            # Do not consider chemicals that are already done or chemicals that are on the path
            precursor = min(
                (
                    c for c in self.tree.successors(reaction)
                    if not self.tree.nodes[c]["done"] and c not in invalid_options
                ),
                key=lambda _node: self.tree.nodes[_node]["visit_count"],
                default=None
            )
            if precursor is None:
                # There are no valid options from this reaction node, need to backtrack
                invalid_options.add(reaction)
                continue
            else:
                chem_path.append(precursor)
                rxn_path.append(reaction)
        

        return chem_path, rxn_path

    def ucb(
        self,
        node: str,
        chem_path: List[str],
        invalid_options: Set[str],
        exploration_weight: float
    ) -> List[Tuple[float, str]]:
        """
        Calculate UCB score for all exploration options from the specified node. Only
        considers *explored reactions*, as reactions would have always been explored.

        Returns a list of (score, option) tuples sorted by score.
        """
        options = []
        product_visits = self.tree.nodes[node]["visit_count"]
        
        # Get scores for explored templates (reaction node exists)
        for rxn in self.tree.successors(node):
            rxn_data = self.tree.nodes[rxn]

            if (
                rxn in invalid_options
                # Simplified is_reaction_done
                or all(self.tree.nodes[c]["done"] for c in self.tree.successors(rxn))
                or len(set(self.tree.successors(rxn)) & set(chem_path)) > 0     # FIXME: why this condition
            ):
                continue

            est_value = rxn_data["est_value"]
            node_visits = rxn_data["visit_count"]
            # normalized_score is a generalized version of template score,
            # which would have been handled/computed by the retro controller
            rxn_probability = rxn_data["rxn_score_from_model"]

            # Q represents how good a move is
            q_sa = rxn_probability * est_value / node_visits
            # U represents how many times this move has been explored
            u_sa = np.sqrt(np.log(product_visits) / node_visits)

            score = q_sa + exploration_weight * u_sa

            # The options here are to follow a reaction down one level
            options.append((score, rxn))
            # print(f"score: {score}, rxn: {rxn},
            # est_value: {est_value}, prob: {rxn_probability}")

        # Sort options from highest to lowest score
        options.sort(key=lambda x: x[0], reverse=True)

        return options

    def _expand(self, chem_path: List[str]) -> None:
        """
        Expand the tree by running one-step retro prediction to a chemical node
        """
        _st = time.time()

        leaf = chem_path[-1]
        retro_results = self.expand_one(
            smiles=leaf,
            expand_one_options=self.expand_one_options
        )

        self.tree.nodes[leaf]["expanded"] = True
        if not retro_results:
            # if no retro_results, then this node is done
            self.tree.nodes[leaf]["done"] = True
            return
        
        for result in retro_results:
            precursor_smiles = result["outcome"]
            reactant_list = precursor_smiles.split(".")
            reaction_smiles = precursor_smiles + ">>" + leaf
        
            try:
                template = result["template"]
                template_tuple = (template["index"], template["template_set"])
                num_examples = template["num_examples"]
            except (KeyError, TypeError):
                # try-except just for backward compatibility
                template = None
                template_tuple = None
                num_examples = 0

            _st_res = time.time()
            if reaction_smiles in self.reactions:
                # This Reaction node already exists
                rxn_data = self.tree.nodes[reaction_smiles]
                if (
                    template_tuple is not None
                    and template_tuple not in rxn_data["template_tuples"]
                ):
                    rxn_data["template_tuples"].append(template_tuple)
                    rxn_data["num_examples"] += num_examples

                # retro controller now computes normalized_model_score
                rxn_data["rxn_score_from_model"] = max(
                    rxn_data["rxn_score_from_model"], result["normalized_model_score"]
                )

            else:
                # This is new, so create a Reaction node
                tforms = template.get(
                    "tforms") if isinstance(template, dict) else None
                tsources = template.get(
                    "tsources") if isinstance(template, dict) else None
                
                if not tsources:
                    tsources = template.get(
                        "template_set") if isinstance(template, dict) else None
                    if tsources is not None:
                        tsources = [tsources] * len(tforms)

                necessary_reagent = template.get(
                    "necessary_reagent") if isinstance(template, dict) else None
                self.create_reaction_node(
                    smiles=reaction_smiles,
                    precursor_smiles=precursor_smiles,
                    tforms=tforms,
                    tsources=tsources,
                    necessary_reagent=necessary_reagent,
                    template_tuple=template_tuple,
                    rxn_score_from_model=result["normalized_model_score"],
                    plausibility=result["plausibility"],
                    num_examples=num_examples,
                    forward_score=result.get("forward_score"),
                    rms_molwt=result.get("rms_molwt"),
                    num_rings=result.get("num_rings"),
                    scscore=result.get("scscore"),
                    rank=result.get("rank"),
                    score=result.get("score", result["normalized_model_score"]),
                    class_num=result.get("class_num"),
                    class_name=result.get("class_name"),
                    template=template,
                    retro_backend=result.get("retro_backend"),
                    retro_model_name=result.get("retro_model_name"),
                    models_predicted_by=result.get("models_predicted_by")
                )

            # Add edges to connect target -> reaction -> precursors
            self.tree.add_edge(leaf, reaction_smiles)

            for reactant in reactant_list:
                if reactant not in self.chemicals:
                    # This is new, so create a Chemical node
                    self.create_chemical_node(smiles=reactant)
                self.tree.add_edge(reaction_smiles, reactant)
                # initialize/change min_depth for reactants 
                self.tree.nodes[reactant]['min_depth'] = \
                    int(nx.shortest_path_length(self.tree, source=self.target, target=reactant)/2)
            # This _update_value only updates reactions *below* leaf
            self._update_value(reaction_smiles)

    def _update(self, chem_path: List[str], rxn_path: List[str]) -> None:
        """
        Update status and reward for nodes in this path.

        Reaction nodes are guaranteed to only have a single parent. Thus, the
        status of its parent chemical will always be updated appropriately in
        ``_update`` and will not change until the next time the chemical is
        in the selected path. Thus, the done state of the chemical can be saved.

        However, chemical nodes can have multiple parents (i.e. can be reached
        via multiple reactions), so a given update cycle may only pass through
        one of multiple parent reactions. Thus, the done state of a reaction
        must be determined dynamically and cannot be saved.
        """
        _st = time.time()
        assert (
            chem_path[0] == self.target
        ), "Chemical path should start at the root node."

        # Iterate over the full path in reverse
        # On each iteration, rxn will be the parent reaction of chem
        # For the root (target) node, rxn will be None
        for i, chem, rxn in itertools.zip_longest(
            range(len(chem_path) - 1, -1, -1), reversed(chem_path), reversed(rxn_path)
        ):
            chem_data = self.tree.nodes[chem]
            chem_data["visit_count"] += 1
            # updates min_depth for chemicals in the pathway
            chem_data["min_depth"] = (
                min(chem_data["min_depth"], i)
                if chem_data["min_depth"] is not None
                else i
            )
            # simplified update logic from is_chemical_done()
            if chem_data["done"]:
                done = True
            # min_depth can change - chemical only "done" if all of its children reactions are done 
            # elif chem_data["min_depth"] >= self.build_tree_options.max_depth: 
            #     done = True

            # chemical is done if all or at least max_branching children reactions are "done"
            elif (
                sum(self.is_reaction_done(r) for r in self.tree.successors(chem))
                >= min(self.build_tree_options.max_branching, self.tree.out_degree(chem))
            ):
                done = True
            else:
                done = False
            chem_data["done"] = done

            if rxn is not None:
                rxn_data = self.tree.nodes[rxn]
                rxn_data["visit_count"] += 1
                # This _update_value only updates reactions *above* the expanded leaf
                self._update_value(rxn)

    def _update_value(self, smiles: str):
        """
        Update the value of the specified reaction node and its parent.
        """
        rxn_data = self.tree.nodes[smiles]

        if rxn_data["type"] == "reaction":
            # Calculate value as the sum of the values of all precursors
            est_value = sum(
                self.tree.nodes[c]["est_value"] for c in self.tree.successors(smiles)
            )

            # Update estimated value of reaction
            rxn_data["est_value"] += est_value

            # Update estimated value of parent chemical
            chem_data = self.tree.nodes[next(self.tree.predecessors(smiles))]

            # use running average rather than summation for est_value,
            # to be more consistent with V1
            # chem_data["est_value"] += est_value
            chem_data["est_value_sum"] += est_value
            chem_data["est_value_cnt"] += 1
            chem_data["est_value"] = \
                chem_data["est_value_sum"] / chem_data["est_value_cnt"]

            # Check if this node is solved
            if not rxn_data["solved"]:
                rxn_data["solved"] = all(
                    self.tree.nodes[c]["solved"] for c in self.tree.successors(smiles)
                )
            # propagate to the parent chemical node
            if not chem_data["solved"]:
                chem_data["solved"] = rxn_data["solved"]

    def enumerate_paths(self) -> List:
        """
        Return list of paths to buyables starting from the target node.
        """
        if self.build_tree_options.return_first:
            self.enumerate_paths_options.score_trees = False
            self.enumerate_paths_options.cluster_trees = False

        self.paths, self.target_uuid = nx_graph_to_paths(
            self.tree,
            self.target,
            max_depth=self.build_tree_options.max_depth,
            max_trees=self.build_tree_options.max_trees,
            sorting_metric=self.enumerate_paths_options.sorting_metric,
            validate_paths=self.enumerate_paths_options.validate_paths,
            pathway_ranker=self.pathway_ranker,
            cluster_method=self.enumerate_paths_options.cluster_method,
            min_samples=self.enumerate_paths_options.min_samples,
            min_cluster_size=self.enumerate_paths_options.min_cluster_size
        )

        if self.paths: 
            print(f"Found {len(self.paths)} paths to buyable chemicals at iteration {self.iterations}")
        max_paths = self.enumerate_paths_options.max_paths
        if len(self.paths) > max_paths:
            print(f"Number of paths exceeds max_paths {max_paths}, down sampling")
            # random.shuffle(self.paths)
            self.paths = self.paths[:max_paths]

        path_format = self.enumerate_paths_options.path_format
        json_format = self.enumerate_paths_options.json_format

        if path_format == "graph":
            paths = self.paths
        elif path_format == "json":
            paths = nx_paths_to_json(
                paths=self.paths,
                root_uuid=self.target_uuid,
                json_format=json_format
            )
        else:
            raise ValueError(f"Unrecognized format type {path_format}")

        return paths

    def print_stats(self) -> None:
        """
        Print tree statistics.
        """
        info = "\n"
        info += f"Number of iterations: {self.iterations}\n"
        num_nodes = self.tree.number_of_nodes()
        info += f"Number of nodes: {num_nodes:d}\n"
        info += f"    Chemical nodes: {len(self.chemicals):d}\n"
        info += f"    Reaction nodes: {len(self.reactions):d}\n"
        info += f"Number of edges: {self.tree.number_of_edges():d}\n"
        if num_nodes > 0:
            info += f"Average in degree: " \
                    f"{sum(d for _, d in self.tree.in_degree()) / num_nodes:.4f}\n"
            info += f"Average out degree: " \
                    f"{sum(d for _, d in self.tree.out_degree()) / num_nodes:.4f}"

        print(info)

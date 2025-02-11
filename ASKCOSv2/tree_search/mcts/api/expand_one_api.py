import requests
import traceback as tb
from options import ClusterSetting, ExpandOneOptions, RetroBackendOption
from pydantic import BaseModel, error_wrappers
from typing import Any, Dict, List, Optional


class RetroResult(BaseModel):
    # from retro_controller
    outcome: str
    model_score: float
    normalized_model_score: float
    template: Optional[Dict[str, Any]]
    reaction_id: Optional[str]
    reaction_set: Optional[str]

    # extended from postprocessing in expand_one_controller
    retro_backend: str
    retro_model_name: str
    models_predicted_by: list[tuple[str, str, float]]
    plausibility: Optional[float]
    rms_molwt: float
    num_rings: int
    scscore: float
    group_id: Optional[int]
    group_name: Optional[str]
    mapped_smiles: Optional[str]
    reacting_atoms: Optional[List[int]]
    selec_error: Optional[bool]
    mapped_outcomes: Optional[str]
    mapped_precursors: Optional[str]

    score: float
    rank: int


class ExpandOneInput(BaseModel):
    # mirroring the (default) wrapper; convenient to turn into a client library
    smiles: str
    retro_backend_options: List[RetroBackendOption] = [RetroBackendOption()]
    banned_chemicals: List[str] = None
    banned_reactions: List[str] = None
    use_fast_filter: bool = True
    fast_filter_threshold: float = 0.75
    retro_rerank_backend: Optional[str] = None
    cluster_precursors: bool = False
    cluster_setting: Optional[ClusterSetting] = None
    extract_template: bool = False
    return_reacting_atoms: bool = True
    selectivity_check: bool = False


class ExpandOneResponse(BaseModel):
    # mirroring the (default) wrapper, but without BaseResponse (semi-hardcode)
    status_code: int
    message: str
    result: List[RetroResult]


class ExpandOneAPI:
    """ExpandOne API to be used as a one-step expansion engine"""
    def __init__(self, default_url: str):
        self.default_url = default_url
        self.session = requests.Session()

    def __call__(
        self,
        smiles: str,
        expand_one_options: ExpandOneOptions,
        url: str = None
    ) -> Optional[List[Dict[str, any]]]:
        if not url:
            url = self.default_url

        # Overriding backend option fields if provided in expand_one_options
        retro_backend_options = expand_one_options.retro_backend_options
        # if expand_one_options.template_max_count:
        #     for option in expand_one_options.retro_backend_options:
        #         option.max_num_templates = expand_one_options.template_max_count
        # if expand_one_options.template_max_cum_prob:
        #     for option in expand_one_options.retro_backend_options:
        #         option.max_cum_prob = expand_one_options.template_max_cum_prob

        # Modify the overriding logic to apply only if not provided for each option
        for option in expand_one_options.retro_backend_options:
            if not option.max_num_templates and expand_one_options.template_max_count:
                option.max_num_templates = expand_one_options.template_max_count
            if not option.max_cum_prob and expand_one_options.template_max_cum_prob:
                option.max_cum_prob = expand_one_options.template_max_cum_prob

        cluster_setting = expand_one_options.cluster_setting
        if cluster_setting is not None:
            cluster_setting = cluster_setting.dict()

        input = {
            "smiles": smiles,
            "retro_backend_options": [
                option.dict() for option in retro_backend_options
            ],
            "banned_chemicals": expand_one_options.banned_chemicals,
            "banned_reactions": expand_one_options.banned_reactions,
            "use_fast_filter": expand_one_options.use_fast_filter,
            "fast_filter_threshold": expand_one_options.filter_threshold,
            "retro_rerank_backend": expand_one_options.retro_rerank_backend,
            "cluster_precursors": expand_one_options.cluster_precursors,
            "cluster_setting": cluster_setting,
            "extract_template": expand_one_options.extract_template,
            "return_reacting_atoms": expand_one_options.return_reacting_atoms,
            "selectivity_check": expand_one_options.selectivity_check
        }
        # additional validation. Sending null/none value to FastAPI seems to
        # fail the validation check and break the defaulting mechanism
        input = {k: v for k, v in input.items() if v is not None}

        ExpandOneInput(**input)                     # merely validate the input
        try:
            response = self.session.post(url=url, json=input).json()
            ExpandOneResponse(**response)           # merely validate the response
        except requests.exceptions.ConnectionError:
            # Handle the connection error appropriately
            print("Connection error for ExpandOneAPI:")
            tb.print_exc()

            return None
        except error_wrappers.ValidationError:
            tb.print_exc()
            print(response)

            return None

        except Exception:
            # Handle any other exception that might occur
            print("An error occurred for ExpandOneAPI:")
            tb.print_exc()

            return None

        result = response["result"]

        return result

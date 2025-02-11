import argparse
import copy
import logging
import os
import sys
import traceback
import uvicorn
from api.cluster_api import ClusterSetting
from datetime import datetime
from expand_one_controller import ExpandOneController, RetroBackendOption
from fastapi import FastAPI
from pydantic import BaseModel
from rdkit import RDLogger
from typing import List

app = FastAPI()

base_response = {
    "status": "FAIL",
    "error": "",
    "results": []
}


def parse_args():
    parser = argparse.ArgumentParser("expand_one_server")
    parser.add_argument("--server_ip",
                        help="Server IP to use", type=str, default="0.0.0.0")
    parser.add_argument("--server_port",
                        help="Server port to use", type=int, default=9301)
    parser.add_argument("--log_file",
                        help="Log file", type=str, default="expand_one_server")

    return parser.parse_args()


class RequestBody(BaseModel):
    smiles: str
    retro_backend_options: List[RetroBackendOption] = [RetroBackendOption()]
    banned_chemicals: List[str] = None
    banned_reactions: List[str] = None
    use_fast_filter: bool = False
    fast_filter_threshold: float = 0.75
    retro_rerank_backend: str = "relevance_heuristic"
    cluster_precursors: bool = False
    cluster_setting: ClusterSetting = ClusterSetting()
    extract_template: bool = False
    return_reacting_atoms: bool = False
    selectivity_check: bool = False


@app.post("/get_outcomes")
def expand_one_service(request: RequestBody):
    response = copy.deepcopy(base_response)

    try:
        results = controller.get_outcomes(
            smiles=request.smiles,
            retro_backend_options=request.retro_backend_options,
            banned_chemicals=request.banned_chemicals,
            banned_reactions=request.banned_reactions,
            use_fast_filter=request.use_fast_filter,
            fast_filter_threshold=request.fast_filter_threshold,
            retro_rerank_backend=request.retro_rerank_backend,
            cluster_precursors=request.cluster_precursors,
            cluster_setting=request.cluster_setting,
            extract_template=request.extract_template,
            return_reacting_atoms=request.return_reacting_atoms,
            selectivity_check=request.selectivity_check,
        )
        response["results"] = results
        response["status"] = "SUCCESS"

    except Exception:
        response["error"] = f"Error during expand one, traceback: " \
                            f"{traceback.format_exc()}"
        traceback.print_exc()

    return response


if __name__ == "__main__":
    args = parse_args()

    # logger setup
    RDLogger.DisableLog("rdApp.warning")

    os.makedirs(f"./logs", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"./logs/{args.log_file}.{dt}.log")
    sh = logging.StreamHandler(sys.stdout)
    fh.setLevel(logging.INFO)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    # set up model
    controller = ExpandOneController()

    # start running
    uvicorn.run(app, host=args.server_ip, port=args.server_port)

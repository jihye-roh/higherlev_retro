import argparse
import copy
import logging
import os
import sys
import traceback
import uvicorn
from datetime import datetime
from fastapi import FastAPI
from mcts_controller import MCTS
from options import ExpandOneOptions, BuildTreeOptions, EnumeratePathsOptions
from pydantic import BaseModel
from rdkit import RDLogger

app = FastAPI()

base_response = {
    "status": "FAIL",
    "error": "",
    "results": {}
}


def parse_args():
    parser = argparse.ArgumentParser("mcts_server")
    parser.add_argument("--server_ip",
                        help="Server IP to use", type=str, default="0.0.0.0")
    parser.add_argument("--server_port",
                        help="Server port to use", type=int, default=9311)
    parser.add_argument("--log_file",
                        help="Log file", type=str, default="mcts_server")

    return parser.parse_args()


class RequestBody(BaseModel):
    smiles: str
    expand_one_options: ExpandOneOptions = ExpandOneOptions()
    build_tree_options: BuildTreeOptions = BuildTreeOptions()
    enumerate_paths_options: EnumeratePathsOptions = EnumeratePathsOptions()


@app.post("/get_buyable_paths")
def mcts_service(request: RequestBody):
    response = copy.deepcopy(base_response)

    try:
        controller = MCTS()
        paths, stats, graph = controller.get_buyable_paths(
            target=request.smiles,
            expand_one_options=request.expand_one_options,
            build_tree_options=request.build_tree_options,
            enumerate_paths_options=request.enumerate_paths_options
        )
        if request.enumerate_paths_options.paths_only:
            results = {
                "paths": paths
            }
        else:
            results = {
                "stats": stats,
                "paths": paths,
                "graph": graph,
                "version": 2,
            }

        response["results"] = results
        response["status"] = "SUCCESS"

        # V1 does create a controller per query. TODO: We'll optimize this later
        del controller

    except Exception:
        response["error"] = f"Error during mcts, traceback: " \
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

    # start running
    uvicorn.run(app, host=args.server_ip, port=args.server_port)

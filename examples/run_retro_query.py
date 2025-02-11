import copy
import pprint
import argparse
import requests


query_template = {
    "backend": "template_relevance",
    "model_name": "", 
    "smiles": [],
    "max_num_templates": 100,
    "max_cum_prob": 1,
    "attribute_filter": []
}

def get_args():

    args = argparse.ArgumentParser()
    args.add_argument("--data", type=str, default="example")
    args.add_argument("--model_name", type=str, default="uspto_higher-level_consol")
    args.add_argument("--max_num_templates", type=int, default=100)
    args.add_argument("--max_cum_prob", type=float, default=1)
    args = args.parse_args()

    return args

def get_query_template(args):

    with open(f"./data/targets/{args.data}.txt", "r") as f:
        smiles = f.readlines()
    smiles = [smi.split(",")[0].strip() for smi in smiles]

    data = copy.deepcopy(query_template)
    
    data["smiles"] = smiles
    data["model_name"] = args.model_name
    data["max_num_templates"] = args.max_num_templates
    data["max_cum_prob"] = args.max_cum_prob

    return data, smiles

def main():
    HOST = "0.0.0.0"
    PORT = "9100"
    
    args = get_args()

    data, smiles = get_query_template(args)

    resp = requests.post(
        url=f"http://{HOST}:{PORT}/api/retro/controller/call-sync",
        json=data
    ).json()

    for smi, res in zip(smiles, resp["result"]):
        print(f"Printing the first two results for SMILES {smi}")
        pprint.pprint(res[:2])
        print()


if __name__ == "__main__":
    main()

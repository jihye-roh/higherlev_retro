import argparse
import gzip
import json
import numpy as np
import os
import templ_rel_parser
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from scipy.special import softmax
from tqdm import tqdm


attrs = {
    "reaxys": {
        "template_file_old": "./tmp/legacy_v1/retro.templates.json.gz",
        "template_file_new": "./tmp/legacy_v2/retro.templates.reaxys.jsonl",
        "model_folder_old": "./tmp/legacy_v1/reaxys/1",
        "model_ckpt_new": "./tmp/legacy_v2/reaxys.pt",
        "n_templates": 163723,
        "fp_size": 2048,
        "hidden_sizes": "300,300,300,300,300",
        "n_params": 50256523,
        "hidden_activation": "relu",
        "skip_connection": "none"
    },
    "cas": {
        "template_file_old": "./tmp/legacy_v1/retro.templates.cas.json.gz",
        "template_file_new": "./tmp/legacy_v2/retro.templates.cas.jsonl",
        "model_folder_old": "./tmp/legacy_v1/cas/1",
        "model_ckpt_new": "./tmp/legacy_v2/cas.pt",
        "n_templates": 179475,
        "fp_size": 2048,
        "hidden_sizes": "1500",
        "n_params": 272465475,
        "hidden_activation": "relu",
        "skip_connection": "none"
    },
    "pistachio": {
        "template_file_old": "./tmp/legacy_v1/retro.templates.pistachio.json.gz",
        "template_file_new": "./tmp/legacy_v2/retro.templates.pistachio.jsonl",
        "model_folder_old": "./tmp/legacy_v1/pistachio/1",
        "model_ckpt_new": "./tmp/legacy_v2/pistachio.pt",
        "n_templates": 219536,
        "fp_size": 2048,
        "hidden_sizes": "1024",
        "n_params": 227122576,
        "hidden_activation": "relu",
        "skip_connection": "none"
    },
    "pistachio_ringbreaker": {
        "template_file_old": "./tmp/legacy_v1/retro.templates.pistachio_ringbreaker.json.gz",
        "template_file_new": "./tmp/legacy_v2/retro.templates.pistachio_ringbreaker.jsonl",
        "model_folder_old": "./tmp/legacy_v1/pistachio_ringbreaker/1",
        "model_ckpt_new": "./tmp/legacy_v2/pistachio_ringbreaker.pt",
        "n_templates": 32493,
        "fp_size": 2048,
        "hidden_sizes": "1024",
        "n_params": 35403501,
        "hidden_activation": "relu",
        "skip_connection": "none"
    },
    "bkms_metabolic": {
        "template_file_old": "./tmp/legacy_v1/retro.templates.bkms.json.gz",
        "template_file_new": "./tmp/legacy_v2/retro.templates.bkms_metabolic.jsonl",
        "model_folder_old": "./tmp/legacy_v1/bkms/1",
        "model_ckpt_new": "./tmp/legacy_v2/bkms_metabolic.pt",
        "n_templates": 7984,
        "fp_size": 2048,
        "hidden_sizes": "4096",
        "n_params": 41103152,
        "hidden_activation": "relu",
        "skip_connection": "none"
    },
    "reaxys_biocatalysis": {
        "template_file_old": "./tmp/legacy_v1/retro.templates.reaxys_enzymatic.json.gz",
        "template_file_new": "./tmp/legacy_v2/retro.templates.reaxys_biocatalysis.jsonl",
        "model_folder_old": "./tmp/legacy_v1/reaxys_enzymatic/1",
        "model_ckpt_new": "./tmp/legacy_v2/reaxys_biocatalysis.pt",
        "n_templates": 8802,
        "fp_size": 2048,
        "hidden_sizes": "4096",
        "n_params": 44454498,
        "hidden_activation": "relu",
        "skip_connection": "none"
    }
}


def main():
    parser = argparse.ArgumentParser("Converter")
    templ_rel_parser.add_model_opts(parser)

    os.makedirs("./tmp/legacy_v2", exist_ok=True)
    do_convert_template = True

    test_data = np.random.random([256, 2048]).astype(np.float32)

    for model_name, metadata in attrs.items():
        print(f"----------------------Converting {model_name}----------------------")
        args, _ = parser.parse_known_args()
        for k, v in metadata.items():
            setattr(args, k, v)

        # template conversion
        fn = metadata["template_file_old"]
        ofn = metadata["template_file_new"]
        if do_convert_template and not os.path.exists(ofn):
            print(f"Converting templates for {model_name}")
            with gzip.open(fn, "rt") as f:
                templates = json.load(f)
            with open(ofn, "w") as of:
                for template in tqdm(templates):
                    of.write(f"{json.dumps(template)}\n")

        tf_model = tf.keras.models.load_model(metadata["model_folder_old"])

        torch_model, _ = utils.get_model(args, torch.device("cpu"))
        torch_model.eval()
        print(torch_model)
        assert utils.param_count(torch_model) == metadata["n_params"]

        # model conversion
        if model_name == "reaxys":
            wi = tf_model.get_layer("dense_16").kernel.numpy()
            bi = tf_model.get_layer("dense_16").bias.numpy()
            w1 = tf_model.get_layer("dense_17").kernel.numpy()
            b1 = tf_model.get_layer("dense_17").bias.numpy()
            w2 = tf_model.get_layer("dense_18").kernel.numpy()
            b2 = tf_model.get_layer("dense_18").bias.numpy()
            w3 = tf_model.get_layer("dense_19").kernel.numpy()
            b3 = tf_model.get_layer("dense_19").bias.numpy()
            w4 = tf_model.get_layer("dense_20").kernel.numpy()
            b4 = tf_model.get_layer("dense_20").bias.numpy()
            wo = tf_model.get_layer("dense_21").kernel.numpy()
            bo = tf_model.get_layer("dense_21").bias.numpy()

            # Important note: tf Dense weight is transpose of torch Linear weight
            torch_model.layers[0].linear.weight = nn.Parameter(torch.from_numpy(wi).T)
            torch_model.layers[0].linear.bias = nn.Parameter(torch.from_numpy(bi))
            torch_model.layers[1].linear.weight = nn.Parameter(torch.from_numpy(w1).T)
            torch_model.layers[1].linear.bias = nn.Parameter(torch.from_numpy(b1))
            torch_model.layers[2].linear.weight = nn.Parameter(torch.from_numpy(w2).T)
            torch_model.layers[2].linear.bias = nn.Parameter(torch.from_numpy(b2))
            torch_model.layers[3].linear.weight = nn.Parameter(torch.from_numpy(w3).T)
            torch_model.layers[3].linear.bias = nn.Parameter(torch.from_numpy(b3))
            torch_model.layers[4].linear.weight = nn.Parameter(torch.from_numpy(w4).T)
            torch_model.layers[4].linear.bias = nn.Parameter(torch.from_numpy(b4))
            torch_model.output_layer.weight = nn.Parameter(torch.from_numpy(wo).T)
            torch_model.output_layer.bias = nn.Parameter(torch.from_numpy(bo))
        else:
            wi = tf_model.get_layer("dense").kernel.numpy()
            bi = tf_model.get_layer("dense").bias.numpy()
            wo = tf_model.get_layer("dense_1").kernel.numpy()
            bo = tf_model.get_layer("dense_1").bias.numpy()

            # Important note: tf Dense weight is transpose of torch Linear weight
            torch_model.layers[0].linear.weight = nn.Parameter(torch.from_numpy(wi).T)
            torch_model.layers[0].linear.bias = nn.Parameter(torch.from_numpy(bi))
            torch_model.output_layer.weight = nn.Parameter(torch.from_numpy(wo).T)
            torch_model.output_layer.bias = nn.Parameter(torch.from_numpy(bo))

        tf_output = tf_model.predict(test_data)
        if model_name in ["reaxys", "bkms_metabolic", "reaxys_biocatalysis"]:
            tf_output = softmax(tf_output, axis=1)

        with torch.no_grad():
            torch_output = torch_model(torch.as_tensor(test_data, dtype=torch.float32))
            torch_output = F.softmax(torch_output, dim=1)

        torch_output = torch_output.detach().numpy()
        em = np.mean(np.argmax(tf_output, axis=1) == np.argmax(torch_output, axis=1))
        print(f"argmax EM: {em}")

        assert em == 1.0, em
        state = {
            "args": args,
            "state_dict": torch_model.state_dict()
        }
        torch.save(state, metadata["model_ckpt_new"])


if __name__ == "__main__":
    # torch.set_printoptions(profile="full")
    utils.set_seed(42)

    main()

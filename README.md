# Higher-level Strategies for Computer-Aided Retrosynthesis #

This repo contains the code for [Higher-level Strategies for Computer-Aided Retrosynthesis](https://chemrxiv.org/engage/chemrxiv/article-details/67a11f0ffa469535b9f4648c). 

<img align="center" src="Higher-level_retrosynthesis.png" width="900px" />



## Overview
The scripts for generating higher-level route and reaction datasets are provided in the `dataset_curation/` directory. The scripts for deploying and running synthesis planning with ASKCOS are provided in the `ASKCOSv2/` directory. All example scripts are provided in the `examples/` directory. 


## Data

The data for this project are available at [this link](https://doi.org/10.6084/m9.figshare.28306673).

- `datasets.zip`: The dataset curation pipeline in this project relies on classified and atom-mapped reaction data generated using the NameRXN software, which we are unable to release. We release the reaction and route datasets that were generated as a result of this pipeline. The resulting datasets are included in this file. 

- `template_relevance_models_and_data.zip`: Contains the files necessary to deploy ASKCOS and run synthesis planning with all four single-step models used in this projects (i.e., .mar files for the four model, buyables file with price information). This zip file also contains the reaction splits, templates, and model checkpoints that are not necessary for deployment. 

- `higher-level_consol_model_and_data.zip`: Contains the files necessary to deploy ASKCOS and run synthesis planning with just the higher-level single-step model (with template consolidation).

-----

## Generating higher-level dataset from a list of reactions

After cloning this repository, follow the following steps to generate your own higher-level dataset (from a list of reactions).

### 1. Set up the environment and activate it
```shell
$ cd higherlev_retro
$ conda env create -f environment.yml
$ conda activate higherlev_retro
$ pip install -e rdchiral
```

### 2. Generate higher-level route/reaction datasets

The following script will run reaction processing, multistep (original) route extraction, and higher-level route generation.

```shell
$ sh scripts/00_generate_higher-level_dataset.sh
```

This script uses the reactions in `data/reactions/uspto.reactions.example.json.gz` to generate the original & higher-level reaction/route datasets. Note that the atom-mapping in this reaction file was not generated by NameRXN and may lead to different results. The resulting routes (both original and higher-level) will be saved as `data/routes/uspto.routes.jsonl.gz`. The jupyter notebook `examples/view_generated_routes.ipynb` contains examples for viewing the data once it is generated. The reaction dataset will be saved separately as `data/reactions/uspto_original.csv` and `data/reactions/uspto_higher-level.csv`. 



### 3. Generate higher-level routes from a multistep (original) route

A higher-level route can be generated directly from a multistep route instead of extracting multistep (original) routes from a reaction dataset.

```Python
from datastructs.abs_tree import AbsTree
 
route = [
"[F:1][c:2]1[cH:3][cH:4][c:5](-[c:6]2[o:7][c:8]3[cH:9][cH:10][c:11]([OH:12])[cH:13][c:14]3[c:15]2[C:16](=[O:17])[OH:22])[cH:20][cH:21]1.[NH2:18][CH3:19]>>[F:1][c:2]1[cH:3][cH:4][c:5](-[c:6]2[o:7][c:8]3[cH:9][cH:10][c:11]([OH:12])[cH:13][c:14]3[c:15]2[C:16](=[O:17])[NH:18][CH3:19])[cH:20][cH:21]1",
      "[c:1]1([O:31][S:32]([C:33]([F:34])([F:35])[F:36])(=[O:37])=[O:38])[cH:12][cH:13][c:14]2[o:15][c:16](-[c:17]3[cH:18][cH:19][c:20]([F:21])[cH:22][cH:23]3)[c:24]([C:25](=[O:26])[NH:27][CH3:28])[c:29]2[cH:30]1.[c:2]1([B:39]([OH:40])[OH:41])[cH:3][c:4]([C:5]([OH:6])=[O:7])[cH:8][cH:9][c:10]1[Cl:11]>>[c:1]1(-[c:2]2[cH:3][c:4]([C:5]([OH:6])=[O:7])[cH:8][cH:9][c:10]2[Cl:11])[cH:12][cH:13][c:14]2[o:15][c:16](-[c:17]3[cH:18][cH:19][c:20]([F:21])[cH:22][cH:23]3)[c:24]([C:25](=[O:26])[NH:27][CH3:28])[c:29]2[cH:30]1",
      "[F:1][c:2]1[cH:3][cH:4][c:5](-[c:6]2[o:7][c:8]3[cH:9][cH:10][c:11]([OH:12])[cH:20][c:21]3[c:22]2[C:23]([NH:24][CH3:25])=[O:26])[cH:27][cH:28]1.[S:13]([C:14]([F:15])([F:16])[F:17])(=[O:18])(=[O:19])[N:29]([c:30]1[cH:31][cH:32][cH:33][cH:34][cH:35]1)[S:36]([C:37]([F:38])([F:39])[F:40])(=[O:41])=[O:42]>>[F:1][c:2]1[cH:3][cH:4][c:5](-[c:6]2[o:7][c:8]3[cH:9][cH:10][c:11]([O:12][S:13]([C:14]([F:15])([F:16])[F:17])(=[O:18])=[O:19])[cH:20][c:21]3[c:22]2[C:23]([NH:24][CH3:25])=[O:26])[cH:27][cH:28]1"
]
abs_tree = AbsTree(route)
route_data = abs_tree.get_abstraction_data()
```
The jupyter notebook `examples/route_abstraction_example.ipynb` contains more details about generating higher-level routes. Note that the reactions in the route must be atom-mapped. 

### Optional: Training a new template-relevance model

To train your own model using `data/reactions/uspto_higher-level.csv`, run 

```shell
$ cd ASKCOSv2/retro/template_relevance/
$ sh scripts/benchmark.sh
$ cd ../../../
```
-----

## Running MCTS with ASKCOS with pretrained models
This repo contains the ASKCOS modules that were modified for the purpose of this project. After cloning this repo, please follow the following steps to deploy ASKCOS with the modified code. Note that this is a minimal version of ASKCOS, so you would not be able to access any of the other modules (e.g., SCScore, forward predictor). Please see [ASKCOSv2 Gitlab](https://gitlab.com/mlpds_mit/askcosv2) to access all functionalities and documentation.

### 1. Download Data

**< For all single-step models >**
1) Download `template_relevance_models_and_data.zip` from [this link](https://doi.org/10.6084/m9.figshare.28306673). Place this zip file in your `higherlev_retro/` directory
2) Unzip and move the subdirectories into the corresponding locations with the following:
```shell
$ unzip template_relevance_models_and_data.zip
$ mv template_relevance_models_and_data/askcos2_core_data/* ASKCOSv2/askcos2_core
$ mv template_relevance_models_and_data/template_relevance_data/* ASKCOSv2/retro/template_relevance
$ rm -rf template_relevance_models_and_data
```

**< For higher-level single-step model (with template consolidation) ONLY >**
1) Download `template_relevance_models_and_data.zip` from [this link](https://doi.org/10.6084/m9.figshare.28306673). Place this zip file in your `higherlev_retro/` directory
2) Unzip and move the subdirectories into the corresponding locations with the following:
```shell
$ unzip higher-level_consol_model_and_data.zip
$ mv higher-level_consol_model_and_data/askcos2_core_data/* ASKCOSv2/askcos2_core
$ mv higher-level_consol_model_and_data/template_relevance_data/* ASKCOSv2/retro/template_relevance
$ rm -rf higher-level_consol_model_and_data
```

### 2. Deploy ASKCOS
Move to `ASKCOSv2/askcos2_core` directory, activate the higherlev_retro environment, then deploy ASKCOS with the following commands

```shell
$ cd ASKCOSv2/askcos2_core
$ conda activate higherlev_retro
$ make deploy
$ cd ../../
```

### 3. Run synthesis planning

**< Running single-step retrosynthesis with the template-relevance model >**

Example scripts for running single-step retrosynthesis experiments can be found at `examples/run_retro_query.py`. This can be run with:

```shell
$ python examples/run_retro_query.py --model_name=uspto_higher-level_consol --max_num_templates=25 --max_cum_prob=1.0 --data=example
```

**< Running multi-step retrosynthesis planning with MCTS >**

Example scripts for running MCTS experiments can be found at `examples/run_mcts_query.py`. This script will take ~30s to run. 

```shell
$ python examples/run_mcts_query.py --model_name=uspto_higher-level_consol --max_depth=8 --max_num_templates=25 --num_workers=1 --data=example
```

The jupyter notebook `examples/mcts_network_evaluation.ipynb` includes examples for evaluating the generated results (enumerating/evaluating pathways, searching for buyables). The list of target molecules used to generate the results in this project is available at  `data/mcts/`.


### 5. Stop ASKCOS
To stop running ASKCOS, run the following:

```shell
$ cd ASKCOSv2/askcos2_core
$ make stop
$ cd ../../
```

You can re-run ASKCOS with:

```shell
$ cd ASKCOSv2/askcos2_core
$ make update
$ cd ../../
```



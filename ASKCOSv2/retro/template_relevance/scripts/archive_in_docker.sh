#!/bin/bash

if [ -z "${ASKCOS_REGISTRY}" ]; then
  export ASKCOS_REGISTRY=registry.gitlab.com/mlpds_mit/askcosv2/askcos2_core
fi

export DATA_NAME="uspto_original_consol"
export PROCESSED_DATA_PATH=$PWD/data/$DATA_NAME/processed
export MODEL_PATH=$PWD/checkpoints/$DATA_NAME

export EXTRA_FILES="\
misc.py,\
models.py,\
templ_rel_parser.py,\
/app/template_relevance/checkpoints/model_latest.pt,\
/app/template_relevance/data/processed/templates.jsonl,\
utils.py\
"

docker run --rm \
  -v "$PROCESSED_DATA_PATH/templates.jsonl":/app/template_relevance/data/processed/templates.jsonl \
  -v "$MODEL_PATH/model_latest.pt":/app/template_relevance/checkpoints/model_latest.pt \
  -v "$PWD/mars":/app/template_relevance/mars \
  -t "${ASKCOS_REGISTRY}"/retro/template_relevance:1.0-gpu \
  torch-model-archiver \
  --model-name="$DATA_NAME" \
  --version=1.0 \
  --handler=/app/template_relevance/templ_rel_handler.py \
  --extra-files="$EXTRA_FILES" \
  --export-path=/app/template_relevance/mars \
  --force

#!/bin/bash
export DATA_NAME="example_consol"
# export DATA_NAME="uspto_higher-level_nonconsol"
echo "DATA_NAME: $DATA_NAME"

case "$DATA_NAME" in
  *_consol*)
    export RUN_CONSOL="True"
    export USE_PROCESSED_DATA="False"
    echo "RUN_CONSOL: $RUN_CONSOL"
    echo "USE_PROCESSED_DATA: $USE_PROCESSED_DATA"
    ;;
  *)
    export RUN_CONSOL="False"
    export USE_PROCESSED_DATA="True"
    echo "RUN_CONSOL: $RUN_CONSOL"
    echo "USE_PROCESSED_DATA: $USE_PROCESSED_DATA"
    ;;
esac

export RAW_FILE_PATH=$PWD/data/$DATA_NAME/raw
mkdir -p $RAW_FILE_PATH

export RXN_DIR="../../../data/reactions"

export ALL_REACTION_FILE=$RXN_DIR/uspto_higher-level.csv
export TRAIN_FILE=$PWD/data/$DATA_NAME/raw/raw_train.csv
export VAL_FILE=$PWD/data/$DATA_NAME/raw/raw_val.csv
export TEST_FILE=$PWD/data/$DATA_NAME/raw/raw_test.csv
export NUM_CORES=32

export PROCESSED_DATA_PATH=$PWD/data/$DATA_NAME/processed
export MODEL_PATH=$PWD/checkpoints/$DATA_NAME
export TEST_OUTPUT_PATH=$PWD/results/$DATA_NAME

mkdir -p $PROCESSED_DATA_PATH
mkdir -p $MODEL_PATH
mkdir -p $TEST_OUTPUT_PATH


if [ "$RUN_CONSOL" = "True" ]; then
  [ -f $ALL_REACTION_FILE ] || { echo $ALL_REACTION_FILE does not exist; exit; }
else
  [ -f $TRAIN_FILE ] || { echo $TRAIN_FILE does not exist; exit; }
  [ -f $VAL_FILE ] || { echo $VAL_FILE does not exist; exit; }
  [ -f $TEST_FILE ] || { echo $TEST_FILE does not exist; exit; }
fi


SPLIT="8:1:1"

# Loop through the command-line arguments
while [ $# -gt 0 ]; do
  key="$1"

  case $key in
    --split_ratio)
      # If the argument is "--split_ratio", store the next argument in SPLIT
      SPLIT="$2"
      shift 2 # Consume both arguments
      ;;
    *)
      # Handle other arguments or options if needed
      shift
      ;;
  esac
done

export SPLIT="$SPLIT"

echo "SPLIT: $SPLIT"

export TOPK=10
export MAX_NUM_TEMPL=100

if [ "$RUN_CONSOL" = "True" ]; then  
  echo "Running consolidation..., "
  bash scripts/consolidate.sh
else
  echo "Skipping template consolidation..."
fi

echo "Running pre-split data processing... with using processed data: $USE_PROCESSED_DATA"
bash scripts/preprocess.sh
bash scripts/train.sh

export IS_OPTIMISTIC_RANKING="False"
bash scripts/predict.sh
export IS_OPTIMISTIC_RANKING="True"
bash scripts/predict.sh
# bash scripts/archive_in_docker.sh # requires docker

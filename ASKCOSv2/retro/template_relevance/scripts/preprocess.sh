#!/bin/bash



# Function to run the pre-split data processing
run_presplit() {
  echo "Running pre-split data processing..."
  python templ_rel_preprocessor.py \
    --model_name="template_relevance" \
    --data_name="$DATA_NAME" \
    --log_file="template_relevance_preprocess_$DATA_NAME" \
    --train_file="$TRAIN_FILE" \
    --val_file="$VAL_FILE" \
    --test_file="$TEST_FILE" \
    --processed_data_path="$PROCESSED_DATA_PATH" \
    --use_processed_data="$USE_PROCESSED_DATA" \
    --num_cores="$NUM_CORES" \
    --min_freq=1 \
    --seed=42
}

# Function to run the non pre-split data processing
run_nonsplit() {
  echo "Running non pre-split data processing..."
  echo "SPLIT: $SPLIT"
  python templ_rel_preprocessor.py \
    --model_name="template_relevance" \
    --data_name="$DATA_NAME" \
    --log_file="template_relevance_preprocess_$DATA_NAME" \
    --all_reaction_file="$ALL_REACTION_FILE" \
    --processed_data_path="$PROCESSED_DATA_PATH" \
    --num_cores="$NUM_CORES" \
    --split_ratio="$SPLIT" \
    --min_freq=1 \
    --seed=42
}

# Check if raw_train.csv exists to determine the preprocessing step
if [[ -f "$TRAIN_FILE" ]]; then
  run_presplit
else
  run_nonsplit
fi
#!/bin/bash
python templ_rel_consolidator.py \
  --model_name="template_relevance" \
  --data_name="$DATA_NAME" \
  --log_file="template_relevance_consolidate_$DATA_NAME" \
  --all_reaction_file="$ALL_REACTION_FILE" \
  --train_file="$TRAIN_FILE" \
  --val_file="$VAL_FILE" \
  --test_file="$TEST_FILE" \
  --processed_data_path="$PROCESSED_DATA_PATH" \
  --num_cores="$NUM_CORES" \
  --split_ratio="$SPLIT" \
  --min_freq=1 \
  --seed=42
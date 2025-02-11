#!/bin/bash
echo "Topk: $TOPK"
echo "Optimistic Ranking: $IS_OPTIMISTIC_RANKING"

python templ_rel_predictor.py \
  --backend=nccl \
  --model_name="template_relevance" \
  --data_name="$DATA_NAME" \
  --log_file="template_relevance_predict_$DATA_NAME" \
  --test_file="$TEST_FILE" \
  --processed_data_path="$PROCESSED_DATA_PATH" \
  --model_path="$MODEL_PATH" \
  --test_output_path="$TEST_OUTPUT_PATH" \
  --num_cores="$NUM_CORES" \
  --test_batch_size=1024 \
  --topk=$TOPK \
  --max_num_templ=$MAX_NUM_TEMPL \
  --is_optimistic_ranking="$IS_OPTIMISTIC_RANKING" \
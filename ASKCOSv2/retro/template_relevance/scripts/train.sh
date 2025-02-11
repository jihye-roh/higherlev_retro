export BATCH_SIZE=1024
export NUM_NODES=1
export NUM_GPU=1
export DROPOUT=0.3
export LR=0.001
export HIDDEN_SIZES="1024,1024"
export HIDDEN_ACTIVATION="relu"

python templ_rel_trainer.py \
  --backend=nccl \
  --model_name="template_relevance" \
  --data_name="$DATA_NAME" \
  --log_file="template_relevance_train_$DATA_NAME" \
  --processed_data_path="$PROCESSED_DATA_PATH" \
  --model_path="$MODEL_PATH" \
  --dropout="$DROPOUT" \
  --seed=42 \
  --epochs=150 \
  --hidden_activation="$HIDDEN_ACTIVATION" \
  --hidden_sizes="$HIDDEN_SIZES" \
  --learning_rate="$LR" \
  --train_batch_size=$((BATCH_SIZE / NUM_NODES / NUM_GPU)) \
  --val_batch_size=$((BATCH_SIZE / NUM_NODES / NUM_GPU)) \
  --test_batch_size=$((BATCH_SIZE / NUM_NODES / NUM_GPU)) \
  --num_cores="$NUM_CORES" \
  --lr_scheduler_factor 0.3 \
  --lr_scheduler_patience 1 \
  --early_stop \
  --early_stop_patience 2 \

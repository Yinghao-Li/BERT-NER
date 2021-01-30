#!/bin/bash

# Author: Yinghao Li @ Georgia Tech ECE
# All rights reserved

# -------------------------------------
# This script is used to run txt_gen_train.py
# -------------------------------------

# Quit if there's any errors
set -e

DATASET=Co03
EPOCH=10
BATCH_SIZE=24
MAX_SEQ_LEN=256
OUTPUT_DIR=./Co03-output
SEED=0

for DATA_RATIO in 0.5 0.1 0.05 0.01
do
  CUDA_VISIBLE_DEVICES=$1 python bert_ner.py \
      --data_dir ../data/ \
      --dataset_name $DATASET \
      --model_name_or_path bert-base-uncased \
      --output_dir $OUTPUT_DIR \
      --max_seq_length $MAX_SEQ_LEN \
      --num_train_epochs $EPOCH \
      --per_device_train_batch_size $BATCH_SIZE \
      --per_device_eval_batch_size $BATCH_SIZE \
      --save_steps 999999999 \
      --seed $SEED \
      --do_train \
      --do_eval \
      --do_predict \
      --overwrite_output_dir \
      --overwrite_cache \
      --data_ratio $DATA_RATIO
done

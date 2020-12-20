#!/bin/bash

# Author: Yinghao Li @ Georgia Tech ECE
# All rights reserved

# -------------------------------------
# This script is used to run txt_gen_train.py
# -------------------------------------

# Quit if there's any errors
set -e

DATASET=NCBI
EPOCH=100
BATCH_SIZE=10
MAX_SEQ_LEN=512
OUTPUT_DIR=./NCBI-output
MODEL=dmis-lab/biobert-v1.1

for SEED in 0 42 24601 234 476
do
  CUDA_VISIBLE_DEVICES=$1 python bert_ner.py \
      --data_dir ../data/ \
      --dataset_name $DATASET \
      --weak_src nhmm \
      --model_name_or_path $MODEL \
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
      --overwrite_output_dir
done

# ----------------------------------------

for SEED in 0 42 24601 234 476
do
  CUDA_VISIBLE_DEVICES=$1 python bert_ner.py \
      --data_dir ../data/ \
      --dataset_name $DATASET \
      --weak_src hmm \
      --model_name_or_path $MODEL \
      --output_dir $OUTPUT_DIR  \
      --max_seq_length $MAX_SEQ_LEN \
      --num_train_epochs $EPOCH \
      --per_device_train_batch_size $BATCH_SIZE \
      --per_device_eval_batch_size $BATCH_SIZE \
      --save_steps 999999999 \
      --seed $SEED \
      --do_train \
      --do_eval \
      --do_predict \
      --overwrite_output_dir
done

# ----------------------------------------

for SEED in 0 42 24601 234 476
do
  CUDA_VISIBLE_DEVICES=$1 python bert_ner.py \
      --data_dir ../data/ \
      --dataset_name $DATASET \
      --weak_src majority \
      --model_name_or_path $MODEL \
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
      --overwrite_output_dir
done

# ----------------------------------------

for SEED in 0 42 24601 234 476
do
  CUDA_VISIBLE_DEVICES=$1 python bert_ner.py \
      --data_dir ../data/ \
      --dataset_name $DATASET \
      --weak_src iid \
      --model_name_or_path $MODEL \
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
      --overwrite_output_dir
done

# ----------------------------------------

for SEED in 0 42 24601 234 476
do
  CUDA_VISIBLE_DEVICES=$1 python bert_ner.py \
      --data_dir ../data/ \
      --dataset_name $DATASET \
      --model_name_or_path $MODEL \
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
      --overwrite_output_dir
done

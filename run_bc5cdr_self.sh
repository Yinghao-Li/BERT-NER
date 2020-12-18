#!/bin/bash

# Author: Yinghao Li @ Georgia Tech ECE
# All rights reserved

# -------------------------------------
# This script is used to run txt_gen_train.py
# -------------------------------------

# Quit if there's any errors
set -e

DATASET=BC5CDR
EPOCH=100
BATCH_SIZE=10
MAX_SEQ_LEN=512
SELF_TRAINING_START_EPOCH=40
TEACHER_UPDATE_PERIOD=3
OUTPUT_DIR=./BC5CDR-self-threshold
MODEL=allenai/scibert_scivocab_uncased

for SEED in 24601 234 123
do
  CUDA_VISIBLE_DEVICES=$1 python self_train.py \
      --data_dir ../data/ \
      --dataset_name $DATASET \
      --weak_src nhmm \
      --model_name_or_path $MODEL \
      --output_dir $OUTPUT_DIR \
      --max_seq_length $MAX_SEQ_LEN \
      --num_train_epochs $EPOCH \
      --per_device_train_batch_size $BATCH_SIZE \
      --per_device_eval_batch_size $BATCH_SIZE \
      --self_training_start_epoch $SELF_TRAINING_START_EPOCH \
      --teacher_update_period $TEACHER_UPDATE_PERIOD \
      --save_steps 999999999 \
      --seed $SEED \
      --do_train \
      --do_eval \
      --do_predict \
      --overwrite_output_dir
done

# ----------------------------------------

for SEED in 24601 234 123
do
  CUDA_VISIBLE_DEVICES=$1 python self_train.py \
      --data_dir ../data/ \
      --dataset_name $DATASET \
      --weak_src hmm \
      --model_name_or_path $MODEL \
      --output_dir $OUTPUT_DIR  \
      --max_seq_length $MAX_SEQ_LEN \
      --num_train_epochs $EPOCH \
      --per_device_train_batch_size $BATCH_SIZE \
      --per_device_eval_batch_size $BATCH_SIZE \
      --self_training_start_epoch $SELF_TRAINING_START_EPOCH \
      --teacher_update_period $TEACHER_UPDATE_PERIOD \
      --save_steps 999999999 \
      --seed $SEED \
      --do_train \
      --do_eval \
      --do_predict \
      --overwrite_output_dir
done

# ----------------------------------------

for SEED in 24601 234 123
do
  CUDA_VISIBLE_DEVICES=$1 python self_train.py \
      --data_dir ../data/ \
      --dataset_name $DATASET \
      --weak_src majority \
      --model_name_or_path $MODEL \
      --output_dir $OUTPUT_DIR \
      --max_seq_length $MAX_SEQ_LEN \
      --num_train_epochs $EPOCH \
      --per_device_train_batch_size $BATCH_SIZE \
      --per_device_eval_batch_size $BATCH_SIZE \
      --self_training_start_epoch $SELF_TRAINING_START_EPOCH \
      --teacher_update_period $TEACHER_UPDATE_PERIOD \
      --save_steps 999999999 \
      --seed $SEED \
      --do_train \
      --do_eval \
      --do_predict \
      --overwrite_output_dir
done

# ----------------------------------------

for SEED in 24601 234 123
do
  CUDA_VISIBLE_DEVICES=$1 python self_train.py \
      --data_dir ../data/ \
      --dataset_name $DATASET \
      --weak_src iid \
      --model_name_or_path $MODEL \
      --output_dir $OUTPUT_DIR \
      --max_seq_length $MAX_SEQ_LEN \
      --num_train_epochs $EPOCH \
      --per_device_train_batch_size $BATCH_SIZE \
      --per_device_eval_batch_size $BATCH_SIZE \
      --self_training_start_epoch $SELF_TRAINING_START_EPOCH \
      --teacher_update_period $TEACHER_UPDATE_PERIOD \
      --save_steps 999999999 \
      --seed $SEED \
      --do_train \
      --do_eval \
      --do_predict \
      --overwrite_output_dir
done

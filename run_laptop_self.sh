#!/bin/bash

# Author: Yinghao Li @ Georgia Tech ECE
# All rights reserved

# -------------------------------------
# This script is used to run txt_gen_train.py
# -------------------------------------

# Quit if there's any errors
set -e

DATASET=Laptop
EPOCH=100
BATCH_SIZE=64
MAX_SEQ_LEN=128
SELF_TRAINING_START_EPOCH=60
TEACHER_UPDATE_PERIOD=3
LR=0.00001
OUTPUT_DIR=./Laptop-self-threshold

for SEED in 24601 234 123
do
  CUDA_VISIBLE_DEVICES=$1 python self_train.py \
      --data_dir ../data/ \
      --dataset_name $DATASET \
      --learning_rate $LR \
      --weak_src nhmm \
      --model_name_or_path bert-base-uncased \
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
      --learning_rate $LR \
      --weak_src hmm \
      --model_name_or_path bert-base-uncased \
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
      --learning_rate $LR \
      --weak_src majority \
      --model_name_or_path bert-base-uncased \
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
      --learning_rate $LR \
      --weak_src iid \
      --model_name_or_path bert-base-uncased \
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

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
SELF_TRAINING_START_EPOCH=30
TEACHER_UPDATE_PERIOD=3

for SEED in 0 42 24601 234 476
do
  CUDA_VISIBLE_DEVICES=$1 python self_train.py \
      --data_dir ../data/ \
      --dataset_name $DATASET \
      --weak_src nhmm \
      --model_name_or_path bert-base-uncased \
      --output_dir ./Laptop-output \
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

for SEED in 0 42 24601 234 476
do
  CUDA_VISIBLE_DEVICES=$1 python bert_ner.py \
      --data_dir ../data/ \
      --dataset_name $DATASET \
      --weak_src hmm \
      --model_name_or_path bert-base-uncased \
      --output_dir ./Laptop-output \
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

for SEED in 0 42 24601 234 476
do
  CUDA_VISIBLE_DEVICES=$1 python bert_ner.py \
      --data_dir ../data/ \
      --dataset_name $DATASET \
      --weak_src majority \
      --model_name_or_path bert-base-uncased \
      --output_dir ./Laptop-output \
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

for SEED in 0 42 24601 234 476
do
  CUDA_VISIBLE_DEVICES=$1 python bert_ner.py \
      --data_dir ../data/ \
      --dataset_name $DATASET \
      --weak_src iid \
      --model_name_or_path bert-base-uncased \
      --output_dir ./Laptop-output \
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

for SEED in 0 42 24601 234 476
do
  CUDA_VISIBLE_DEVICES=$1 python bert_ner.py \
      --data_dir ../data/ \
      --dataset_name $DATASET \
      --model_name_or_path bert-base-uncased \
      --output_dir ./Laptop-output \
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

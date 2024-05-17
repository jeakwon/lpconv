#!/bin/bash
cd ./transfer_learning

TRAINER=CONVNET
SHOTS=16 # 16


DATASET=oxford_flowers

for SEED in 1
do
    CUDA_VISIBLE_DEVICES=5 python3 train.py \
    --root /data1/lsj9862/data \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/BASE_CONV/convnextv2.yaml \
    --output-dir /data1/lsj9862/lpconv/transfer_learning/exp_result/${DATASET}_${SHOTS}shot/seed${SEED}/BASE_CONV \
    --use_wandb \
    TRAIN.CHECKPOINT_FREQ 500 \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES all
done
#!/usr/bin/env bash
seeds=(1 2 3 4 5)
gard_iter=1
lr=2e-3
bert_lr=2e-5
droupout=0.1
gpu=$1
mode=$2
val_step=$3
early_stop=$4
checkpoint_id=$5

for s in ${seeds[@]};do
    CUDA_VISIBLE_DEVICES=${gpu} python3 train_demo.py  \
    --lr ${lr} \
    --K 1 \
    --mode ${mode} \
    --dataset cross-dataset \
    --seed ${s} \
    --bert_lr ${bert_lr} \
    --dropout ${droupout} \
    --grad_iter ${gard_iter} \
    --bert_path bert-base-uncased \
    --warmup_step 400 \
    --num_heads 1 \
    --max_o_num 1000 \
    --early_stop $4 \
    --hidsize 100 \
    --eposide_tasks 5 \
    --val_step ${val_step} \
    --max_epoch 10 \
    --batch_size 1 \
    --checkpoint_id ${checkpoint_id} \
    --shuffle 
done



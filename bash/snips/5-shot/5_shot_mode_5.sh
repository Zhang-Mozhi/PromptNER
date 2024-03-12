#!/usr/bin/env bash
seeds=(1 2 3 4 5)
mode=5
bert_lr=5e-5
gard_iter=1
gpu=$1
lr=5e-4
droupout=0.1
val_step=$2
early_stop=$3
checkpoint_id=$4
threshold=$5
finetuning_steps=$6
kNN_ratio=${7}
just_predict=${8}

for s in ${seeds[@]};do
     CUDA_VISIBLE_DEVICES=${gpu} python3 train_demo.py  \
    --lr ${lr} \
    --K 5 \
    --mode ${mode} \
    --dataset snips \
    --seed ${s} \
    --bert_lr ${bert_lr} \
    --dropout ${droupout} \
    --grad_iter ${gard_iter} \
    --bert_path bert-base-uncased \
    --warmup_step 1000 \
    --num_heads 1 \
    --max_o_num 1000 \
    --early_stop 6 \
    --hidsize 100 \
    --eposide_tasks 10 \
    --max_epoch 10 \
    --batch_size 1 \
    --shuffle \
    --val_step ${val_step} \
    --early_stop ${early_stop} \
    --checkpoint_id ${checkpoint_id} \
    --threshold ${threshold} \
    --finetuning_steps ${finetuning_steps} \
    --kNN_ratio ${kNN_ratio} \
    --just_predict ${just_predict}
done



#!/usr/bin/env bash
N=$3
K=$4
seeds=(1 2 3)
modes=$2
bert_lr=2e-5
gard_iter=1
gpu=$1
val_step=$5
orilr=5e-4
lr=2e-3
droupout=0.1
early_stop=$6
checkpoint_id=$7
threshold=$8
finetuning_steps=$9
kNN_ratio=${10}
dataset=${11}
just_predict=${12}
prompt_id=${13}

# shellcheck disable=SC2068
for k in ${K[@]};do
  for mode in ${modes[@]};do
    for s in ${seeds[@]};do
        CUDA_VISIBLE_DEVICES=${gpu} python3 train_demo.py  \
        --lr ${lr} \
        --N ${N} \
        --K ${k} \
        --mode ${mode} \
        --dataset ${dataset} \
        --warmup_step 100 \
        --seed ${s} \
        --num_heads 1 \
        --bert_lr ${bert_lr} \
        --dropout ${droupout} \
        --grad_iter ${gard_iter} \
        --bert_path bert-base-uncased \
        --hidsize 100 \
        --val_step ${val_step} \
        --early_stop ${early_stop} \
        --checkpoint_id ${checkpoint_id} \
        --threshold ${threshold} \
        --finetuning_steps ${finetuning_steps} \
        --kNN_ratio ${kNN_ratio} \
        --just_predict ${just_predict} \
        --prompt_id ${prompt_id}
    done
  done
  done
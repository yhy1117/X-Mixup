#!/usr/bin/env bash
set -e
repo=$PWD
pretrained_model=${1:-xlmr}
task_name=${2:-mlqa}
start_layer=${3:-15}
end_layer=${4:-16}
data_dir=${5:-"$REPO/data"}
output_dir=${6:-"$REPO/outputs"}

if [ ${task_name} == "mlqa" ];
then
  eval_languages="ar,de,en,es,hi,vi,zh"
  num_train_epochs=2
  lr=5e-6
elif [ ${task_name} == "xquad" ];
then
  eval_languages="ar,de,el,en,es,hi,ru,th,tr,vi,zh"
  num_train_epochs=2
  lr=5e-6
elif [ ${task_name} == "tydiqa" ];
then
  eval_languages="ar,bn,en,fi,id,ko,ru,sw,te"
  num_train_epochs=4
  lr=3e-5
fi
train_languages=${eval_languages}

max_seq_length=384
doc_stride=128
batch_size=4
save_steps=100
eval_steps=100
save_total_limit=3
seed=42
warmup_prop=0.1
threads=36
alpha=0.2
p_k=2000

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 ${repo}/xmixup/entry/run_squad.py --data_dir ${data_dir} \
--model_name_or_path ${pretrained_model} \
--output_dir ${output_dir} \
--logging_dir ${output_dir}/run.log \
--task_name ${task_name} \
--train_languages ${train_languages} \
--eval_languages ${eval_languages} \
--max_seq_length  ${max_seq_length} \
--doc_stride ${doc_stride} \
--learning_rate ${lr} \
--num_train_epochs ${num_train_epochs} \
--per_device_train_batch_size ${batch_size} \
--per_device_eval_batch_size ${batch_size} \
--threads ${threads} \
--save_steps ${save_steps} \
--eval_steps ${eval_steps} \
--logging_steps ${eval_steps} \
--save_total_limit ${save_total_limit} \
--warmup_prop ${warmup_prop} \
--seed ${seed} \
--do_train \
--do_eval \
--do_predict \
--evaluate_during_training \
--overwrite_output_dir \
--alpha ${alpha} \
--start_layer ${start_layer} \
--end_layer ${end_layer} \
--mixup_inference \
--p_k ${p_k}

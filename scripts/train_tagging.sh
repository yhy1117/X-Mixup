#!/usr/bin/env bash
set -e
repo=$PWD
pretrained_model=${1:-xlmr}
task_name=${2:-udpos}
start_layer=${3:-3}
end_layer=${4:-4}
data_dir=${5:-"$REPO/data"}
output_dir=${6:-"$REPO/outputs"}

if [ ${task_name} == "udpos" ];
then
  eval_languages="af,ar,bg,de,el,en,es,et,eu,fa,fi,fr,he,hi,hu,id,it,ja,kk,ko,mr,nl,pt,ru,ta,te,th,tl,tr,ur,vi,yo,zh"

elif [ ${task_name} == "panx" ];
then
  eval_languages="af,ar,bg,bn,de,el,en,es,et,eu,fa,fi,fr,he,hi,hu,id,it,ja,jv,ka,kk,ko,ml,mr,ms,my,nl,pt,ru,sw,ta,te,th,tl,tr,ur,vi,yo,zh"
fi
train_languages=${eval_languages}

max_seq_length=128
num_train_epochs=4
batch_size=16
save_steps=100
eval_steps=100
save_total_limit=3
seed=1
lr=5e-6
alpha=0.8
p_k=1000

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ${repo}/xmixup/entry/run_tagging.py --data_dir ${data_dir}/${task_name} \
--model_name_or_path ${pretrained_model} \
--output_dir ${output_dir} \
--logging_dir ${output_dir}/run.log \
--task_name ${task_name} \
--labels "labels.txt" \
--train_languages ${train_languages} \
--eval_languages ${eval_languages} \
--max_seq_length  ${max_seq_length} \
--learning_rate ${lr} \
--num_train_epochs ${num_train_epochs} \
--per_device_train_batch_size ${batch_size} \
--per_device_eval_batch_size ${batch_size} \
--save_steps ${save_steps} \
--eval_steps ${eval_steps} \
--logging_steps ${eval_steps} \
--save_total_limit ${save_total_limit} \
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

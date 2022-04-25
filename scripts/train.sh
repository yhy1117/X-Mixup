#!/usr/bin/env bash
set -e
repo=$PWD
pretrained_model=${1:-xlmr}
task_name=${2:-pawsx}
data_dir=${3:-"$REPO/data"}
output_dir=${4:-"$REPO/outputs"}

if [ ${pretrained_model} == "mbert" ];
then
  pretrained_model="bert-base-multilingual-cased"
elif [ ${pretrained_model} == "xlmr" ];
then
  pretrained_model="xlm-roberta-large"
else
  echo "Unsupported model: $pretrained_model";
fi

if [ ${task_name} == "xnli" ] || [ ${task_name} == "pawsx" ];
then
  bash ${repo}/scripts/train_classification.sh $pretrained_model $task_name 0 1 $data_dir $output_dir
elif [ ${task_name} == "udpos" ] || [ ${task_name} == "panx" ];
then
  bash ${repo}/scripts/train_tagging.sh $pretrained_model $task_name 3 4 $data_dir $output_dir
elif [ ${task_name} == "mlqa" ] || [ ${task_name} == "xquad" ] || [ ${task_name} == "tydiqa" ];
then
  bash ${repo}/scripts/train_qa.sh $pretrained_model $task_name 15 16 $data_dir $output_dir
else
  echo "Unsupported task: $task_name";
fi

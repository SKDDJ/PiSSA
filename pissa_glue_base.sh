#!/bin/bash
# hp from lora
export TRANSFORMERS_CACHE=/root/.cache/huggingface/hub
export HF_HOME=/root/.cache/huggingface
export XDG_CACHE_HOME=/root/.cache

# LATEST LoRA config

declare -A epochs=(["mnli"]=30 ["sst2"]=60 ["mrpc"]=30 ["cola"]=80 ["qnli"]=25 ["qqp"]=25 ["rte"]=80  ["stsb"]=40 )

declare -A bs=(["mnli"]=64 ["sst2"]=64 ["mrpc"]=64 ["cola"]=64 ["qnli"]=64 ["qqp"]=64 ["rte"]=64  ["stsb"]=64 )

declare -A ml=(["mnli"]=512 ["sst2"]=512 ["mrpc"]=512 ["cola"]=512 ["qnli"]=512 ["qqp"]=512 ["rte"]=512  ["stsb"]=512 )
# here I noticed that the learning rate for mnli is 5e-4 will crash the mnli task, so I will reduce it to 4e-4
declare -A lr=(["mnli"]="5e-4" ["sst2"]="5e-4" ["mrpc"]="4e-4" ["cola"]="4e-4" ["qnli"]="4e-4" ["qqp"]="5e-4" ["rte"]="5e-4"  ["stsb"]="4e-4" )

declare -A metrics=(["mnli"]="accuracy" ["mrpc"]="accuracy" ["qnli"]="accuracy" ["qqp"]="accuracy" ["rte"]="accuracy" ["sst2"]="accuracy" ["stsb"]="pearson" ["cola"]="matthews_correlation")

# export WANDB_MODE=offline

run(){
  task_name=$1
  learning_rate=${lr[$1]}
  num_train_epochs=${epochs[$1]}
  per_device_train_batch_size=${bs[$1]}
  rank=8 # paper # rank = 8
  l_num=12
  seed=42
  lora_alpha="8"
  target_modules="query value"
  lora_dropout=0.
  lora_bias=none
  lora_task_type=SEQ_CLS
  export WANDB_PROJECT=5-5-bf16-pissa_base_hp_lora_glue
  export WANDB_NAME=base-pissa-${task_name}-r-${rank}-target_modules-${target_modules}-seed-${seed}-bs-${per_device_train_batch_size}-lr-${learning_rate}-epochs-${num_train_epochs}

  HF_ENDPOINT=https://hf-mirror.com accelerate launch --num_processes 4 --main_process_port 26697 ./run_glue_pissa.py \
  --model_name_or_path FacebookAI/roberta-base  \
  --task_name ${task_name} \
  --do_train --do_eval \
  --max_seq_length ${ml[$1]} \
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --per_device_eval_batch_size ${per_device_train_batch_size} \
  --load_best_model_at_end True --metric_for_best_model ${metrics[$1]} \
  --learning_rate ${learning_rate} \
  --num_train_epochs ${num_train_epochs} \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --weight_decay 0. \
  --warmup_ratio 0.06 \
  --logging_steps 10 \
  --seed ${seed} \
  --lora_alpha ${lora_alpha} --lora_dropout ${lora_dropout} --lora_bias ${lora_bias} \
  --target_modules ${target_modules} --rank ${rank} \
  --lora_task_type ${lora_task_type}  \
  --output_dir ${exp_dir}/model \
  --logging_dir ${exp_dir}/log \
  --overwrite_output_dir
}

task_base=('cola' 'mrpc' 'mnli' 'qqp' 'qnli' 'rte' 'sst2' 'stsb' )

# 707w params cola for 50mins... for 4 cards

for task in "${task_base[@]}"; do
    run $task
done  
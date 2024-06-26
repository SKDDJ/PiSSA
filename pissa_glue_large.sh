#!/bin/bash
export HF_HOME=/root/.cache/huggingface
export XDG_CACHE_HOME=/root/.cache

# LATEST
# roberta large LoRA 
declare -A epochs=(["mnli"]=10 ["sst2"]=10 ["mrpc"]=20 ["cola"]=20 ["qnli"]=10 ["qqp"]=20 ["rte"]=20  ["stsb"]=30 )


# roberta large LoRA
declare -A bs=(["mnli"]=32 ["sst2"]=32 ["mrpc"]=32 ["cola"]=32 ["qnli"]=32 ["qqp"]=32 ["rte"]=32  ["stsb"]=32 )


# roberta large LoRA
declare -A ml=(["mnli"]=128 ["sst2"]=128 ["mrpc"]=512 ["cola"]=128 ["qnli"]=512 ["qqp"]=512 ["rte"]=512  ["stsb"]=512 )

# roberta large LoRA
declare -A lr=(["mnli"]="4e-4" ["sst2"]="4e-4" ["mrpc"]="3e-4" ["cola"]="2e-4" ["qnli"]="2e-4" ["qqp"]="3e-4" ["rte"]="4e-4"  ["stsb"]="2e-4" )

declare -A metrics=(["mnli"]="accuracy" ["mrpc"]="accuracy" ["qnli"]="accuracy" ["qqp"]="accuracy" ["rte"]="accuracy" ["sst2"]="accuracy" ["stsb"]="pearson" ["cola"]="matthews_correlation")

export WANDB_MODE=offline

run(){
  task_name=$1
  learning_rate=${lr[$1]}
  num_train_epochs=${epochs[$1]}
  per_device_train_batch_size=${bs[$1]}
  rank=8 # paper # rank = 8
  l_num=12
  seed=42
  lora_alpha="16"
  target_modules="query value"
  lora_dropout=0.
  lora_bias=none
  lora_task_type=SEQ_CLS
  export WANDB_PROJECT=5-5-bf16-pissa_large_hp_lora_glue
  export WANDB_NAME=large-pissa-${task_name}-r-${rank}-target_modules-${target_modules}-seed-${seed}-bs-${per_device_train_batch_size}-lr-${learning_rate}-epochs-${num_train_epochs}

  HF_ENDPOINT=https://hf-mirror.com accelerate launch --num_processes 6 --main_process_port 26698 ./run_glue_pissa.py \
  --model_name_or_path FacebookAI/roberta-large  \
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

# task_base=('cola' 'mrpc' 'mnli' 'qqp' 'qnli' 'rte' 'sst2' 'stsb' )
task_base=('mnli' )

for task in "${task_base[@]}"; do
    run $task
done  

#!/bin/bash

# 实际这个脚本应该是git clone的pissa的peft，然后用的是melora的训练roberta的脚本
export WANDB_MODE=offline

run(){
  task_name="cola"
  # vera lr = 2e-2 classifer lr = 1e-2
  learning_rate=1e-2
  num_train_epochs=80
  per_device_train_batch_size=64
  rank=768
  l_num=12
  seed=42
  lora_alpha="768"
  target_modules="query value key"
  mode="base"
  lora_dropout=0.
  lora_bias=none
  lora_task_type=SEQ_CLS
  wandb_project=new_vector_z
  share=false
  wandb_run_name=roberta-lora
  # vera max seq lenght = 128
  exp_dir=../roberta_glue_reproduce/${wandb_run_name}
  #  CUDA_VISIBLE_DEVICES=0,1,2,4
  HF_ENDPOINT=https://hf-mirror.com accelerate launch ./run_glue_lora.py \
  --model_name_or_path FacebookAI/roberta-base  \
  --task_name ${task_name} \
  --do_train --do_eval \
  --max_seq_length 128 \ 
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --per_device_eval_batch_size ${per_device_train_batch_size} \
  --load_best_model_at_end True --metric_for_best_model "matthews_correlation" \
  --learning_rate ${learning_rate} \
  --num_train_epochs ${num_train_epochs} \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --weight_decay 0. \
  --warmup_ratio 0.06 \
  --logging_steps 10 \
  --seed ${seed} --wandb_project ${wandb_project} \
  --lora_alpha ${lora_alpha} --lora_dropout ${lora_dropout} --lora_bias ${lora_bias} \
  --lora_task_type ${lora_task_type} --target_modules ${target_modules} --rank ${rank} \
  --l_num ${l_num} --mode "base" \
  --output_dir ${exp_dir}/model \
  --logging_dir ${exp_dir}/log \
  --run_name ${wandb_run_name} \
  --overwrite_output_dir
}
task_base=('cola')

run $task_base[0]

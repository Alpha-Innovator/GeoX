BASE_ROOT = 
output_model=lm_pretrain
if [ ! -d ${output_model} ];then  
    mkdir ${output_model}
fi
cp ./pretrain.sh ${output_model}
cp ./ds_config_zero*.json ${output_model}
# export CUDA_HOME=/usr/local/cuda/
export NCCL_P2P_DISABLE=1

deepspeed --include pretrain_clm.py \
    --config_name  /cpfs01/user/xiarenqiu/limingsheng/llemma-7b/config.json \
    --tokenizer_name /cpfs01/user/xiarenqiu/limingsheng/llemma-7b \
    --model_name_or_path /cpfs01/user/xiarenqiu/limingsheng/llemma-7b \
    --train_files /cpfs01/user/xiarenqiu/limingsheng/data/language_only_data/lm_pretrain_final.txt \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 32 \
    --do_train \
    --output_dir ${output_model} \
    --evaluation_strategy steps \
    --use_fast_tokenizer false \
    --max_eval_samples 0 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 10 \
    --warmup_ratio 0.1 \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 1000 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps 20000000 \
    --eval_steps 500000000 \
    --save_total_limit 2000 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 1024 \
    --overwrite_output_dir \
    --report_to tensorboard \
    --run_name ${output_model} \
    --bf16 \
    --bf16_full_eval \
    --gradient_checkpointing \
    --deepspeed ./ds_config_zero3.json \
    --ignore_data_skip true \
    --ddp_timeout 18000000 \
    | tee -a ${output_model}/train.log

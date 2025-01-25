export MASTER_PORT=20728
wandb off

deepspeed --include=localhost:0,1,2,3 --master_port=$MASTER_PORT main/train_geox.py \
    --deepspeed ./configs/zero2.json \
    --model_name_or_path /path/to/geo-llm-ckpt \
    --version geo_v1 \
    --data_path ./data/geoqa/train.json \
    --image_folder ./data/geoqa/images \
    --vision_tower /path/to/geo-vit-ckpt \
    --gsformer_path /path/to/gsformer-ckpt \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./experiments/geox_geoqa \
    --num_train_epochs 100 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 100 \
    --learning_rate 3e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True


deepspeed --include=localhost:0,1,2,3 --master_port=$MASTER_PORT main/train_geox.py \
    --deepspeed ./configs/zero2.json \
    --model_name_or_path /path/to/geo-llm-ckpt \
    --version geo_v1 \
    --data_path ./data/unigeo/train.json \
    --image_folder ./data/unigeo/images \
    --vision_tower /path/to/geo-vit-ckpt \
    --gsformer_path /path/to/gsformer-ckpt \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./experiments/geox_unigeo \
    --num_train_epochs 80 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 100 \
    --learning_rate 3e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True




deepspeed --include=localhost:1,2,3,4 --master_port=$MASTER_PORT main/train_geox.py \
    --deepspeed ./configs/zero2.json \
    --model_name_or_path /path/to/geo-llm-ckpt \
    --version geo_v1 \
    --data_path ./data/geometry3k/train.json \
    --image_folder ./data/geometry3k/images \
    --vision_tower /path/to/geo-vit-ckpt \
    --gsformer_path /path/to/gsformer-ckpt \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./experiments/geox_geometry3k \
    --num_train_epochs 30 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1 \
    --save_total_limit 100 \
    --learning_rate 6e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True



deepspeed --include=localhost:0,1,2,3 --master_port=$MASTER_PORT main/train_geox.py \
    --deepspeed ./configs/zero2.json \
    --model_name_or_path /path/to/geo-llm-ckpt \
    --version geo_v1 \
    --data_path ./data/pgps9k/train.json \
    --image_folder ./data/pgps9k/images \
    --vision_tower /path/to/geo-vit-ckpt \
    --gsformer_path /path/to/gsformer-ckpt \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./experiments/geox_pgps9k \
    --num_train_epochs 45 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 100 \
    --learning_rate 6e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True
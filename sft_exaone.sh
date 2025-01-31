
accelerate launch src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-Math-1.5B \
    --dataset_name "heegyu/Bespoke-Stratos-35k-messages,heegyu/OpenO1-SFT-77k-messages-bespoke" \
    --learning_rate 1e-4 \
    --packing \
    --num_train_epochs 3 \
    --max_seq_length 16384 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --trust_remote_code \
    --lr_scheduler_type cosine \
    --warmup_steps 10 \
    --use_peft \
    --lora_target_modules all-linear \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy no \
    --save_strategy epoch \
    --output_dir ./checkpoint/open-r1/Qwen2.5-Math-1.5B-Open-R1-Distill-lora \
    --push_to_hub --hub_model_id "heegyu/Qwen2.5-Math-1.5B-Open-R1-Distill-lora"

accelerate launch src/open_r1/sft.py \
    --model_name_or_path LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --trust_remote_code \
    --bf16 \
    --use_peft False \
    --lora_target_modules all-linear \
    --logging_steps 5 \
    --eval_strategy no \
    --eval_steps 100 \
    --output_dir /data3/heegyu/checkpoint/open-r1/EXAONE-3.5-7.8B-Instruct-Open-R1-Distill

    
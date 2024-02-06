set -x

read -r -d '' training_commands <<EOF
../train_sft.py \
    --max_len 4096 \
    --dataset datasets/train.parquet \
    --dataset_probs 1 \
    --train_batch_size 8 \
    --micro_train_batch_size 1 \
    --max_samples 715 \
    --pretrain models/llama_2_70b_chat_hf \
    --save_path ./ckpt/70b_test \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 2 \
    --max_epochs 1 \
    --bf16 \
    --flash_attn \
    --learning_rate 5e-6 \
    --gradient_checkpointing
    --use_wandb 394c7970d1b0dd8466bae2e50f706f51975930ae
    --wandb_project LLM-FT
    --lora_rank 64
    --lora_alpha 128
EOF
    # --wandb [WANDB_TOKENS]394c7970d1b0dd8466bae2e50f706f51975930ae

if [[ ${1} != "slurm" ]]; then
    export PATH=$HOME/.local/bin/:$PATH
    deepspeed $training_commands
fi
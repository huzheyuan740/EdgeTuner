#export NCCL_DEBUG=INFO --exclude ubt2,ubuntu
# export NCCL_SOCKET_IFNAME='enp3s0'

HF_ENDPOINT=https://hf-mirror.com python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0  src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.sm \
    --init_checkpoint /home/ldmc/hzy/edge_lore/examples/NLG/custom_data/e2e_all_trainset/model.1_lora_1.pt \
    --platform local \
    --lora_dim 8 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir /mnt/data/hzy_data/custom_dataset/e2e_all_trainset \
    --output_file predict.52580.jsonl
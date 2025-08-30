#export NCCL_DEBUG=INFO --exclude ubt2,ubuntu
# export NCCL_SOCKET_IFNAME='enp2s0'

# HF_ENDPOINT=https://hf-mirror.com python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="192.168.131.179" --master_port="29800" src/gpt2_lora.py\

HF_ENDPOINT=https://hf-mirror.com python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="192.168.131.179" --master_port="29800" src/gpt2_lora.py\
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --train_batch_size 4 \
    --grad_acc 1 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card gpt2.md \
    --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
    --platform local \
    --clip 0.0 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 5 \
    --save_interval 1000 \
    --lora_dim 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir ./custom_dataset/e2e_all_trainset \
    --random_seed 110 \
    --client_num 2
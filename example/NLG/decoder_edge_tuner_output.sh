#export NCCL_DEBUG=INFO --exclude ubt2,ubuntu
export NCCL_SOCKET_IFNAME='enp3s0'

# Decode outputs
python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/e2e_all_trainset/predict.70105.b10p08r4.jsonl \
    --input_file ./data/e2e/test_formatted.jsonl \
    --output_ref_file e2e_ref3.txt \
    --output_pred_file e2e_pred3.txt
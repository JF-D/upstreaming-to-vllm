# for arrival_rate in 4096 8192 16384 32768 65536 131072 262144 524288 1048576; do
for max_model_len in 4096; do
model=8
max_num_seqs=128
max_model_len=32768
chunk_size=2048
block_size=32

mkdir -p log/offline_serving
# LOGFILE=log/test.log
LOGFILE=log/offline_serving/llama-${model}b-bs$max_num_seqs-chunk$chunk_size-seq$max_model_len-block$block_size.log

# enable serving to disable logging in model_runner.py
export EVAL_SERVING=1

python examples/benchmark_offline_serving.py \
    --model Meta-Llama-3-${model}B \
    --tp 8 \
    --max-num-seqs $max_num_seqs \
    --max-model-len $max_model_len \
    --chunk-size $chunk_size \
    --block-size $block_size \
    2>&1 | tee $LOGFILE

# python examples/offline_profile.py \
#         --model /home/ubuntu/LLMCKPTs/Meta-Llama-3-${model}B --batch-size 32 \
#         --prompt-len 512 --max-num-batched-tokens 16384 --json Llama31-8b \
#         2>&1 | tee $LOGFILE

# echo "Log to $LOGFILE"

# python examples/parse_log.py --log $LOGFILE

done

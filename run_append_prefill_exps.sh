# for max_num_seqs in 8 32 64 128; do
# for chunk_size in 128 512 1024 2048; do
# for max_model_len in 65536 1024 2048 4096 8192 16384 32768; do
# for max_model_len in 4096 8192 16384 32768 65536 131072 262144 524288 1048576; do
for max_model_len in 32768; do
model=8
max_num_seqs=1
max_model_len=$max_model_len
chunk_size=2048
block_size=32

# LOGFILE=log/test.log
LOGFILE=log/append_prefill_latency/llama-${model}b-bs$max_num_seqs-chunk$chunk_size-seq$max_model_len-block$block_size.log

python examples/benchmark.py \
    --model Meta-Llama-3-${model}B \
    --tp 8 \
    --max-num-seqs $max_num_seqs \
    --max-model-len $max_model_len \
    --chunk-size $chunk_size \
    --block-size $block_size \
    --benchmark-append-prefill \
    # 2>&1 | tee $LOGFILE

# echo "Log to $LOGFILE"

# python examples/parse_log.py --log $LOGFILE

done
# done
# done

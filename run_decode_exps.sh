#!/usr/bin/bash
model=8
max_num_seqs=32
max_model_len=8192
chunk_size=32
block_size=256
num_blocks=2048

mkdir -p log/decode_latency
SUFFIX="-sos-mlp4"
LOGFILE=log/decode_latency/llama-${model}b-bs$max_num_seqs-chunk$chunk_size-seq$max_model_len-block$block_size-nblocks$num_blocks$SUFFIX.log

# NEURON_RT_DISABLE_PACK_DESCRIPTORS=1 \
python examples/benchmark_neuron.py \
    --model Meta-Llama-3-${model}B \
    --max-num-seqs $max_num_seqs \
    --max-model-len $max_model_len \
    --chunk-size $chunk_size \
    --block-size $block_size \
    --num-blocks $num_blocks \
    --mlp-duplicate-degree 4 \
    --sos \
    --layout-opt \
    2>&1 | tee $LOGFILE
echo "Log to $LOGFILE"

python examples/neuron_parse_log.py --log $LOGFILE
python examples/convert_neff.py --log $LOGFILE --profile-dir llama-${model}b-bs$max_num_seqs-chunk$chunk_size-seq$max_model_len-block$block_size-nblocks$num_blocks$SUFFIX

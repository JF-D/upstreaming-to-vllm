#!/usr/bin/bash
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>> Batch Script >>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
for max_num_seqs in 1; do
for chunk_size in 1024 2048 4096 8192 16384 32768; do
# for chunk_size in 16384; do
model=8
# max_num_seqs=2
max_model_len=$chunk_size
# chunk_size=2048
block_size=256
num_blocks=1600 #8192

export FULL_ATTENTION=1

RUN_EXEC_NOSOS=0
RUN_EXEC_NONKI=0
RUN_EXEC_MLP1=1
RUN_EXEC_MLP4=0
RUN_EXEC_MLP8=0
# PARSE_LOG=0

mkdir -p log/full_prefill_latency

SUFFIX="-nosos" #"-sos-attnHSB"
LOGFILE=log/full_prefill_latency/llama-${model}b-bs$max_num_seqs-chunk$chunk_size-seq$max_model_len-block$block_size-nblocks$num_blocks$SUFFIX.log

if [[ $RUN_EXEC_NOSOS == "1" ]]; then
# NEURON_RT_DISABLE_PACK_DESCRIPTORS=1 \
python examples/benchmark_neuron.py \
    --model Meta-Llama-3-${model}B \
    --max-num-seqs $max_num_seqs \
    --max-model-len $max_model_len \
    --chunk-size $chunk_size \
    --block-size $block_size \
    --num-blocks $num_blocks \
    --mlp-duplicate-degree 1 \
    --no-flash-paged-attention \
    2>&1 | tee $LOGFILE
    # --sos \
    # --layout-opt \
echo "Log to $LOGFILE"
fi

# python examples/neuron_parse_log.py --log $LOGFILE
# python examples/convert_neff.py --log $LOGFILE --profile-dir llama-${model}b-bs$max_num_seqs-chunk$chunk_size-seq$max_model_len-block$block_size-nblocks$num_blocks$SUFFIX

SUFFIX="-sos-nonki" #"-sos-attnHSB"
LOGFILE=log/full_prefill_latency/llama-${model}b-bs$max_num_seqs-chunk$chunk_size-seq$max_model_len-block$block_size-nblocks$num_blocks$SUFFIX.log

if [[ $RUN_EXEC_NONKI == "1" ]]; then
# NEURON_RT_DISABLE_PACK_DESCRIPTORS=1 \
python examples/benchmark_neuron.py \
    --model Meta-Llama-3-${model}B \
    --max-num-seqs $max_num_seqs \
    --max-model-len $max_model_len \
    --chunk-size $chunk_size \
    --block-size $block_size \
    --num-blocks $num_blocks \
    --mlp-duplicate-degree 1 \
    --sos \
    --no-flash-paged-attention \
    2>&1 | tee $LOGFILE
    # --layout-opt \
echo "Log to $LOGFILE"
fi

# python examples/neuron_parse_log.py --log $LOGFILE


SUFFIX="-sos-mlp1" #"-sos-attnHSB"
LOGFILE=log/full_prefill_latency/llama-${model}b-bs$max_num_seqs-chunk$chunk_size-seq$max_model_len-block$block_size-nblocks$num_blocks$SUFFIX.log

if [[ $RUN_EXEC_MLP1 == "1" ]]; then
# NEURON_RT_DISABLE_PACK_DESCRIPTORS=1 \
python examples/benchmark_neuron.py \
    --model Meta-Llama-3-${model}B \
    --max-num-seqs $max_num_seqs \
    --max-model-len $max_model_len \
    --chunk-size $chunk_size \
    --block-size $block_size \
    --num-blocks $num_blocks \
    --mlp-duplicate-degree 1 \
    --sos \
    --layout-opt \
    2>&1 | tee $LOGFILE
echo "Log to $LOGFILE"
fi

# python examples/neuron_parse_log.py --log $LOGFILE



SUFFIX="-sos-mlp4" #"-sos-attnHSB"
LOGFILE=log/full_prefill_latency/llama-${model}b-bs$max_num_seqs-chunk$chunk_size-seq$max_model_len-block$block_size-nblocks$num_blocks$SUFFIX.log

if [[ $RUN_EXEC_MLP4 == "1" ]]; then
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
fi

# python examples/neuron_parse_log.py --log $LOGFILE



SUFFIX="-sos-mlp8" #"-sos-attnHSB"
LOGFILE=log/full_prefill_latency/llama-${model}b-bs$max_num_seqs-chunk$chunk_size-seq$max_model_len-block$block_size-nblocks$num_blocks$SUFFIX.log

if [[ $RUN_EXEC_MLP8 == "1" ]]; then
python examples/benchmark_neuron.py \
    --model Meta-Llama-3-${model}B \
    --max-num-seqs $max_num_seqs \
    --max-model-len $max_model_len \
    --chunk-size $chunk_size \
    --block-size $block_size \
    --num-blocks $num_blocks \
    --mlp-duplicate-degree 8 \
    --sos \
    --layout-opt \
    2>&1 | tee $LOGFILE
echo "Log to $LOGFILE"
fi

# python examples/neuron_parse_log.py --log $LOGFILE

done
done

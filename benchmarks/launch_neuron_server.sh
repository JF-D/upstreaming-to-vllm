export NEURONX_DUMP_TO=$PWD/"_compile_cache"
export NEURON_CC_FLAGS=" -O3 --internal-hlo2tensorizer-options=--verify-hlo --internal-enable-dge-levels=vector_dynamic_offsets --disable-internal-io-dge --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' "
export NEURON_RT_DBG_EMBEDDING_UPDATE_BOUND_CHECK="0"
export NEURON_RT_DBG_INDIRECT_MEMCPY_BOUND_CHECK="0"


vllm serve /home/ubuntu/LLMCKPTs/Meta-Llama-3-8B \
    --tensor-parallel-size 32 \
    --max-num-seqs 128 \
    --block-size 256 \
    --num-gpu-blocks-override 3584 \
    --max-model-len 4096 \
    --max-num-batched-tokens 128 \
    --enable-chunked-prefill \
    2>&1 | tee log/llama-8b-chunk128-seq4096-block256-3584-sos.log

# vllm serve /home/ubuntu/LLMCKPTs/Meta-Llama-3-8B \
#     --tensor-parallel-size 32 \
#     --max-num-seqs 128 \
#     --block-size 256 \
#     --num-gpu-blocks-override 14336 \
#     --max-model-len 4096 \
#     --max-num-batched-tokens 128 \
#     --enable-chunked-prefill \
#     2>&1 | tee log/llama-8b.log

# sleep 3600s
# sleep 3600s
# sleep 3600s
# sleep 1800s
# # kill -9 333907

# vllm serve /home/ubuntu/LLMCKPTs/Meta-Llama-3-70B \
#     --tensor-parallel-size 32 \
#     --max-num-seqs 128 \
#     --block-size 256 \
#     --num-gpu-blocks-override 2252 \
#     --max-model-len 4096 \
#     --max-num-batched-tokens 128 \
#     --enable-chunked-prefill \
#     2>&1 | tee log/llama-70b.log

# vllm serve /home/ubuntu/LLMCKPTs/Meta-Llama-3-70B \
#     --tensor-parallel-size 32 \
#     --max-num-seqs 128 \
#     --block-size 256 \
#     --num-gpu-blocks-override 9000 \
#     --max-model-len 4096 \
#     --max-num-batched-tokens 128 \
#     --enable-chunked-prefill \
#     2>&1 | tee log/llama-70b.log
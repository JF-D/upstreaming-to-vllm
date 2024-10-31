import os
os.environ['NEURONX_DUMP_TO'] = os.path.join(os.getcwd(),"_compile_cache")
os.environ["NEURON_CC_FLAGS"]= " -O3 --internal-hlo2tensorizer-options=--verify-hlo --internal-enable-dge-levels=vector_dynamic_offsets --disable-internal-io-dge --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' "
# os.environ["NEURON_CC_FLAGS"]= " -O1 --internal-hlo2tensorizer-options=--verify-hlo --internal-enable-dge-levels=vector_dynamic_offsets "
os.environ["NEURON_RT_DBG_EMBEDDING_UPDATE_BOUND_CHECK"] = "0"
os.environ["NEURON_RT_DBG_INDIRECT_MEMCPY_BOUND_CHECK"] = "0"
# os.environ["NEURON_CC_FLAGS"] += " --internal-compiler-debug-mode=penguin "
# os.environ["HLO_SNAPSHOT_PATH"] = os.path.join(os.getcwd(), "_snapshots")

import time
import random
import numpy as np
import torch
import argparse
import logging
from vllm import LLM, SamplingParams
logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Meta-Llama-3-70B")
parser.add_argument("--max-num-seqs", type=int, default=8)
parser.add_argument("--max-model-len", type=int, default=1024)
parser.add_argument("--chunk-size", type=int, default=128)
parser.add_argument("--sos", action="store_true")
parser.add_argument("--block-size", type=int, default=128)
parser.add_argument("--num-blocks", type=int, default=512)
parser.add_argument("--benchmark", action="store_true")
args = parser.parse_args()

# Create an LLM.
llm = LLM(
    model=f"/home/ubuntu/LLMCKPTs/{args.model}/",
    tensor_parallel_size=32,
    max_num_seqs=args.max_num_seqs,

    max_model_len=args.max_model_len,
    max_num_batched_tokens=args.chunk_size,
    enable_chunked_prefill=True,
    shard_over_sequence=args.sos,
    duplicate_q_weight_sos=args.sos,

    block_size=args.block_size,
    # gpu_memory_utilization=0.05,
    num_gpu_blocks_override=args.num_blocks,
)

def test():
    # Sample prompts.
    vocab_size = 32000
    total_num_seqs = 8
    min_context_len = 2
    max_context_len = 256
    prompts = torch.randint(0, vocab_size, (total_num_seqs, max_context_len)).numpy().tolist()
    prompt_lens = torch.randint(min_context_len, max_context_len, (total_num_seqs,)).numpy().tolist()

    prompt_tokens = [prompts[i][:prompt_lens[i]] for i in range(total_num_seqs)]
    # Create a sampling params object.
    sampling_params = SamplingParams(max_tokens=10, top_k=1)

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(sampling_params=sampling_params, prompt_token_ids=prompt_tokens)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    tokenizer = llm.get_tokenizer()
    for i, prompt in enumerate(prompt_tokens):
        num_input_tokens = len(prompt)
        print(f"prompt {i}, num_input_tokens: {num_input_tokens}")


def benchmark():
    max_context_len = args.max_model_len

    seed = max_context_len
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # sample requests
    vocab_size = 32000
    total_num_seqs = 1000
    max_context_len = max_context_len
    min_context_len = max_context_len // 2
    min_generation_len = 1
    max_generation_len = 128 #max_context_len // 4
    prompts = torch.randint(0, vocab_size, (total_num_seqs, max_context_len)).numpy().tolist()
    seq_lens = torch.randint(min_context_len, max_context_len, (total_num_seqs,)).numpy().tolist()
    generation_lens = torch.randint(min_generation_len, max_generation_len, size=(total_num_seqs,)).numpy().tolist()
    prompt_lens = [seq_len - gen_len for seq_len, gen_len in zip(seq_lens, generation_lens)]

    prompt_tokens = [prompts[i][:prompt_lens[i]] for i in range(total_num_seqs)]
    # Create a sampling params object.
    sampling_params = [SamplingParams(max_tokens=generation_lens[i], top_k=1) for i in range(total_num_seqs)]

    start = time.perf_counter()
    outputs = llm.generate(sampling_params=sampling_params, prompt_token_ids=prompt_tokens)
    end = time.perf_counter()
    
    ## result
    elapsed_sec = end - start
    print(f"Request Throughput: {total_num_seqs / elapsed_sec:.2f} req/s")
    print(f"Token Throughput: {sum(seq_lens) / elapsed_sec:.2f} tokens/s")


if __name__ == "__main__":
    if args.benchmark:
        benchmark()
    else:
        test()
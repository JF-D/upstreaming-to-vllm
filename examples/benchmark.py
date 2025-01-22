import os

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
parser.add_argument("--block-size", type=int, default=128)
parser.add_argument("--tp", type=int, default=8)
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--benchmark-full-prefill", action="store_true")
parser.add_argument("--benchmark-append-prefill", action="store_true")
parser.add_argument("--benchmark-decode", action="store_true")
parser.add_argument("--profile", action="store_true")
args = parser.parse_args()

# Create an LLM.
llm = LLM(
    model=f"/home/ubuntu/LLMCKPTs/{args.model}/",
    tensor_parallel_size=args.tp,
    max_num_seqs=args.max_num_seqs,

    max_model_len=args.max_model_len * 2 if args.benchmark_decode else args.max_model_len,
    max_num_batched_tokens=args.chunk_size,
    enable_chunked_prefill=True,

    block_size=args.block_size,
    gpu_memory_utilization=0.86 if args.max_model_len >= 16384 else 0.9,
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


def profile():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    chunk_size = 1024
    seq_len = 4096 + chunk_size
    max_num_seqs = args.max_num_seqs

    vocab_size = 32000
    prompts = torch.randint(0, vocab_size, (max_num_seqs, seq_len)).numpy().tolist()
    generation_lens = [1] * max_num_seqs
    prompt_lens = [seq_len] * max_num_seqs

    prompt_tokens = [prompts[i][:prompt_lens[i]] for i in range(max_num_seqs)]
    # Create a sampling params object.
    sampling_params = [SamplingParams(max_tokens=generation_lens[i], top_k=1) for i in range(max_num_seqs)]

    outputs = llm.generate(sampling_params=sampling_params, prompt_token_ids=prompt_tokens)


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
    max_generation_len = 128
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


def run_prefill(full_prefill, append_prefill, decode_only):
    max_context_len = args.max_model_len

    seed = max_context_len
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # sample requests
    vocab_size = 32000
    total_num_seqs = 150
    if full_prefill:
        min_context_len = max_context_len
        max_context_len = max_context_len + 1
    elif append_prefill:
        min_context_len = args.chunk_size + max_context_len
        max_context_len = min_context_len + 1
        os.environ["APPEND_CONTEXT_LENGTH"] = str(args.max_model_len)
    else:
        # decode
        assert decode_only
        total_num_seqs = 100 * args.chunk_size
        min_context_len = 1
        max_context_len = min_context_len + 1
        os.environ["DECODE_CONTEXT_LENGTH"] = str(args.max_model_len)
    min_generation_len = 1
    max_generation_len = min_generation_len + 1

    prompts = torch.randint(0, vocab_size, (total_num_seqs, max_context_len)).numpy().tolist()
    seq_lens = torch.randint(min_context_len, max_context_len, (total_num_seqs,)).numpy().tolist()
    generation_lens = torch.randint(min_generation_len, max_generation_len, size=(total_num_seqs,)).numpy().tolist()
    prompt_lens = [seq_len for seq_len, gen_len in zip(seq_lens, generation_lens)]
    # prompt_lens = [seq_len - gen_len for seq_len, gen_len in zip(seq_lens, generation_lens)]

    prompt_tokens = [prompts[i][:prompt_lens[i]] for i in range(total_num_seqs)]
    # Create a sampling params object.
    sampling_params = [SamplingParams(max_tokens=generation_lens[i], top_k=1, temperature=0) for i in range(total_num_seqs)]

    start = time.perf_counter()
    outputs = llm.generate(sampling_params=sampling_params, prompt_token_ids=prompt_tokens)
    end = time.perf_counter()

    ## result
    elapsed_sec = end - start
    print(f"Request Throughput: {total_num_seqs / elapsed_sec:.2f} req/s")
    print(f"Token Throughput: {sum(seq_lens) / elapsed_sec:.2f} tokens/s")


def run_decode(decode=True):
    max_context_len = args.max_model_len

    seed = max_context_len
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # sample requests
    vocab_size = 32000
    total_num_seqs = 150

    if decode:
        min_context_len = 1
        max_context_len = min_context_len + 1
        os.environ["APPEND_CONTEXT_LENGTH"] = str(args.max_model_len)
    min_generation_len = 1
    max_generation_len = min_generation_len + 1

    prompts = torch.randint(0, vocab_size, (total_num_seqs, max_context_len)).numpy().tolist()
    seq_lens = torch.randint(min_context_len, max_context_len, (total_num_seqs,)).numpy().tolist()
    generation_lens = torch.randint(min_generation_len, max_generation_len, size=(total_num_seqs,)).numpy().tolist()
    prompt_lens = [seq_len for seq_len, gen_len in zip(seq_lens, generation_lens)]
    # prompt_lens = [seq_len - gen_len for seq_len, gen_len in zip(seq_lens, generation_lens)]

    prompt_tokens = [prompts[i][:prompt_lens[i]] for i in range(total_num_seqs)]
    # Create a sampling params object.
    sampling_params = [SamplingParams(max_tokens=generation_lens[i], top_k=1, temperature=0) for i in range(total_num_seqs)]

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
    elif args.benchmark_full_prefill or args.benchmark_append_prefill or args.benchmark_decode:
        run_prefill(args.benchmark_full_prefill, args.benchmark_append_prefill, args.benchmark_decode)
    elif args.profile:
        profile()
    else:
        test()

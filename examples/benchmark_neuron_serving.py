import os
os.environ['NEURONX_DUMP_TO'] = os.path.join(os.getcwd(),"_compile_cache_2")
os.environ["NEURON_CC_FLAGS"]= " -O3 --internal-hlo2tensorizer-options=--verify-hlo --internal-enable-dge-levels=vector_dynamic_offsets --disable-internal-io-dge --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' "
# os.environ["NEURON_CC_FLAGS"]= " -O1 --internal-hlo2tensorizer-options=--verify-hlo --internal-enable-dge-levels=vector_dynamic_offsets "
os.environ["NEURON_RT_DBG_EMBEDDING_UPDATE_BOUND_CHECK"] = "0"
os.environ["NEURON_RT_DBG_INDIRECT_MEMCPY_BOUND_CHECK"] = "0"
import argparse
import torch
import json
import time
import numpy as np
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer


def get_LooGLE_settings():
    task2maxlen = {
        "shortdep_qa": 300,
        "longdep_qa": 500,
        "longdep_summarization":500, 
        "shortdep_cloze": 50
    }

    task2prompt = {
        "shortdep_qa": "Please answer the question based on the long texts below. \n{input}\nQuestion: {Q}\nAnswer: ",
        "longdep_qa": "Please answer the question based on the long texts below. \n{input}\nQuestion: {Q}\nAnswer: ",
        "longdep_summarization": "Please generate a summary of the below paper. \n{input}\n Summarization: ",
        "shortdep_cloze": "Please fill in the clozes based on the given long texts below. Each of the placeholder '<mask-n>' in the question could be an entity of Person, Location or Organiocation. The same masks represent the same entity. Output a json format answer, for example: {{'<mask-0>': 'Bob', '<mask-1>': 'Gorrosion Magazine','<mask-2>': 'Bethel Horizon'}}\n{input}\n Question: {Q} What are the masked entities? \nAnswer:"
    }
    return task2prompt, task2maxlen


def get_request(model, data_instance, tokenizer, max_length, max_gen, prompt_format):
    requests = []
    raw_inputs = data_instance['input']
    if data_instance['qa_pairs'] == 'none':
        json_obj = {'input': raw_inputs}
        prompt = prompt_format.format(**json_obj)
        output = data_instance["output"]
        
        tokenized_prompt = tokenizer.encode(prompt)
        tokenized_output = tokenizer.encode(output)
        # if len(tokenized_prompt) > max_length:
        #     half = int(max_length/2)
        #     prompt = tokenizer.decode(tokenized_prompt[:half]) + tokenizer.decode(tokenized_prompt[-half:])
        if len(tokenized_prompt) + len(tokenized_output) < max_length:
            requests.append([prompt, len(tokenized_output)])

    else:
        for j in eval(data_instance['qa_pairs']):
            json_obj = {'Q':j['Q'], 'input': raw_inputs}
            prompt = prompt_format.format(**json_obj)
            output = j['A']
            if isinstance(output, dict):
                output = json.dumps(output)

            tokenized_prompt = tokenizer.encode(prompt)
            tokenized_output = tokenizer.encode(output)
            # if len(tokenized_prompt) > max_length:
            #     half = int(max_length/2)
            #     prompt = tokenizer.decode(tokenized_prompt[:half])+tokenizer.decode(tokenized_prompt[-half:])
            if len(tokenized_prompt) + len(tokenized_output) < max_length:
                requests.append([prompt, len(tokenized_output)])
    return requests


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Meta-Llama-3-70B")
parser.add_argument("--max-num-seqs", type=int, default=8)
parser.add_argument("--max-model-len", type=int, default=1024)
parser.add_argument("--chunk-size", type=int, default=128)
parser.add_argument("--sos", action="store_true")
parser.add_argument("--duplicate_q", action="store_true")
parser.add_argument("--no-flash-paged-attention", action="store_true")
parser.add_argument("--mlp-duplicate-degree", type=int, default=1)
parser.add_argument("--block-size", type=int, default=128)
parser.add_argument("--num-blocks", type=int, default=512)
parser.add_argument("--layout-opt", action="store_true")
args = parser.parse_args()

if args.layout_opt:
    os.environ["NEURON_LAYOUT_OPT"] = "1"
os.environ["NEURON_MLP_DUPLICATE_DEGREE"] = str(args.mlp_duplicate_degree)
if not args.no_flash_paged_attention:
    os.environ["NEURON_FLASH_PA"] = "1"
os.environ["ENABLE_BUCKETING"] = "1"

# bucketing dimension for the sequence length of cache: n_active_blocks = buckets[i] * max_num_seqs // block_size
min_token_bucket_size = args.block_size // args.max_num_seqs
os.environ["NEURON_TOKEN_GEN_BUCKETS"] = ",".join(map(str, [2**i * min_token_bucket_size for i in range(30) if 2**i <= args.max_model_len // min_token_bucket_size // args.max_num_seqs]))
# os.environ["NEURON_TOKEN_GEN_BUCKETS"] = ",".join(map(str, [args.max_model_len // args.max_num_seqs]))
# bucketing dimension for max_num_batched_token or chunk size
# min_chunk_size: 128, max_chunk_size: args.chunk_size
# os.environ["NEURON_CONTEXT_LENGTH_BUCKETS"] = ",".join(map(str, [2**i for i in range(12) if 128 <= 2**i <= args.chunk_size]))


# Create an LLM.
llm = LLM(
    model=f"/home/ubuntu/LLMCKPTs/{args.model}/",
    tensor_parallel_size=32,
    max_num_seqs=args.max_num_seqs,

    max_model_len=args.max_model_len,
    max_num_batched_tokens=args.chunk_size,
    enable_chunked_prefill=True,
    shard_over_sequence=args.sos,
    duplicate_q_weight_sos=args.duplicate_q,

    block_size=args.block_size,
    # gpu_memory_utilization=0.05,
    num_gpu_blocks_override=args.num_blocks,
)


tokenizer = AutoTokenizer.from_pretrained(f"/home/ubuntu/LLMCKPTs/{args.model}/")
task2prompt, task2maxlen = get_LooGLE_settings()
datasets = ["shortdep_qa", "shortdep_cloze", "longdep_qa", "longdep_summarization"]
print(f"Start preparing requests....")

# check cache
CACHE_FILE = "./.loogle_dataset_cache.json"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as fp:
        all_requests = json.load(fp)
else:
    all_requests = []
    for testset in datasets:
        args.task = testset
        data = load_dataset('bigainlco/LooGLE', testset, split='test')
        prompt_format = task2prompt[args.task]
        max_gen = task2maxlen[args.task]
        for i in data:
            requests = get_request(llm, i, tokenizer, args.max_model_len, max_gen, prompt_format)
            all_requests.extend(requests)
        print(f"Finish {testset}")
    with open(CACHE_FILE, "w") as fp:
        json.dump(all_requests, fp)


np.random.seed(0)
num_requests = 1000
sample_ids = np.random.choice(len(all_requests), size=num_requests, replace=False)
for i in sample_ids:
    prompt, max_gen = all_requests[i]
    sampling_params = SamplingParams(max_tokens=max_gen, top_k=1, ignore_eos=True)
    llm._add_request(prompt, sampling_params)

print(f"--> begin to run generate")
start = time.perf_counter()
outputs = llm._run_engine(use_tqdm=True)
end = time.perf_counter()
print(f"Time: {end-start:.2f}s")
print(f"Throughput: {num_requests/(end-start):.2f} req/s")



# import torch
# total_num_seqs = 100
# prompts = torch.randint(0, 32000, (total_num_seqs, 2048)).numpy().tolist()
# seq_lens = torch.randint(512, 2048, (total_num_seqs,)).numpy().tolist()
# generation_lens = torch.randint(10, 50, size=(total_num_seqs,)).numpy().tolist()
# prompt_lens = [seq_len - gen_len for seq_len, gen_len in zip(seq_lens, generation_lens)]
# prompt_tokens = [prompts[i][:prompt_lens[i]] for i in range(total_num_seqs)]
# sampling_params = [SamplingParams(max_tokens=generation_lens[i], top_k=1) for i in range(total_num_seqs)]
# outputs = llm.generate(sampling_params=sampling_params, prompt_token_ids=prompt_tokens)
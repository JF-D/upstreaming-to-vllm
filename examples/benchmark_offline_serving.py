import os
import argparse
import torch
import json
import time
import numpy as np
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Meta-Llama-3-70B")
parser.add_argument("--max-num-seqs", type=int, default=8)
parser.add_argument("--max-model-len", type=int, default=1024)
parser.add_argument("--chunk-size", type=int, default=128)
parser.add_argument("--block-size", type=int, default=128)
parser.add_argument("--tp", type=int, default=8)
args = parser.parse_args()


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


llm = LLM(
    model=f"/home/ubuntu/LLMCKPTs/{args.model}/",
    tensor_parallel_size=args.tp,
    max_num_seqs=args.max_num_seqs,

    max_model_len=args.max_model_len,
    max_num_batched_tokens=args.chunk_size,
    enable_chunked_prefill=True,

    block_size=args.block_size,
    gpu_memory_utilization=0.86 if args.max_model_len >= 16384 else 0.9,
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


start = time.perf_counter()
outputs = llm._run_engine(use_tqdm=True)
end = time.perf_counter()
print(f"Time: {end-start:.2f}s")
print(f"Throughput: {num_requests/(end-start):.2f} req/s")

import random
import numpy as np
import torch
import bisect
import json
from transformers import AutoTokenizer


def main():
    # sample requests
    vocab_size = 32000
    total_num_seqs = 1000
    max_context_len = 4096
    min_context_len = max_context_len // 2
    min_generation_len = 1
    max_generation_len = 128 #max_context_len // 4
    prompts = torch.randint(0, vocab_size, (total_num_seqs, max_context_len)).numpy().tolist()
    seq_lens = torch.randint(min_context_len, max_context_len, (total_num_seqs,)).numpy().tolist()
    generation_lens = torch.randint(min_generation_len, max_generation_len, size=(total_num_seqs,)).numpy().tolist()
    prompt_lens = [seq_len - gen_len for seq_len, gen_len in zip(seq_lens, generation_lens)]

    # >>> LooGLE dataset
    tokenizer = AutoTokenizer.from_pretrained(f"/home/ubuntu/LLMCKPTs/Meta-Llama-3-8B/")
    CACHE_FILE = "./.loogle_dataset_cache.json"
    with open(CACHE_FILE, "r") as fp:
        all_requests = json.load(fp)
    np.random.seed(0)
    num_requests = 1000
    sample_ids = np.random.choice(len(all_requests), size=num_requests, replace=False)
    prompt_lens, generation_lens = [], []
    for i in sample_ids:
        prompt, max_gen = all_requests[i]
        prompt_tokens = tokenizer.encode(prompt)
        prompt_lens.append(len(prompt_tokens))
        generation_lens.append(max_gen)

    chunk_size = 2048
    max_num_seqs = 16
    block_size = 256 // 4
    unfinished = [(p, g, 0) for (p, g) in zip(prompt_lens, generation_lens)]
    finished = []
    buckets = {}
    waste_track = []
    batch_sizes = {}
    bs_to_max_blocks = {}
    available_buckets = [(2**i) for i in range(30)]
    # available_buckets.append(393216)
    available_buckets = sorted(available_buckets)
    while len(unfinished) > 0:
        batched_token = 0
        schedule_req_idx = 0
        scheduled = []
        while batched_token < chunk_size and len(scheduled) < max_num_seqs:
            req = unfinished[schedule_req_idx]
            if req[2] < req[0]: # prefill phase
                chk_sz = min(req[0] - req[2], chunk_size - batched_token)
            else: # decode phase
                chk_sz = 1
            scheduled.append((chk_sz, req[2]))
            unfinished[schedule_req_idx] = (req[0], req[1], req[2] + chk_sz)
            batched_token += chk_sz
        # pad each request to align with block_size
        bucket_size = sum([(sched[1] + block_size - 1) // block_size * block_size for sched in scheduled])
        unfinished = [req for req in unfinished if req[2] < req[0] + req[1]]
        # # pad to 2048
        # bucket_size = max(2048, (bucket_size + 2048 - 1) // 2048 * 2048)
        index = bisect.bisect_left(available_buckets, bucket_size, hi=len(available_buckets) - 1)
        bucket_size = available_buckets[index]
        if len(scheduled) not in batch_sizes:
            batch_sizes[len(scheduled)] = 0
            bs_to_max_blocks[len(scheduled)] = 0
        batch_sizes[len(scheduled)] += 1
        bs_to_max_blocks[len(scheduled)] = max(bs_to_max_blocks[len(scheduled)], bucket_size)

        if bucket_size not in buckets:
            buckets[bucket_size] = 0
        buckets[bucket_size] += 1

        waste_track.append((sum([sched[1] for sched in scheduled]), bucket_size))

    buckets = sorted(buckets.items(), key=lambda x: x[1])
    util = sum([w[0] for w in waste_track]) / sum([w[1] for w in waste_track])
    print(len(buckets), buckets)
    print(f"Utilization: {util:.3f}")
    print(f"Batch size: {batch_sizes}")
    print(f"Batch size to max bucket: {bs_to_max_blocks}")


if __name__ == "__main__":
    main()
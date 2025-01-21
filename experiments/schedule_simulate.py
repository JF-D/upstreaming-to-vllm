import random
import numpy as np
import torch
import bisect


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

    chunk_size = 128
    max_num_seqs = 16
    block_size = 128 // 4
    unfinished = [(p, g, 0) for (p, g) in zip(prompt_lens, generation_lens)]
    finished = []
    buckets = {}
    waste_track = []
    available_buckets = [(2**i) * 2048 for i in range(12)]
    available_buckets.append(393216)
    available_buckets = sorted(available_buckets)
    while len(unfinished) > 0:
        batched_token = 0
        schedule_req_idx = 0
        scheduled = []
        while batched_token < chunk_size:
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

        if bucket_size not in buckets:
            buckets[bucket_size] = 0
        buckets[bucket_size] += 1

        waste_track.append((sum([sched[1] for sched in scheduled]), bucket_size))
    
    buckets = sorted(buckets.items(), key=lambda x: x[1])
    util = sum([w[0] for w in waste_track]) / sum([w[1] for w in waste_track])
    print(len(buckets), buckets)
    print(f"Utilization: {util:.3f}")


if __name__ == "__main__":
    main()
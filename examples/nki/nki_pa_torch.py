import torch
import os
import torch_neuronx
import numpy as np
import math
import random
from dataclasses import dataclass
os.environ["XLA_FLAGS"] = " --xla_cpu_enable_fast_math=false "

import torch_xla
from torch_xla.core import xla_model as xm
device = xm.xla_device()
import neuronxcc.nki as nki
from torch_neuronx import nki_jit
from neuronxcc.nki import benchmark
from neuronxcc.nki import baremetal
from neuronxcc.nki.kernels import flash_fwd
import neuronxcc.nki.language as nl
from transformers_neuronx.layers.flash_attention import flash_paged_attention, FlashConfig
os.environ['NEURONX_DUMP_TO'] = os.path.join(os.getcwd(),"_compile_cache_nki")
os.environ["NEURON_COMPILE_CACHE_URL"] = os.path.join(os.getcwd(),"_compile_cache_nki")
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["NEURON_CC_FLAGS"]= " -O3 --internal-hlo2tensorizer-options=--verify-hlo --internal-enable-dge-levels=vector_dynamic_offsets --disable-internal-io-dge --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' "
# os.environ["NEURON_CC_FLAGS"]= " --disable-dge "
os.environ["XLA_FLAGS"] = " --xla_cpu_enable_fast_math=false "


def softmax(x: np.ndarray, dim: int, zero_max_mode=False,
            mixed_precision=False, return_max_reduce=False):
    max_value = np.amax(x, axis=dim, keepdims=True)
    max_value = np.maximum(0, max_value) if zero_max_mode else max_value
    exp = np.exp(x - max_value, dtype=np.float32)
    if mixed_precision:
        reduce = np.add.reduce(exp.astype(np.float32), axis=dim, keepdims=True).astype(x.dtype)
    else:
        reduce = np.add.reduce(exp, axis=dim, keepdims=True)
    if return_max_reduce:
        return exp / reduce, -max_value, np.reciprocal(reduce)
    return exp / reduce


def cpu_attention_forward(q, k, v, scale=None, mask=None, mixed_precision=True):
    """
    NumPy implementation of attention computation.

    Args:
        q: [bs, nheads, d, seqlen_q]
        k: [bs, kv_heads, d, seqlen_k]
        v: [bs, kv_heads, d, seqlen_k]
    """
    def mixed_precision_matmul(a, b, output_dtype=None):
        output_dtype = output_dtype if output_dtype else a.dtype
        a, b = a.astype(np.float32), b.astype(np.float32)
        c = np.matmul(a, b, dtype=output_dtype)
        return c

    _, _, d, _ = q.shape

    # Compute golden output
    softmax_scale = scale if scale else 1.0 / (d ** 0.5)
    q_scaled = q * softmax_scale
    nheads = q.shape[1]
    kv_heads = k.shape[1]
    if nheads > kv_heads:
        k = np.repeat(k, nheads//kv_heads, axis=1)
        v = np.repeat(v, nheads//kv_heads, axis=1)
    # raw_score: [bs, nheads, seqlen_q, seqlen_k]
    raw_score = mixed_precision_matmul(q_scaled.transpose(0, 1, 3, 2), k, output_dtype=np.float32)
    if mask is not None:
        mask = mask.astype(np.int16) # make numerical promotion to np.float32
        raw_score = raw_score * mask + (1 - mask) * np.float32(-3.e4)

    norm_score, cached_negative_max, cached_sum_reciprocal = \
        softmax(raw_score, dim=-1, mixed_precision=mixed_precision, return_max_reduce=True)

    # Transpose the result so it has the same layout as ours
    out_golden = mixed_precision_matmul(norm_score, v.transpose(0, 1, 3, 2), output_dtype=q.dtype).transpose(0, 1, 3, 2)

    return out_golden, cached_negative_max, cached_sum_reciprocal, norm_score


def numpy_paged_attention(
    query: np.ndarray,
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    block_tables: np.ndarray,
    context_lens: np.ndarray,
    prompt_lens: np.ndarray,
    seq_lens: np.ndarray,
    scale: float,
) -> np.ndarray:
    """
    NumPy implementation of paged attention with chunked query.

    Args:
        query: Query tensor of shape [num_tokens, num_query_heads, head_size]
        key_cache: Key cache tensor of shape [num_blocks, block_size, num_kv_heads, head_size]
        value_cache: Value cache tensor of shape [num_blocks, block_size, num_kv_heads, head_size]
        block_tables: Block tables tensor of shape [num_seqs, max_num_blocks_per_seq]
        context_lens: Context lengths tensor of shape [num_seqs]
        prompt_lens: Prompt lengths tensor of shape [num_seqs]
        seq_lens: Sequence lengths tensor of shape [num_seqs]
        scale: Scale factor for attention scores

    Returns:
        Output tensor of shape [num_tokens, num_query_heads, head_size]
    """
    num_tokens, num_query_heads, head_size = query.shape
    _, block_size, num_kv_heads, _ = value_cache.shape
    num_seqs = context_lens.shape[0]

    # Initialize output tensor
    output = np.zeros((num_tokens, num_query_heads, head_size), dtype=query.dtype)

    start_prompt_idx = 0
    for i in range(num_seqs):
        prompt_len = prompt_lens[i]
        context_len = context_lens[i]
        seq_len = seq_lens[i]
        kv_len = context_len

        # Get current sequence queries
        q = query[start_prompt_idx:start_prompt_idx + prompt_len]

        # Gather keys and values for the current sequence
        keys = []
        values = []
        for j in range(kv_len):
            block_number = block_tables[i, j // block_size]
            block_offset = j % block_size

            k = key_cache[block_number, block_offset]
            keys.append(k)

            v = value_cache[block_number, block_offset]
            values.append(v)

        keys = np.stack(keys, axis=0)      # [seq_len, num_kv_heads, head_size]
        values = np.stack(values, axis=0)   # [seq_len, num_kv_heads, head_size]

        q = q.transpose(1, 2, 0).reshape(1, num_query_heads, head_size, prompt_len)
        keys = keys.transpose(1, 2, 0).reshape(1, num_kv_heads, head_size, kv_len)
        values = values.transpose(1, 2, 0).reshape(1, num_kv_heads, head_size, kv_len)
        attn_mask = np.tril(np.ones((kv_len, kv_len), dtype=np.int32))
        attn_mask = attn_mask[-prompt_len:]

        o_golden, _, _, _  = \
            cpu_attention_forward(q, keys, values, scale=scale, mask=attn_mask, mixed_precision=True)
        o_golden = o_golden.transpose(0, 3, 1, 2) # (b, seq, h, d)
        output[start_prompt_idx:start_prompt_idx+prompt_len] = o_golden[0]

        start_prompt_idx += prompt_len

    return output


def chunked_paged_attention(
    query: np.ndarray,
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    active_block_tables: np.ndarray,
    context_lens: np.ndarray,
    prompt_lens: np.ndarray,
    seq_lens: np.ndarray,
    scale: float,
    mask: np.ndarray,
    return_attention_score: bool = False,
):
    num_tokens, num_query_heads, head_size = query.shape
    _, block_size, num_kv_heads, _ = value_cache.shape

    # Initialize output tensor
    output = np.zeros((num_tokens, num_query_heads, head_size), dtype=query.dtype)

    keys, values = [], []
    for block_id in active_block_tables:
        k = key_cache[block_id]
        v = value_cache[block_id]
        keys.append(k)
        values.append(v)

    keys = np.stack(keys, axis=0).reshape(-1, num_kv_heads, head_size)      # [seq_len, num_kv_heads, head_size]
    values = np.stack(values, axis=0).reshape(-1, num_kv_heads, head_size)   # [seq_len, num_kv_heads, head_size]

    query = query.transpose(1, 2, 0).reshape(1, num_query_heads, head_size, num_tokens)
    keys = keys.transpose(1, 2, 0).reshape(1, num_kv_heads, head_size, -1)
    values = values.transpose(1, 2, 0).reshape(1, num_kv_heads, head_size, -1)
    o_golden, _, _, attention_score  = \
            cpu_attention_forward(query, keys, values, scale=scale, mask=mask, mixed_precision=True)
    o_golden = o_golden.transpose(0, 3, 1, 2) # (b, seq, h, d)
    
    num_active_tokens = sum(prompt_lens)
    output[:num_active_tokens] = o_golden[0, :num_active_tokens]
    if return_attention_score:
        return output, attention_score
    return output


def test_flash_paged_attention(
      num_heads=(8, 1), head_size=128, num_active_tokens=128, num_seqs=8, 
      block_size=128, dtype=torch.float16, seed=0
):
    random.seed(seed)
    torch.random.manual_seed(seed)
    n_heads, n_kv_heads = num_heads
    num_queries_per_kv = n_heads // n_kv_heads
    scale = float(1.0 / (head_size**0.5))
    MAX_SEQ_LEN = 2048
    NUM_BLOCKS = 600

    context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_seqs)]
    context_lens[-1] = MAX_SEQ_LEN - 1

    prompt_lens = [random.randint(1, num_active_tokens // num_seqs) for _ in range(num_seqs)]
    prompt_lens[-1] = 1
    assert sum(prompt_lens) <= num_active_tokens

    position_ids = []
    for ctx_len, prompt_len in zip(context_lens, prompt_lens):
        position_ids.extend(list(range(ctx_len, ctx_len + prompt_len)))

    context_lens = np.array(context_lens, dtype=np.int32)
    prompt_lens = np.array(prompt_lens, dtype=np.int32)

    seq_lens = context_lens + prompt_lens
    max_seq_len = max(seq_lens)

    config = FlashConfig(training=False, seq_tile_size=2048)
    assert config.seq_tile_size % block_size == 0

    # Create the block tables.
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_randperm = np.random.permutation(NUM_BLOCKS - 1) + 1
    block_tables = block_randperm[:num_seqs * max_num_blocks_per_seq].reshape(num_seqs, -1)
    active_block_tables = []
    num_active_blocks = (seq_lens + block_size - 1) // block_size
    for seq_id in range(num_seqs):
        active_block_tables.extend(block_tables[seq_id, :num_active_blocks[seq_id]])
    # pad
    max_num_keys = ((num_seqs * max_seq_len + config.seq_tile_size - 1) // config.seq_tile_size) * config.seq_tile_size
    max_num_blocks = max_num_keys // block_size
    active_block_tables = active_block_tables + [0] * (max_num_blocks - len(active_block_tables))
    print("max_num_keys, blocks: ", max_num_keys, max_num_blocks)

    # build mask
    seq_id_x = np.array([len(prompt_lens) for _ in range(num_active_tokens)], dtype=np.int32)
    pos_id_x = np.array([0 for _ in range(num_active_tokens)], dtype=np.int32)
    cum_prompt_len = 0
    for seq_id, prompt_len in enumerate(prompt_lens):
        context_len = context_lens[seq_id]
        seq_id_x[cum_prompt_len:cum_prompt_len+prompt_len] = seq_id
        pos_id_x[cum_prompt_len:cum_prompt_len+prompt_len] = np.arange(context_len - prompt_len, context_len) #torch.arange(context_len, context_len + prompt_len)
        cum_prompt_len += prompt_len
    seq_id_y = np.array([len(prompt_lens) + 1 for _ in range(max_num_keys)], dtype=np.int32)
    pos_id_y = np.array([max_num_keys + 1 for _ in range(max_num_keys)], dtype=np.int32)
    cum_kv_len = 0
    for seq_id, context_len in enumerate(context_lens):
        context_len_pad = ((context_len + block_size - 1) // block_size) * block_size
        seq_id_y[cum_kv_len:cum_kv_len+context_len] = seq_id
        pos_id_y[cum_kv_len:cum_kv_len+context_len] = np.arange(0, context_len)
        cum_kv_len += context_len_pad
    seq_mask = (seq_id_x.reshape(num_active_tokens, 1) == seq_id_y.reshape(1, max_num_keys))
    pos_mask = (pos_id_x.reshape(num_active_tokens, 1) >= pos_id_y.reshape(1, max_num_keys))
    mask = seq_mask & pos_mask

    query = np.random.randn(1, n_heads, head_size, num_active_tokens)
    key = np.random.randn(1, n_kv_heads, head_size, num_active_tokens)
    value = np.random.randn(1, n_kv_heads, head_size, num_active_tokens)

    context_lens = torch.tensor(context_lens, dtype=torch.int)
    prompt_lens = torch.tensor(prompt_lens, dtype=torch.int)
    seq_lens = torch.tensor(seq_lens, dtype=torch.int)
    position_ids = torch.tensor(position_ids, dtype=torch.int).reshape(1, -1)

    mask = torch.tensor(mask, dtype=torch.int32).to(device=device)
    active_block_tables = torch.tensor(active_block_tables, dtype=torch.int).to(device=device)
    block_tables = torch.tensor(block_tables, dtype=torch.int)

    query = torch.tensor(query, dtype=dtype).to(device=device)
    key = torch.tensor(key, dtype=dtype).to(device=device)
    value = torch.tensor(value, dtype=dtype).to(device=device)
    key_cache = torch.empty((NUM_BLOCKS, block_size, n_kv_heads, head_size), dtype=dtype).to(device=device)
    # key_cache = torch.empty((NUM_BLOCKS, block_size, n_kv_heads, head_size), dtype=dtype).transpose(0, 3, 2, 1).to(device=device)
    value_cache = torch.empty((NUM_BLOCKS, block_size, n_kv_heads, head_size), dtype=dtype).to(device=device)
    key_cache.uniform_(-scale, scale)
    value_cache.uniform_(-scale, scale)

    assert num_active_tokens % 128 == 0
    print(context_lens)
    print(seq_lens)
    print(active_block_tables.size())

    o, lse = flash_paged_attention[1, 1](
        query, key, value, key_cache, value_cache,
        active_block_tables, mask,
        softmax_scale=scale, config=config
    )
    o = o.permute(0, 2, 1, 3)
    o = o.cpu().numpy()
    print(o)

    # o, _ = nki.simulate_kernel(flash_paged_attention[1, 1], 
    #                     query.cpu().numpy(), key.cpu().numpy(), value.cpu().numpy(), key_cache.cpu().numpy(), value_cache.cpu().numpy(),
    #                     active_block_tables.cpu().numpy(), mask.cpu().numpy(),
    #                     softmax_scale=scale, config=config)
    # o = o.transpose(0, 2, 1, 3)

    ref_output = numpy_paged_attention(
        query.permute(0, 3, 1, 2).squeeze(0).cpu().numpy(),
        key_cache.cpu().numpy(),
        value_cache.cpu().numpy(),
        block_tables.cpu().numpy(),
        context_lens.cpu().numpy(),
        prompt_lens.cpu().numpy(),
        seq_lens.cpu().numpy(),
        scale,
    )
    ref_output = ref_output.reshape((1, num_active_tokens, n_heads, head_size))

    # chunked_output, attention_score = chunked_paged_attention(
    #     query.permute(0, 3, 1, 2).squeeze(0).cpu().numpy(),
    #     key_cache.cpu().numpy(),
    #     value_cache.cpu().numpy(),
    #     active_block_tables.cpu().numpy(),
    #     context_lens.cpu().numpy(),
    #     prompt_lens.cpu().numpy(),
    #     seq_lens.cpu().numpy(),
    #     scale,
    #     mask.cpu().numpy().reshape(1, 1, num_active_tokens, max_num_keys),
    #     return_attention_score=True,
    # )
    # chunked_output = chunked_output.reshape((1, num_active_tokens, n_heads, head_size))
    # assert np.allclose(ref_output, chunked_output, atol=1.e-5)

    o[0, num_active_tokens:] = 0
    # import pdb; pdb.set_trace()
    assert np.allclose(ref_output, o, atol=1.e-2)


if __name__ == "__main__":
    # bs, n_heads, seq_q, head_size = 1, 4, 128, 128
    # n_kv_heads, seq_kv = 1, 16384
    # q = torch.rand((bs, n_heads, head_size, seq_q), dtype=torch.float16).to(device=device)
    # k = torch.rand((bs, n_kv_heads, head_size, seq_kv), dtype=torch.float16).to(device=device)
    # v = torch.rand((bs, n_kv_heads, seq_kv, head_size), dtype=torch.float16).to(device=device)
    # config = FlashConfig(training=False, should_transpose_v=False)
    # # print(os.environ)
    # o = flash_fwd[bs, n_kv_heads](q, k, v, None, config=config)
    # print(o)
    test_flash_paged_attention()

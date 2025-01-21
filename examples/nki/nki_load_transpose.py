import numpy as np
import neuronxcc.nki.language as nl
from neuronxcc.nki.language import par_dim
from neuronxcc.nki import baremetal


@baremetal
def large_load_transpose_kernel(key_cache, value_cache, block_tables):
    """
    key_cache: [num_blocks, block_size, n_kv_heads, head_size]
    value_cache: [num_blocks, block_size, n_kv_heads, head_size]
    block_tables: [num_active_blocks]
    k_output: [head_size, num_active_blocks * block_size]
    v_output: [num_active_blocks * block_size, head_size]
    """
    num_blocks, block_size, n_kv_heads, head_size = key_cache.shape
    num_active_blocks, = block_tables.shape

    head_id = nl.program_id(axis=0)

    B_P_SIZE = 128
    B_D_SIZE = head_size
    LARGE_TILE_SZ = 2048
    kernel_dtype = key_cache.dtype

    num_large_k_tile = num_active_blocks * block_size // LARGE_TILE_SZ
    num_blocks_per_large_tile = LARGE_TILE_SZ // block_size

    i_q_p = nl.arange(B_P_SIZE)[:,None]
    i_0_f = nl.arange(1)[None, :]

    k_output = nl.ndarray((head_size, num_active_blocks * block_size), dtype=key_cache.dtype, buffer=nl.shared_hbm)
    v_output = nl.ndarray((num_active_blocks * block_size, head_size), dtype=value_cache.dtype, buffer=nl.shared_hbm)

    block_tables_sbuf = nl.full((par_dim(B_P_SIZE), num_large_k_tile), 0, dtype=np.int32, buffer=nl.sbuf)
    for j in range(num_large_k_tile):
        i_p = nl.arange(num_blocks_per_large_tile)[:, None]
        block_tables_sbuf[i_p, j + i_0_f] = nl.load(
           block_tables[j*num_blocks_per_large_tile + i_p], dtype=np.int32
        )

    j = 0
    cur_k_tile = nl.ndarray((par_dim(B_D_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
    cur_v_tile = nl.ndarray((LARGE_TILE_SZ//B_P_SIZE, par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype)
    for k_i in nl.affine_range(num_blocks_per_large_tile):
        load_p = nl.arange(B_D_SIZE)[:, None]
        load_f = nl.arange(block_size)[None, :]

        loaded = nl.load(key_cache[
            block_tables_sbuf[k_i, j],
            nl.arange(block_size)[:, None], head_id, nl.arange(B_D_SIZE)[None, :]
        ])
        cur_k_tile[load_p, k_i * block_size + load_f] = nl.transpose(loaded)

    # Load and process value cache for the first large tile
    blocks_per_partition = B_P_SIZE // block_size  # This is a compile-time constant
    
    # Process each partition of size B_P_SIZE
    for partition_idx in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
        # Process each block within the partition
        for block_in_partition in nl.affine_range(blocks_per_partition):
            v_i = partition_idx * blocks_per_partition + block_in_partition
            
            load_p = nl.arange(B_D_SIZE)[None, :]
            load_f = nl.arange(block_size)[:, None]

            # Load values from value cache based on block table
            loaded_v = nl.load(value_cache[
                block_tables_sbuf[v_i, 0],
                nl.arange(block_size)[:, None], head_id, nl.arange(B_D_SIZE)[None, :]
            ])

            # Place block in appropriate position within partition
            cur_v_tile[partition_idx, block_in_partition * block_size + load_f, load_p] = loaded_v

    # Store the processed key cache for the first large tile
    i_p = nl.arange(B_D_SIZE)[:, None]
    i_t = nl.arange(LARGE_TILE_SZ)[None, :]
    nl.store(k_output[i_p, i_t], cur_k_tile[i_p, i_t])

    # Store the processed value cache for the first large tile
    i_p = nl.arange(B_D_SIZE)[None, :]
    i_t = nl.arange(B_P_SIZE)[:, None]
    for v_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
        nl.store(v_output[v_i * B_P_SIZE + i_t, i_p], cur_v_tile[v_i, i_t, i_p])

    # Process subsequent large tiles
    for j in nl.sequential_range(1, num_large_k_tile):
        cur_k_tile = nl.ndarray((par_dim(B_D_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
        cur_v_tile = nl.ndarray((LARGE_TILE_SZ//B_P_SIZE, par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype)
        
        # Process key cache for current large tile
        for k_i in nl.affine_range(num_blocks_per_large_tile):
            load_p = nl.arange(B_D_SIZE)[:, None]
            load_f = nl.arange(block_size)[None, :]
            
            loaded = nl.load(key_cache[
                block_tables_sbuf[k_i, j],
                nl.arange(block_size)[:, None], head_id, nl.arange(B_D_SIZE)[None, :]
            ])
            cur_k_tile[load_p, k_i * block_size + load_f] = nl.transpose(loaded)

        # Store processed key cache for current large tile
        i_p = nl.arange(B_D_SIZE)[:, None]
        i_t = nl.arange(LARGE_TILE_SZ)[None, :]
        nl.store(k_output[i_p, j * LARGE_TILE_SZ + i_t], cur_k_tile[i_p, i_t])

        # Process value cache for current large tile
        blocks_per_partition = B_P_SIZE // block_size  # This is a compile-time constant
        
        # Process each partition of size B_P_SIZE
        for partition_idx in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
            # Process each block within the partition
            for block_in_partition in nl.affine_range(blocks_per_partition):
                v_i = partition_idx * blocks_per_partition + block_in_partition
                
                load_p = nl.arange(B_D_SIZE)[None, :]
                load_f = nl.arange(block_size)[:, None]
                
                loaded_v = nl.load(value_cache[
                    block_tables_sbuf[v_i, j],
                    nl.arange(block_size)[:, None], head_id, nl.arange(B_D_SIZE)[None, :]
                ])
                
                # Place block in appropriate position within partition
                cur_v_tile[partition_idx, block_in_partition * block_size + load_f, load_p] = loaded_v

        # Store processed value cache for current large tile
        i_p = nl.arange(B_D_SIZE)[None, :]
        i_t = nl.arange(B_P_SIZE)[:, None]
        for v_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
            nl.store(v_output[j * LARGE_TILE_SZ + v_i * B_P_SIZE + i_t, i_p], cur_v_tile[v_i, i_t, i_p])
    
    return k_output, v_output


def test_load_transpose():
    np.random.seed(0)
    num_blocks, block_size, n_kv_heads, head_size = 300, 64, 1, 128
    num_active_blocks = 128
    max_num_keys = 8192

    key_cache = np.random.rand(num_blocks, block_size, n_kv_heads, head_size).astype(np.float16)
    value_cache = np.random.rand(num_blocks, block_size, n_kv_heads, head_size).astype(np.float16)
    active_block_tables = np.random.randint(1, num_blocks, size=(num_active_blocks, ), dtype=np.int32)
    k_output = np.zeros((head_size, max_num_keys), dtype=np.float16)
    v_output = np.zeros((max_num_keys, head_size), dtype=np.float16)

    k_output, v_output = large_load_transpose_kernel[1, 1](key_cache, value_cache, active_block_tables)

    expect_k_output = np.zeros((head_size, max_num_keys), dtype=np.float16)
    expect_v_output = np.zeros((max_num_keys, head_size), dtype=np.float16)
    for i, block_idx in enumerate(active_block_tables):
        expect_k_output[:, i*block_size:(i+1)*block_size] = np.transpose(key_cache[block_idx, :, 0, :])
        expect_v_output[i*block_size:(i+1)*block_size, :] = value_cache[block_idx, :, 0, :]
    assert np.allclose(expect_k_output, k_output)
    assert np.allclose(expect_v_output, v_output)


if __name__ == "__main__":
    test_load_transpose()
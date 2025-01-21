import os
from neuronxcc.nki import benchmark
from neuronxcc.nki.typing import tensor
import neuronxcc.nki.language as nl
import math
from transformers_neuronx.layers.flash_attention import flash_paged_attention
os.environ["XLA_FLAGS"] = " --xla_cpu_enable_fast_math=false "
os.environ["NEURON_CC_FLAGS"]= " -O3 --internal-hlo2tensorizer-options=--verify-hlo --internal-enable-dge-levels=vector_dynamic_offsets --disable-internal-io-dge --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' "
print(os.environ)

if __name__ == "__main__":
    flash_paged_attention[1, 1](
        tensor[[1, 8, 128, 128], nl.float16],
        tensor[[1, 1, 128, 128], nl.float16],
        tensor[[1, 1, 128, 128], nl.float16],
        tensor[[600, 128, 1, 128], nl.float16],
        tensor[[600, 128, 1, 128], nl.float16],
        tensor[[128], nl.int32],
        tensor[[128, 16384], nl.int32],
    )

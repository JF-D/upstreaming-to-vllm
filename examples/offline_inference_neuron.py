import os, torch
os.environ['NEURONX_DUMP_TO'] = os.path.join(os.getcwd(),"_compile_cache")
os.environ["NEURON_CC_FLAGS"]= " -O1 --internal-hlo2tensorizer-options=--verify-hlo --internal-enable-dge-levels=vector_dynamic_offsets "
os.environ["NEURON_RT_DBG_EMBEDDING_UPDATE_BOUND_CHECK"] = "0"
os.environ["NEURON_RT_DBG_INDIRECT_MEMCPY_BOUND_CHECK"] = "0"
# os.environ["HLO_SNAPSHOT_PATH"] = os.path.join(os.getcwd(), "_snapshots")

from vllm import LLM, SamplingParams
import logging
logging.basicConfig(level=logging.DEBUG)
# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "It is not the critic who counts; not the man who points out how the strong man stumbles, or where the doer of deeds could have done them better. The credit belongs to the man who is actually in the arena, whose face is marred by dust and sweat and blood; who strives valiantly; who errs, who comes short again and again, because there is no effort without error and shortcoming; but who does actually strive to do the deeds; who knows great enthusiasms, the great devotions; who spends himself in a worthy cause; who at the best knows",
    "The future of AI is",
    "Hello, my name can",
    "The president of the United States can",
    "The capital of France can",
    "The future of AI can",
    "The sun rises in the",
    "The quick brown fox",
    "The biggest city in India is",
]
# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=10, top_k=1)

# Create an LLM.
llm = LLM(
    # model="nickypro/tinyllama-15M",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # model="openlm-research/open_llama_3b",
    # model="/shared_3/chndkv/llama-models/Meta-Llama-3.1-8B-Instruct/",
    tensor_parallel_size=32,
    max_num_seqs=8,

    max_model_len=1024,
    max_num_batched_tokens=64,
    enable_chunked_prefill=True,

    # max_model_len=256,
    # max_num_batched_tokens=256,
    # enable_chunked_prefill=True,

    block_size=32,
    # gpu_memory_utilization=0.05,
    num_gpu_blocks_override=300,
)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
from transformers_neuronx import global_debugger
with global_debugger.debug_context():
    outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

tokenizer = llm.get_tokenizer()
for i, prompt in enumerate(prompts):
    input_ids = tokenizer.encode(prompt, return_tensors="pt") 
    num_input_tokens = len(input_ids[0])
    print(f"prompt {i}, num_input_tokens: {num_input_tokens}")

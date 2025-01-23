from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
    local_dir="/root/LLM-Software-Forge-Factory",
    filename="DeepSeek-R1-Distill-Qwen-7B-Q6_K_L.gguf",
    verbose=False
)
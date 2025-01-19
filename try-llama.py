from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="QuantFactory/Hermes-3-Llama-3.2-3B-GGUF",
    filename="Hermes-3-Llama-3.2-3B.Q2_K.gguf",
)

print(
        llm.create_chat_completion(
            messages = [
                {
                    "role": "user",
                    "content": "What is the capital of France?"
                }
            ]
        )
)
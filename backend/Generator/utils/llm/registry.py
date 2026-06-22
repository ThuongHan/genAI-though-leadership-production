from Generator.utils.llm.gpt import GPTLLM
from Generator.utils.llm.claude import ClaudeLLM

def get_llm(model_name: str):

    if "gpt" in model_name:
        return GPTLLM(model_name)

    if "claude" in model_name:
        return ClaudeLLM(model_name)

    raise ValueError(f"Unknown model: {model_name}")



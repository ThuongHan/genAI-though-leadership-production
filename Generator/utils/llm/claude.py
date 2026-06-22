from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
from Generator.utils.llm.base import BaseLLM
from pydantic import SecretStr

# cluade models: ["claude-sonnet-4-6", "claude-sonnet-4-6", "claude-opus-4-8"]

class ClaudeLLM(BaseLLM):

    def __init__(self, model: str = "claude-sonnet-4-6"):
        load_dotenv("Generator/secrets/claude_api_key.env", override=True)

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Missing ANTHROPIC_API_KEY")

        self.model = model

        self.llm = ChatAnthropic(
            api_key=SecretStr(api_key),
            model_name=self.model,
            timeout=None,
            stop=None
        )

    def invoke(self, prompt: str):
        return self.llm.invoke(prompt)
    
if __name__ == "__main__":
    query = "Are you present? (Yes/No)"

    claude_haiku = ClaudeLLM("claude-haiku-4-5")
    result = claude_haiku.invoke(query)
    print("Haiku: ", result.content)

    claude_sonnet = ClaudeLLM()
    result = claude_sonnet.invoke(query)
    print("Sonnet: ", result.content)

    claude_opus = ClaudeLLM("claude-opus-4-8")
    result = claude_haiku.invoke(query)
    print("Opus: ", result.content)

# python3 -m Generator.utils.llm.claude


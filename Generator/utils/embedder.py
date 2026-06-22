from dotenv import load_dotenv
from pathlib import Path
import os
from pydantic import SecretStr

from langchain_openai import OpenAIEmbeddings

# For few shot prompting, we will need to embed the current posts
# so that we can later extract post that are semantically similar
# to the topic - which is the output from the interpreter

load_dotenv("Generator/secrets/openai_api_key.env")

# "text-embedding-3-large" returns a 3072 dimensional embedding
class Embedder:
    def __init__(self, model_name="text-embedding-3-large") -> None:
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found")

        self.embeddings = OpenAIEmbeddings(
            api_key=SecretStr(openai_api_key),
            model=model_name,
        )
    def embed_text(self, text: str):
        return self.embeddings.embed_query(text)
    
if __name__ == "__main__":
    query = "football is great"
    embedder = Embedder()

    print(embedder.embed_text(query))

# python3 -m Generator.utils.embedder


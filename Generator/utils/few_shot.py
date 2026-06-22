import json
from typing import Any

import numpy as np
import pandas as pd

from Generator.utils.embedder import Embedder

# Few-shot retrieval system for LinkedIn posts.
# Combines rule-based filtering (length + tags)
# and semantic search (embedding similarity).

class FewShotPost:
    def __init__(self, file_path="Generator/data/LinkedIn_processed_data.json") -> None:
        self.df: pd.DataFrame = pd.DataFrame()
        self.unique_tags: set[str] = set()
        self.load_posts(file_path)

    def load_posts(self, file_path: str) -> None:
        with open(file_path, encoding="utf-8") as file:
            posts: list[dict[str, Any]] = json.load(file)

        self.df = pd.json_normalize(posts)

        # Categorize length
        self.df["length"] = self.df["line_count"].apply(self.categorize_length)

        # Extract unique tags
        all_tags: list[str] = self.df["tags"].apply(lambda x: x).sum()
        self.unique_tags = set(all_tags)

    @staticmethod
    def categorize_length(line_count: int) -> str:
        if line_count < 8:
            return "Short"
        elif 8 <= line_count <= 13:
            return "Medium"
        else:
            return "Long"

    def get_tags(self) -> set[str]:
        return self.unique_tags
    
    # Rule based filtering: length + tag
    def get_filtered_posts(self, length: str, tag: str):
        df_filtered = self.df[
            (self.df["length"]==length) &
            (self.df["tags"].apply(lambda tags: tag in tags))
        ] 
        return df_filtered.to_dict(orient="records")
    
    # Semantics-based filtering
    def get_similar_posts(self, query_embedding: list[float], top_k = 3) -> list[dict[str, Any]]:
        """query_embedding: embedded output from the Interpreter"""
        df: pd.DataFrame = self.df.copy()

        df["similarity"] = df["embedding"].apply(
            lambda post_embedding: 
            cosine_similarity(post_embedding, query_embedding)
        )

        top_posts: pd.DataFrame = df.sort_values("similarity", ascending=False).head(top_k)

        return top_posts.to_dict(orient="records")
    
def cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


# Test    
if __name__=="__main__":
    fs = FewShotPost()

    #1. Fake Interpreter output (structured event)
    interpreter_output = {
        "what_happened": "AI tools are transforming how developers build software",
        "why_global": "AI is reshaping global productivity",
        "why_nl": "Dutch startups are adopting AI rapidly",
        "why_kickstartai": "KickstartAI focuses on AI education and adoption",
        "stance": "AI should be adopted quickly in education",
        "arguments": [
            "Increases productivity",
            "Reduces development time",
            "Creates new opportunities"
        ]
    }

    # 2. Convert to text into a better structure as input to embedding
    text = f"""
    What happened: {interpreter_output['what_happened']}
    Why global: {interpreter_output['why_global']}
    Why NL: {interpreter_output['why_nl']}
    Why KickstartAI: {interpreter_output['why_kickstartai']}
    Stance: {interpreter_output['stance']}
    Arguments: {', '.join(interpreter_output['arguments'])}
    """

    # 3. Embed it
    embedder = Embedder()
    query_embedding = embedder.embed_text(text)

    # 4. Get similar posts
    posts = fs.get_similar_posts(query_embedding, top_k=3)

    # 5. Print nicely
    for i, post in enumerate(posts, 1):
        print(f"\n POST {i}")
        print(post["text"])
        print("Similarity:", post["similarity"])



# python3 -m src.utils.few_shot 




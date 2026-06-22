import json
import re
from typing import Any

import pandas as pd
from Generator.utils.embedder import Embedder

# This module processes raw LinkedIn export data into JSON.
# It extracts relevant fields for content generation and adds
# embeddings for few-shot prompting / semantic retrieval.

def doc_to_json(doc_file: str, json_file: str) -> None:
    df : pd.DataFrame = pd.read_excel(
        doc_file, 
        sheet_name="All posts", 
        skiprows=1
    )

    # rename columns (for easier usage)
    df = df.rename(columns={
        "Post title": "text",
        "Post link" : "url",
        "Post type": "post_type",
        "Created date": "created_date",
        "Impressions": "impressions",
        "Clicks": "clicks",
        "Likes": "likes",
        "Comments": "comments",
        "Reposts": "reposts",
        "Engagement rate": "engagement_rate"
    })
    
    df = df.fillna(0)
    df = df.drop_duplicates(subset=["text"])

    records = df.to_dict(orient="records")

    with open(json_file, "w") as file:
        json.dump(records, file, indent=2)
    
    print(f"Saved processed data to {json_file}")

def process_posts(json_file: str, processed_json_file: str) -> None:
    embedder: Embedder = Embedder()
    enriched_posts: list[dict[str, Any]] = []

    with open(json_file, encoding="utf-8") as file:
        posts: list[dict[str, Any]] = json.load(file)
        
        for i, post in enumerate(posts, start=1):
            # Keep relevant fields 
            text: str = post.get("text", "")
            clean_post: dict[str, Any] = {
                "id": i,
                "text": text,
                "post_type": post.get("post_type"),
                "created_date": post.get("created_date"),
                "line_count": text.count("\n") + 1,
                "tags": re.findall(r"#\w+", text),
                "embedding": embedder.embed_text(text)
            }
        
            enriched_posts.append(clean_post)
    
    with open(processed_json_file, encoding="utf-8", mode="w") as outfile:
        json.dump(enriched_posts, outfile, indent=4)

    print(f"Saved enriched dataset to {processed_json_file}")

if __name__ == "__main__":
    raw_file = "Generator/data/KickstartAI LinkedIn post year to date.xlsx"
    json_file = "Generator/data/LinkedIn_data.json"
    processed_file = "Generator/data/LinkedIn_processed_data.json"

    doc_to_json(raw_file, json_file)
    process_posts(json_file, processed_file)


# python3 -m Generator.utils.linkedIn_data_processing
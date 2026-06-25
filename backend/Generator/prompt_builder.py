from pathlib import Path

from backend.Generator.utils.few_shot import FewShotPost
from backend.Generator.utils.embedder import Embedder
from langchain_core.output_parsers import PydanticOutputParser
from backend.Generator.schemas.generator_schema import GeneratedPosts


class PromptBuilder:
    def __init__(self, config_path: str) -> None:
        with open(config_path, "r", encoding="utf-8") as file:
            self.brand_guidelines = file.read()

    def build(
        self,
        event: dict,
        format_instructions: str,
        use_few_shot: bool,
        few_shot_posts: list[dict] | None = None,
    ) -> str:

        # --- Few-shot block (optional) ---
        examples_block = ""
        if use_few_shot and few_shot_posts:
            examples_block = "\n\n".join(
                f"POST {i+1}:\n{post['text']}"
                for i, post in enumerate(few_shot_posts)
            )

        # --- Event block (mirrors interpreter JSON schema) ---
        arguments = "\n- ".join(event["arguments"])

        event_block = (
            f"* What happened:              {event['what_happened']}\n"
            f"* Global & NL relevance:       {event['why_relevance']}\n"
            f"* KickstartAI stance:          {event['stance']}\n"
            f"* Arguments supporting stance:\n- {arguments}\n"
            f"* Source:                      {event['source']}\n"
        )

        # --- Prompt assembly ---
        prompt = f"{self.brand_guidelines}\n\n---\n\n"
        if examples_block:
            prompt += f"""
## STYLE REFERENCE POSTS

The posts below are real LinkedIn posts published by KickstartAI, retrieved because their topic
is similar to the current one. Use them only as references for voice, tone, sentence rhythm,
structure, and formatting.

Do not reuse their factual content, claims, names, figures, dates, or any specific details. All
factual content of your posts must come from the structured interpretation provided below, not
from these reference posts. The examples show you HOW KickstartAI writes, not WHAT to write
about.
"""
            
            prompt += f"\n\n{examples_block}\n\n---\n"
        
        prompt += f"\n## EVENT INFORMATION\n{event_block}\n\n---\n"

        prompt += f"\n## OUTPUT INSTRUCTIONS\n{format_instructions}\n"

        prompt += "\nWrite three LinkedIn posts now.\n"

        return prompt


# Example few-shot (k=1) Generator prompt 

if __name__ == "__main__":
    # System components
    fs = FewShotPost()
    embedder = Embedder()
    builder = PromptBuilder(str(Path(__file__).resolve().parent / "config" / "zeroshot-prompt.md"))

    # 1. Example interpreter output (matches interpreter JSON schema)
    interpreter_output = {
        "what_happened": (
            "ING COO Risk Leon Dusée shared lessons on AI adoption in large organizations, "
            "explaining why many AI pilots stall and how organizations can scale AI successfully."
        ),
        "why_relevance": (
            "Organizations worldwide are struggling with moving AI from isolated experiments into "
            "real operational impact. Dutch organizations across banking, aviation, retail, education, "
            "and healthcare are trying to accelerate AI adoption while avoiding fragmented pilots."
        ),
        "stance": (
            "AI adoption should be treated as an organizational transformation challenge, "
            "not just a technology experiment. Focus, leadership, and investing in people "
            "are just as important as the AI systems themselves."
        ),
        "arguments": [
            "AI pilots often fail because organizations think about scaling and ownership too late",
            "Organizations need to focus on a limited number of high-impact use cases instead of experimenting everywhere",
            "AI requires employees and leaders to develop critical thinking and adaptability",
            "The biggest AI opportunities often require redesigning entire processes, not just optimizing individual tasks",
            "Cross-organizational collaboration helps accelerate AI adoption and societal impact in the Netherlands",
            "AI should not only improve productivity, but also contribute to areas like financial health, education, and healthcare",
        ],
        "source": "https://lnkd.in/eU8jqvT9",
    }

    # 2. Embed event for semantic few-shot retrieval
    query_text = (
        f"What happened: {interpreter_output['what_happened']} "
        f"Why relevance: {interpreter_output['why_relevance']} "
        f"Stance: {interpreter_output['stance']} "
        f"Arguments: {', '.join(interpreter_output['arguments'])}"
    )
    query_embedding = embedder.embed_text(query_text)

    # 3. Retrieve semantically similar historical posts
    few_shot_posts = fs.get_similar_posts(query_embedding, top_k=1)

    # 4. Build prompt with Pydantic output parser
    parser = PydanticOutputParser(pydantic_object=GeneratedPosts)
    format_instructions = parser.get_format_instructions()

    prompt = builder.build(
        event=interpreter_output,
        format_instructions=format_instructions,
        use_few_shot=True,
        few_shot_posts=few_shot_posts,
    )

    print(prompt)

# python3 -m Generator.prompt_builder

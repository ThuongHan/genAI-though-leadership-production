from Generator.prompt_builder import PromptBuilder
from Generator.utils.llm.registry import get_llm
from Generator.utils.few_shot import FewShotPost
from Generator.utils.embedder import Embedder

from langchain_core.output_parsers import PydanticOutputParser
from Generator.schemas.generator_schema import GeneratedPosts

from pathlib import Path
import json

# gpt models: ["gpt-5.1", "gpt-5", "gpt-4.1", "gpt-4o"]
# cluade models: ["claude-haiku-4-6", "claude-sonnet-4-6", "claude-opus-4-8"]

class PostGenerator:
    def __init__(self, model: str = "claude-sonnet-4-6", config_path: str = "Generator/config/post-reformulated-prompt.md"):
        self.llm = get_llm(model)

        self.prompt_builder = PromptBuilder(config_path)
        
        self.few_shot = FewShotPost()
        self.embedder = Embedder()

        self.parser = PydanticOutputParser(
            pydantic_object=GeneratedPosts
        )

    def generate(
            self,
            interpreter_output: dict,
            k_posts: int,
            use_few_shot: bool = True,
            save: bool = True,
    ) -> dict:

        # 1. Convert event to text
        text = f"""
    What happened: {interpreter_output['what_happened']}
    Why relevance: {interpreter_output['why_relevance']}
    Why KickstartAI: {interpreter_output['why_kickstartai']}
    Stance: {interpreter_output['stance']}
    Arguments: {', '.join(interpreter_output['arguments'])}
    """

        # 2. Embed query
        query_embedding = self.embedder.embed_text(text)

        # 3. Retrieve similar posts (few shot exmaples)
        few_shot_posts = []
        if use_few_shot and k_posts and k_posts > 0:
            few_shot_posts = self.few_shot.get_similar_posts(
                query_embedding,
                top_k=k_posts
            ) or []

        # 4. Build prompt
        prompt = self.prompt_builder.build(
            event=interpreter_output,
            format_instructions=self.parser.get_format_instructions(),
            use_few_shot=use_few_shot,
            few_shot_posts=few_shot_posts
        )

        # 5. LLM call — retry up to 5 times on API errors or parse failures
        import time
        parsed = None
        for attempt in range(5):
            try:
                response = self.llm.invoke(prompt)
                assert isinstance(response.content, str)
                parsed = self.parser.parse(response.content)
                break
            except Exception as e:
                if attempt == 4:
                    raise
                wait = (attempt + 1) * 3
                print(f"  Attempt {attempt + 1}/5 failed: {e!s:.120} — retrying in {wait}s...")
                time.sleep(wait)
        if parsed is None:
            raise RuntimeError("Generation failed after 5 attempts")

        result = {
            "posts": parsed.posts,
            "few_shot_examples": few_shot_posts
        }

        # 6. Save to file
        if save:
            saved_path = self.save_to_file(result)
            print(f"\nSaved to: {saved_path}")

        return result
    
    def save_to_file(self, result: dict, filename: str = "generated_posts.json"):

        BASE_DIR = Path(__file__).resolve().parent
        output_dir = BASE_DIR / "example_generated"
        output_dir.mkdir(exist_ok=True)

        data = { 
            "posts": [post.model_dump() for post in result["posts"]],
            "few_shot_examples": [
                {"text": example["text"]}
                for example in result["few_shot_examples"]
            ]
        }

        file_path = output_dir / filename

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        
        return str(file_path)

if __name__ == "__main__":
    
    interpreter_output = {
    "what_happened": (
        "ING COO Risk Leon Dusée shared lessons on AI adoption in large organizations, "
        "explaining why many AI pilots stall and how organizations can scale AI successfully."
    ),

    "why_relevance": (
        "Organizations worldwide are struggling with moving AI from isolated experiments into "
        "real operational impact. Dutch organizations across banking, aviation, retail, education, "
        "and healthcare are trying to scale AI adoption while avoiding fragmented pilot initiatives."
    ),

    "why_kickstartai": (
        "KickstartAI helps organizations share practical lessons about AI implementation, "
        "scaling, governance, and adoption. The collaboration between companies like ING, "
        "KLM, NS, and Ahold Delhaize aims to accelerate AI adoption in the Netherlands "
        "while also creating broader societal impact."
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
        "AI should not only improve productivity, but also contribute to areas like financial health, education, and healthcare"
    ],

    "source": "https://lnkd.in/eU8jqvT9"
}

    generator: PostGenerator = PostGenerator()

    result = generator.generate(
        interpreter_output=interpreter_output, 
        k_posts=0
    )

    for i, post in enumerate(result["posts"], 1):
        print(f"\nPOST {i}")
        print("ANGLE:", post.angle)
        print()
        print(post.content)
        print("-" * 60)
    
    print("\nUSED FEW-SHOT POSTS:\n")
    for i, post in enumerate(result["few_shot_examples"], 1):
        print(f"\n EXAMPLE {i}")
        print(post["text"])

# python3 -m Generator.post_generator
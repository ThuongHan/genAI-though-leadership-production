"""Claude Sonnet 4.6 as judge."""
from Generator.utils.llm.claude import ClaudeLLM
from Generator.judge.runner import run_judge

POSTS_PATH = "Generator/example_generated/generated_posts.json"

if __name__ == "__main__":
    llm = ClaudeLLM("claude-opus-4-8")
    run_judge(
        judge_name="opus",
        llm=llm,
        posts_path=POSTS_PATH,
    )

# python3 -m Generator.judge.judge_opus

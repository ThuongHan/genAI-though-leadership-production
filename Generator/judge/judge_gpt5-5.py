"""GPT-5 as judge."""
from Generator.utils.llm.gpt import GPTLLM
from Generator.judge.runner import run_judge

POSTS_PATH = "Generator/example_generated/generated_posts.json"

if __name__ == "__main__":
    llm = GPTLLM("gpt-5.5")
    run_judge(
        judge_name="gpt-5.5",
        llm=llm,
        posts_path=POSTS_PATH,
    )

# python3 -m Generator.judge.judge_gpt5-5

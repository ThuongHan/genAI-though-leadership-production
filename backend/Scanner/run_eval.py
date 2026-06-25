"""
run_eval.py — fire the full model x strategy scoring matrix in one go.

Final experiment (per project decisions):
  - 2 models, ONE provider (Claude) so the vendor is held fixed and the prompting
    STRATEGY is the main axis: Haiku 4.5 (cheap) vs Sonnet 4.6 (strong).
  - Undated aliases on purpose: the system always uses the latest snapshot. Each
    output records the resolved snapshot (e.g. claude-haiku-4-5 -> ...-20251001)
    for reproducibility.
  - 3 strategies: zero_shot, few_shot, cot.
  => 2 x 3 = 6 scoring runs over the 150 annotated articles.

Each run is score_dataset.py: the LLM judge reads the BLANK annotation sheet
(title + source + url + full summary — NO human scores), self-heals failed
articles via retries, and writes data/eval/model_scores_<provider>_<model>_<strategy>.json
with time / token / cost recorded.

After this finishes, compare against the human scores with:
    python evaluate_models.py

Just hit Run (or `python run_eval.py`). Edit MODELS/STRATEGIES below to change
the matrix.
"""

import sys
import traceback

from score_dataset import main as score_main

# --- the experiment matrix (edit here) ---------------------------------------
MODELS = [
    ("claude", "claude-haiku-4-5"),    # cheap tier  (resolves to ...-20251001)
    ("claude", "claude-sonnet-4-6"),   # strong tier
]
STRATEGIES = ["zero_shot", "few_shot", "cot"]


def run_all() -> None:
    base = sys.argv[0]
    combos = [(p, m, s) for (p, m) in MODELS for s in STRATEGIES]
    total = len(combos)
    failures = []

    for i, (provider, model, strat) in enumerate(combos, 1):
        print(f"\n{'#' * 70}")
        print(f"# RUN {i}/{total}:  {provider} / {model} / {strat}")
        print(f"{'#' * 70}")
        sys.argv = [base, "--provider", provider, "--model", model, "--strategy", strat]
        try:
            score_main()
        except SystemExit as e:        # score_dataset calls sys.exit on a missing key etc.
            if e.code not in (0, None):
                failures.append((provider, model, strat, f"exited with code {e.code}"))
        except Exception as e:         # one bad run must not kill the rest of the batch
            traceback.print_exc()
            failures.append((provider, model, strat, str(e).splitlines()[0][:120]))

    print(f"\n{'=' * 70}")
    if failures:
        print(f"FINISHED with {len(failures)}/{total} run(s) that errored:")
        for p, m, s, why in failures:
            print(f"  - {p}/{m}/{s}: {why}")
        print("Re-run those individually, e.g.:")
        p, m, s, _ = failures[0]
        print(f"  python score_dataset.py --provider {p} --model {m} --strategy {s}")
    else:
        print(f"All {total} runs completed. Score files are in data/eval/.")
    print("Next:  python evaluate_models.py")
    print("=" * 70)


if __name__ == "__main__":
    run_all()

"""
compute_pvalues.py — write the Spearman p-values to a SEPARATE file.

Read-only with respect to everything except its own output: it reads the 6
model_scores_*.json files and the scored annotation Excel, computes the Spearman
correlation and its two-sided p-value per config x dimension with scipy, and
writes ONLY data/eval/eval_pvalues.csv. 

(Spearman is the only one of the three reported metrics with a significance test:
MAE is a descriptive error measure and QWK an agreement coefficient — neither has
a p-value as computed here.)

    python3 -m backend.Scanner.compute_pvalues
"""

from __future__ import annotations

import csv
import glob
import sys
from pathlib import Path

_SCANNER = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCANNER))

from scipy.stats import spearmanr

import evaluate_models as em   # reuse the exact loaders used for the metrics

OUTPUT = str(_SCANNER / "data" / "eval" / "eval_pvalues.csv")


def main() -> None:
    human = em.load_human_scores(em.DEFAULT_ANNOTATIONS)
    if not human:
        print("No human scores found — nothing to compute.")
        return

    files = sorted(glob.glob(em.DEFAULT_GLOB))
    if not files:
        print(f"No model score files found ({em.DEFAULT_GLOB}).")
        return

    rows = []
    print(f"{'config':<42}{'dimension':<16}{'rho':>8}{'p_value':>12}")
    print("-" * 78)
    for path in files:
        label, model = em.load_model_scores(path)
        for dim in em.RUBRIC_DIMS:
            h, m = em.paired(human, model, dim)
            n = len(h)
            if n < 2:
                continue
            res = spearmanr(h, m)
            rho, p = float(res.correlation), float(res.pvalue)
            rows.append({"model": label, "dimension": dim, "n": n,
                         "spearman": round(rho, 4), "p_value": p})
            print(f"{label:<42}{dim:<16}{rho:>8.3f}{p:>12.2e}")

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "dimension", "n", "spearman", "p_value"])
        w.writeheader()
        w.writerows(rows)

    pmax = max(r["p_value"] for r in rows)
    print("-" * 78)
    print(f"largest (weakest) p-value: {pmax:.2e}  |  all p < 0.001: {all(r['p_value'] < 0.001 for r in rows)}")
    print(f"Wrote {len(rows)} rows -> {OUTPUT}  (eval_summary.csv untouched)")


if __name__ == "__main__":
    main()
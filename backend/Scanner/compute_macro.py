"""
compute_macro.py — write the macro-averaged headline table to a SEPARATE file.

For each config it averages the three reported metrics (Spearman, QWK, MAE) across
the four dimensions (the macro-average) and also carries the weighted-grade rho/MAE
straight from the per-dimension computation. Writes ONLY data/eval/eval_summary_macro.csv
— it never opens eval_summary.csv.

    python compute_macro.py
"""

from __future__ import annotations

import csv
import glob
import sys
from pathlib import Path

_SCANNER = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCANNER))

import evaluate_models as em   # reuse the exact metric computation

OUTPUT = str(_SCANNER / "data" / "eval" / "eval_summary_macro.csv")


def _mean(values: list[float]) -> float | str:
    vals = [v for v in values if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else ""


def main() -> None:
    human = em.load_human_scores(em.DEFAULT_ANNOTATIONS)
    if not human:
        print("No human scores found — nothing to compute.")
        return
    files = sorted(glob.glob(em.DEFAULT_GLOB))
    if not files:
        print(f"No model score files found ({em.DEFAULT_GLOB}).")
        return

    out_rows = []
    for path in files:
        label, model = em.load_model_scores(path)
        rows = em.evaluate_one(label, human, model)
        dim_rows = [r for r in rows if r["dimension"] in em.RUBRIC_DIMS and r["n"] > 0]
        wrow = next((r for r in rows if r["dimension"] == "weighted"), None)
        out_rows.append({
            "config": label,
            "n": dim_rows[0]["n"] if dim_rows else 0,
            "spearman_macro": _mean([r["spearman"] for r in dim_rows]),
            "qwk_macro": _mean([r["qwk"] for r in dim_rows]),
            "mae_macro": _mean([r["mae"] for r in dim_rows]),
            "spearman_weighted": round(wrow["spearman"], 4) if wrow else "",
            "mae_weighted": round(wrow["mae"], 4) if wrow else "",
        })

    cols = ["config", "n", "spearman_macro", "qwk_macro", "mae_macro",
            "spearman_weighted", "mae_weighted"]
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)

    # Console view
    print(f"{'config':<42}{'rho':>8}{'qwk':>8}{'mae':>8}{'rho_w':>8}{'mae_w':>8}")
    print("-" * 82)
    for r in out_rows:
        print(f"{r['config']:<42}{r['spearman_macro']:>8}{r['qwk_macro']:>8}"
              f"{r['mae_macro']:>8}{r['spearman_weighted']:>8}{r['mae_weighted']:>8}")
    print("-" * 82)
    print(f"Wrote {len(out_rows)} rows -> {OUTPUT}  (eval_summary.csv untouched)")


if __name__ == "__main__":
    main()
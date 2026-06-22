#!/bin/bash
# run_all_extractors.sh
# =====================
# Runs the full pipeline (Steps 1–5) for all three extractor methods.
# Step 1 (Ingestion) only runs once since the raw data is shared.
#
# Usage:
#   chmod +x run_all_extractors.sh
#   ./run_all_extractors.sh

set -e  # stop immediately if any command fails

echo ""
echo "========================================================"
echo "  KickstartAI — Running all 3 extractors"
echo "========================================================"

# ── Step 1: run once (raw data is the same for all extractors) ────────────────
echo ""
echo ">>> STEP 1 — Ingestion (shared, runs once)"
python main.py --steps 1

# ── Method 1: LLM ─────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo ">>> METHOD 1: llm"
echo "========================================================"
python main.py --extractor llm --steps 2 3 4 5

# ── Method 2: Sensemaking ─────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo ">>> METHOD 2: sensemaking"
echo "========================================================"
python main.py --extractor sensemaking --steps 2 3 4 5

# ── Method 3: Multi-Model Agreement ──────────────────────────────────────────
echo ""
echo "========================================================"
echo ">>> METHOD 3: multimodel"
echo "========================================================"
python main.py --extractor multimodel --steps 2 3 4 5

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  All extractors complete. Output files:"
echo "========================================================"
echo ""
echo "  data/processed/beliefs_extracted_llm.json"
echo "  data/processed/beliefs_extracted_sensemaking.json"
echo "  data/processed/beliefs_extracted_multimodel.json"
echo ""
echo "  data/processed/belief_repository_llm.json"
echo "  data/processed/belief_repository_sensemaking.json"
echo "  data/processed/belief_repository_multimodel.json"
echo ""
echo "  data/belief_store/beliefs_with_embeddings_llm.json"
echo "  data/belief_store/beliefs_with_embeddings_sensemaking.json"
echo "  data/belief_store/beliefs_with_embeddings_multimodel.json"
echo ""

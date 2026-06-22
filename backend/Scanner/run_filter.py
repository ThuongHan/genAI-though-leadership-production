"""Entry point for running the weighted filter on the newest scan.

Mirrors run_scanner.py — made for the VS Code "Run Python File" button, which
passes no CLI flags. Edit the two constants below to change behaviour, then hit
Run. It filters the newest scanner_output*.json in data/scans/ down to a top-5.

For full control (a specific scan, --excerpt, --max-chars, --model, ...) use the
module form instead:
    python -m main_scanner.weighted_filter --strategy few_shot --provider openai
"""

import sys

from main_scanner.weighted_filter import main

# --- knobs (the Run button can't pass flags, so set them here) ---------------
STRATEGY = "zero_shot"   # zero_shot | few_shot | cot
PROVIDER = "claude"      # claude | openai


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--strategy", STRATEGY, "--provider", PROVIDER]
    main()
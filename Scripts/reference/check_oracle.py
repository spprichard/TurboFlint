#!/usr/bin/env python3
"""Runs the release `turboflint` binary on every oracle fixture and checks greedy token ids.

    swift build -c release --product turboflint && python3 -I Scripts/reference/check_oracle.py

Fixtures come from Scripts/reference/flint_reference.py (FP32). A mismatch at a step whose oracle
top-2 logit gap is tiny (< 0.05) is reported as a near-tie rather than a failure.
"""
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
fixtures = json.loads((root / "Scripts/reference/fixtures/oracle_greedy.json").read_text())
binary = root / ".build/release/turboflint"
failed = False
for name, fx in fixtures.items():
    out = subprocess.run(
        [str(binary), "--prompt-ids", ",".join(map(str, fx["prompt_ids"])),
         "--max-new", str(len(fx["generated_ids"])), "--json"],
        capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    got = json.loads(out)["generated_ids"]
    want = fx["generated_ids"]
    first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
    if first is None and len(got) == len(want):
        print(f"PASS {name}: {len(want)} tokens")
        continue
    gap = None
    if first is not None and "top_logits" in fx:
        top = fx["top_logits"][first]
        gap = top[0] - top[1]
    if gap is not None and gap < 0.05:
        print(f"NEAR-TIE {name}: diverged at step {first} where the oracle's top-2 gap is {gap:.3f}")
    else:
        failed = True
        print(f"FAIL {name}: diverged at step {first}\n  want {want}\n  got  {got}")
sys.exit(1 if failed else 0)

"""
Ensemble scorer: loads checkpoints, predicts the frozen val split (h-flip TTA),
scores every member solo, then greedy forward selection over averaged preds.
If APPLY=1 (default), the selected subset is written to models/ens_NN.pt so
submission_notebook.py picks it up.

  ENS_PATTERNS  space-separated globs (default "models/r1_*.pt models/r2_*.pt")
  APPLY         "1" to rewrite models/ens_*.pt with the selected subset
"""
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import prepare
import train as T

PATTERNS = os.environ.get("ENS_PATTERNS", "models/r1_*.pt models/r2_*.pt").split()
APPLY = os.environ.get("APPLY", "1") == "1"

d = prepare.load()
paths = sorted(p for pat in PATTERNS for p in glob.glob(pat))
if not paths:
    print("ens_eval: no checkpoints matched", PATTERNS)
    sys.exit(1)

preds = {}
for p in paths:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    cfg.setdefault("DECODE", "cumsum")   # pre-DECODE checkpoints are cumsum;
    for k, v in cfg.items():             # must reset the global per checkpoint
        setattr(T, k, v)
    m = T.Model().to(T.DEVICE)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    stats = {k: np.asarray(v) for k, v in ck["stats"].items()}
    pr = T.predict_val(d, m, stats)
    del m
    preds[p] = pr
    print(f"member {p} solo {prepare.evaluate(d, pr):.6f}", flush=True)

names = list(preds)
pool_avg = np.mean([preds[n] for n in names], 0)
print(f"ens_pool_rmse: {prepare.evaluate(d, pool_avg):.6f}  ({len(names)} members)", flush=True)

# greedy forward selection, equal weights, no re-adding (the submission
# notebook averages the selected files uniformly, so selection must match)
chosen, best = [], np.inf
while True:
    cand = None
    for n in names:
        if n in chosen:
            continue
        avg = np.mean([preds[c] for c in chosen + [n]], 0)
        s = prepare.evaluate(d, avg)
        if s < best - 1e-4:
            best, cand = s, n
    if cand is None:
        break
    chosen.append(cand)
    print(f"greedy + {cand} -> {best:.6f}", flush=True)

print(f"ens_best_rmse: {best:.6f}")
print(f"ens_members: {json.dumps(chosen)}")

if APPLY and chosen:
    for old in glob.glob("models/ens_*.pt"):
        os.remove(old)
    for i, src in enumerate(sorted(chosen)):
        shutil.copy(src, f"models/ens_{i:02d}.pt")
    (ROOT / "models/ENSEMBLE.json").write_text(
        json.dumps({"members": chosen, "val_rmse": best}, indent=2))
    print(f"applied: {len(chosen)} files -> models/ens_*.pt", flush=True)

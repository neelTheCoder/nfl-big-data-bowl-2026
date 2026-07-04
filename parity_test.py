"""
Parity test: does the submission notebook's API-reconstruction path produce the
same predictions as the direct-array path used for all local validation?

The local eval (prepare.load -> predict_val) and the Kaggle submission
(play_arrays/predict over API dataframes) are DIFFERENT code paths; only the
former has ever been scored locally. A systematic discrepancy (slot ordering,
frame windowing, angle handling, n_out mapping) would depress the real-test
score while leaving local val untouched. This feeds the raw weeks-17/18 csvs
through the notebook's predict() exactly as the eval API would, and compares
RMSE against predict_val on the same plays with the same checkpoint.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch

ROOT = Path(__file__).resolve().parent
COMP = Path(os.path.expanduser(
    "~/.cache/kagglehub/competitions/nfl-big-data-bowl-2026-prediction"))
CKPT = os.environ.get("PARITY_CKPT", "/tmp/dl_kf0.pt")

# --- stage a scratch MODEL_DIR with just the one checkpoint ---
scratch = ROOT / "kaggle_pkg/parity_scratch"
(scratch / "models").mkdir(parents=True, exist_ok=True)
import shutil
shutil.copy(ROOT / "prepare.py", scratch / "prepare.py")
shutil.copy(ROOT / "train.py", scratch / "train.py")
shutil.copy(CKPT, scratch / "models/ens_00.pt")

os.environ["MODEL_DIR"] = str(scratch)
os.environ["COMP_DIR"] = str(COMP)

# --- exec the notebook up to (not including) the gateway launch ---
src = (ROOT / "kaggle_pkg/notebook/nfl_bdb26_submit.py").read_text()
marker = "sys.path.insert(0, COMP_DIR)"
assert marker in src
src = src[:src.index(marker)]
ns = {"__name__": "parity_nb"}
exec(compile(src, "nfl_bdb26_submit.py<truncated>", "exec"), ns)
predict = ns["predict"]
print(f"notebook loaded: {len(ns['MODELS'])} model(s)", flush=True)

# --- run every weeks-17/18 play through the API path ---
sse, n = 0.0, 0
plays_done = 0
for wk in (17, 18):
    fin = pd.read_csv(COMP / f"train/input_2023_w{wk:02d}.csv")
    fout = pd.read_csv(COMP / f"train/output_2023_w{wk:02d}.csv")
    for (gid, pid), gout in fout.groupby(["game_id", "play_id"]):
        gin = fin[(fin.game_id == gid) & (fin.play_id == pid)]
        if gin.empty:
            continue
        test = pl.from_pandas(gout[["game_id", "play_id", "nfl_id", "frame_id"]])
        test_input = pl.from_pandas(gin)
        try:
            pred = predict(test, test_input)
        except Exception as e:
            print(f"play {gid}_{pid} FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        dx = pred["x"].to_numpy() - gout["x"].to_numpy()
        dy = pred["y"].to_numpy() - gout["y"].to_numpy()
        sse += float((dx * dx).sum() + (dy * dy).sum())
        n += 2 * len(gout)
        plays_done += 1
        if plays_done % 200 == 0:
            print(f"  {plays_done} plays, running api-path rmse "
                  f"{np.sqrt(sse / n):.6f}", flush=True)

api_rmse = np.sqrt(sse / n)
print(f"API-path RMSE (weeks 17-18, {plays_done} plays): {api_rmse:.6f}", flush=True)

# --- direct-array path on the same plays with the same checkpoint ---
os.environ["NFL_SPLIT"] = "weeks:17,18"
sys.path.insert(0, str(scratch))
import prepare
import train as T

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = dict(ck["config"])
cfg.setdefault("DECODE", "cumsum")
for k, v in cfg.items():
    setattr(T, k, v)
m = T.Model().to(T.DEVICE)
m.load_state_dict(ck["state_dict"])
m.eval()
stats = {k: np.asarray(v) for k, v in ck["stats"].items()}
d = prepare.load()
direct_rmse = prepare.evaluate(d, T.predict_val(d, m, stats))
print(f"direct-path RMSE (weeks 17-18): {direct_rmse:.6f}")
print(f"PARITY GAP: {api_rmse - direct_rmse:+.6f}")

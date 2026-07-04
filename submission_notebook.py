"""
Kaggle late-submission inference notebook.

Setup on Kaggle:
  1. Create a private dataset "nfl-bdb26-models" containing: prepare.py,
     train.py, and models/*.pt checkpoints.
  2. New notebook, attach that dataset + the competition data, internet OFF.
  3. Paste this file as the notebook body and Submit.

API verified against reference/nfl2026-1st-place-inference.py:
  NFLInferenceServer(predict) / serve() / run_local_gateway((<comp path>,)),
  predict(test: pl.DataFrame, test_input: pl.DataFrame) -> DataFrame with x,y.
"""
import os
import sys
import numpy as np
import pandas as pd
import polars as pl
import torch

MODEL_DIR = os.environ.get("MODEL_DIR", "/kaggle/input/nfl-bdb26-models")
COMP_DIR = os.environ.get("COMP_DIR", "/kaggle/input/nfl-big-data-bowl-2026-prediction")
sys.path.insert(0, MODEL_DIR)

import prepare
import train as T
from prepare import P_MAX, T_IN, T_OUT, ROLE_MAP

MODELS, STATS = [], None
_all = [f for f in sorted(os.listdir(f"{MODEL_DIR}/models")) if f.endswith(".pt")]
# prefer the final overnight ensemble (ens_*.pt); fall back to any checkpoint
_ckpts = [f for f in _all if f.startswith("ens_")] or _all
for f in _ckpts:
    ck = torch.load(f"{MODEL_DIR}/models/{f}", map_location="cpu", weights_only=False)
    for k, v in ck["config"].items():
        setattr(T, k, v)
    m = T.Model()
    m.load_state_dict(ck["state_dict"])
    m.eval()
    MODELS.append(m)
    STATS = {k: np.asarray(v) for k, v in ck["stats"].items()}
print(f"loaded {len(MODELS)} checkpoints")


def play_arrays(test: pl.DataFrame, test_input: pl.DataFrame):
    """Single play -> the same padded arrays prepare.preprocess builds."""
    g = test_input.to_pandas()
    g = prepare._canonicalize(g)
    flipped = bool((g["play_direction"] == "left").any())

    n_out_map = test.to_pandas().groupby("nfl_id")["frame_id"].max().to_dict()

    first = g[g["frame_id"] == g["frame_id"].min()]
    order_key = first.apply(
        lambda r: (ROLE_MAP.get(r["player_role"], 2) if r["player_side"] == "Offense" else 10,
                   r["nfl_id"]), axis=1)
    slot_ids = list(first.assign(k=order_key).sort_values("k")["nfl_id"])[:P_MAX]
    slot_of = {nid: i for i, nid in enumerate(slot_ids)}

    X = np.zeros((1, P_MAX, T_IN, 6), np.float32)
    in_mask = np.zeros((1, P_MAX, T_IN), bool)
    role = np.full((1, P_MAX), 2, np.int8)
    side = np.zeros((1, P_MAX), np.int8)
    n_out = np.zeros((1, P_MAX), np.int16)
    for nid, pg in g.groupby("nfl_id"):
        if nid not in slot_of:
            continue
        i = slot_of[nid]
        pg = pg.sort_values("frame_id").tail(T_IN)
        k = len(pg)
        X[0, i, T_IN - k:, 0] = pg["x"]
        X[0, i, T_IN - k:, 1] = pg["y"]
        X[0, i, T_IN - k:, 2] = pg["s"]
        X[0, i, T_IN - k:, 3] = pg["a"]
        X[0, i, T_IN - k:, 4] = np.deg2rad(pg["dir"])
        X[0, i, T_IN - k:, 5] = np.deg2rad(pg["o"])
        in_mask[0, i, T_IN - k:] = True
        r = pg.iloc[-1]
        role[0, i] = ROLE_MAP.get(r["player_role"], 2)
        side[0, i] = 1 if r["player_side"] == "Offense" else 0
        n_out[0, i] = int(n_out_map.get(nid, 0))
    r0 = g.iloc[-1]
    d = dict(
        X=X, in_mask=in_mask,
        Y=np.zeros((1, P_MAX, T_OUT, 2), np.float32),
        out_mask=np.zeros((1, P_MAX, T_OUT), bool),
        ball_land=np.array([[r0["ball_land_x"], r0["ball_land_y"]]], np.float32),
        role=role, side=side, n_out=n_out,
    )
    return d, slot_of, flipped


def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pl.DataFrame:
    d, slot_of, flipped = play_arrays(test, test_input)
    tstd = torch.as_tensor(STATS["tgt_std"])
    tmean = torch.as_tensor(STATS["tgt_mean"])
    b = T.make_batch(d, np.array([0]), STATS, train=False)
    b2 = T.make_batch(d, np.array([0]), STATS, train=False, flip_all=True)
    with torch.no_grad():
        acc = None
        for m in MODELS:
            mu = (m(b)[0] * tstd + tmean + b["last_xy"][:, :, None, :]).numpy()
            m2 = (m(b2)[0] * tstd + tmean + b2["last_xy"][:, :, None, :]).numpy()
            m2[..., 1] = prepare.FIELD_Y - m2[..., 1]     # unflip TTA branch
            p = 0.5 * (mu + m2)
            acc = p if acc is None else acc + p
        preds = acc / len(MODELS)                # [1, P, T_OUT, 2]

    rows = []
    for gid, pid, nid, fid in test[["game_id", "play_id", "nfl_id", "frame_id"]].iter_rows():
        x, y = preds[0, slot_of[nid], min(int(fid) - 1, T_OUT - 1)]
        if flipped:
            x, y = prepare.FIELD_X - x, prepare.FIELD_Y - y
        rows.append((float(x), float(y)))
    return pl.DataFrame(rows, schema=["x", "y"], orient="row")


sys.path.insert(0, COMP_DIR)   # kaggle_evaluation ships with the competition data
import kaggle_evaluation.nfl_inference_server  # noqa: E402
inference_server = kaggle_evaluation.nfl_inference_server.NFLInferenceServer(predict)
if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway((COMP_DIR + "/",))

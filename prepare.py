"""
prepare.py — FIXED runtime for NFL BDB 2026 autoresearch. DO NOT MODIFY.

Responsibilities:
  1. Download competition data (kagglehub, requires ~/.kaggle/kaggle.json).
  2. Preprocess raw CSVs into padded play-level tensors (data/proc.npz).
  3. Provide the fixed dataloading + evaluation used by train.py.

The evaluation metric here is the competition metric: RMSE over the x and y
coordinates of every scored (player_to_predict) output frame in the
validation split. train.py may not reimplement it.

Usage:
  python prepare.py              # download + preprocess real data
  python prepare.py --synthetic  # write a small synthetic proc.npz (pipeline smoke test)
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

# ----------------------------------------------------------------------------
# Fixed constants (the contract — train.py imports these, never changes them)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
# NFL_DATA_DIR: writable data dir (Kaggle mounts code read-only -> /kaggle/working)
DATA_DIR = Path(os.environ.get("NFL_DATA_DIR", ROOT / "data"))
PROC_PATH = DATA_DIR / "proc.npz"
COMP = "nfl-big-data-bowl-2026-prediction"

P_MAX = 22          # player slots per play
T_IN = 50           # input frames kept (last T_IN before throw; extra history feeds early-start augmentation)
T_OUT = 48          # max output frames predicted (longer plays dropped, matches 1st place decoder)
VAL_WEEKS = (17, 18)  # temporal holdout, mirrors the live-LB "future games" regime
FIELD_X, FIELD_Y = 120.0, 53.3

ROLE_MAP = {"Passer": 0, "Targeted Receiver": 1, "Other Route Runner": 2, "Defensive Coverage": 3}


# ----------------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------------
def download() -> Path:
    import kagglehub
    path = Path(kagglehub.competition_download(COMP))
    print("competition files at:", path)
    return path


def find_raw_dir() -> Path:
    """Locate the raw csvs (env override, kagglehub cache, or data/raw)."""
    candidates = [
        DATA_DIR / "raw",
        Path.home() / ".cache" / "kagglehub" / "competitions" / COMP,
    ]
    if os.environ.get("NFL_RAW_DIR"):
        candidates.insert(0, Path(os.environ["NFL_RAW_DIR"]))
    for c in candidates:
        if c.exists():
            hits = list(c.rglob("input_2023_w01.csv"))
            if hits:
                return hits[0].parent.parent if hits[0].parent.name == "train" else hits[0].parent
    raise FileNotFoundError(
        "raw data not found; run `python prepare.py` after placing kaggle.json in ~/.kaggle/"
    )


# ----------------------------------------------------------------------------
# Preprocess
# ----------------------------------------------------------------------------
def _canonicalize(df):
    """Make all plays left-to-right (offense moving toward increasing x)."""
    left = df["play_direction"] == "left"
    df.loc[left, "x"] = FIELD_X - df.loc[left, "x"]
    df.loc[left, "y"] = FIELD_Y - df.loc[left, "y"]
    for col in ("dir", "o"):
        df.loc[left, col] = (df.loc[left, col] + 180.0) % 360.0
    df.loc[left, "ball_land_x"] = FIELD_X - df.loc[left, "ball_land_x"]
    df.loc[left, "ball_land_y"] = FIELD_Y - df.loc[left, "ball_land_y"]
    return df


def preprocess():
    import pandas as pd

    raw = find_raw_dir()
    train_dir = raw / "train" if (raw / "train").exists() else raw
    weeks = sorted(train_dir.glob("input_2023_w*.csv"))
    assert weeks, f"no input csvs under {train_dir}"

    plays = []  # list of dict per play
    n_dropped_long, n_dropped_nopasser = 0, 0

    for wf in weeks:
        week = int(wf.stem.split("_w")[-1])
        fin = pd.read_csv(wf)
        fout = pd.read_csv(wf.parent / wf.name.replace("input_", "output_"))
        fin = _canonicalize(fin)

        # canonicalize outputs using play_direction from input
        dir_map = fin.groupby(["game_id", "play_id"])["play_direction"].first()
        fout = fout.merge(dir_map.rename("play_direction"),
                          left_on=["game_id", "play_id"], right_index=True, how="left")
        left = fout["play_direction"] == "left"
        fout.loc[left, "x"] = FIELD_X - fout.loc[left, "x"]
        fout.loc[left, "y"] = FIELD_Y - fout.loc[left, "y"]

        out_groups = {k: v for k, v in fout.groupby(["game_id", "play_id"])}

        for (gid, pid), g in fin.groupby(["game_id", "play_id"]):
            og = out_groups.get((gid, pid))
            if og is None:
                continue
            if g["num_frames_output"].max() > T_OUT:
                n_dropped_long += 1
                continue
            if not (g["player_role"] == "Passer").any():
                n_dropped_nopasser += 1
                continue

            # stable slot order: passer, target, other offense, defense
            first = g[g["frame_id"] == g["frame_id"].min()]
            order_key = first.apply(
                lambda r: (ROLE_MAP.get(r["player_role"], 2) if r["player_side"] == "Offense" else 10,
                           r["nfl_id"]), axis=1)
            slot_ids = list(first.assign(k=order_key).sort_values("k")["nfl_id"])[:P_MAX]
            slot_of = {nid: i for i, nid in enumerate(slot_ids)}

            X = np.zeros((P_MAX, T_IN, 6), np.float32)       # x y s a dir_rad o_rad
            in_mask = np.zeros((P_MAX, T_IN), bool)
            Y = np.zeros((P_MAX, T_OUT, 2), np.float32)
            out_mask = np.zeros((P_MAX, T_OUT), bool)
            scored = np.zeros(P_MAX, bool)
            role = np.full(P_MAX, 2, np.int8)
            side = np.zeros(P_MAX, np.int8)
            n_out = np.zeros(P_MAX, np.int16)

            for nid, pg in g.groupby("nfl_id"):
                if nid not in slot_of:
                    continue
                i = slot_of[nid]
                pg = pg.sort_values("frame_id").tail(T_IN)
                k = len(pg)
                X[i, T_IN - k:, 0] = pg["x"]
                X[i, T_IN - k:, 1] = pg["y"]
                X[i, T_IN - k:, 2] = pg["s"]
                X[i, T_IN - k:, 3] = pg["a"]
                X[i, T_IN - k:, 4] = np.deg2rad(pg["dir"])
                X[i, T_IN - k:, 5] = np.deg2rad(pg["o"])
                in_mask[i, T_IN - k:] = True
                r = pg.iloc[-1]
                role[i] = ROLE_MAP.get(r["player_role"], 2)
                side[i] = 1 if r["player_side"] == "Offense" else 0
                scored[i] = bool(r["player_to_predict"])
                n_out[i] = int(r["num_frames_output"])

            for nid, pg in og.groupby("nfl_id"):
                if nid not in slot_of:
                    continue
                i = slot_of[nid]
                pg = pg.sort_values("frame_id")
                fidx = pg["frame_id"].to_numpy() - 1
                keep = fidx < T_OUT
                Y[i, fidx[keep], 0] = pg["x"].to_numpy()[keep]
                Y[i, fidx[keep], 1] = pg["y"].to_numpy()[keep]
                out_mask[i, fidx[keep]] = True

            r0 = g.iloc[-1]
            plays.append(dict(
                X=X, in_mask=in_mask, Y=Y, out_mask=out_mask, scored=scored,
                role=role, side=side, n_out=n_out,
                ball_land=np.array([r0["ball_land_x"], r0["ball_land_y"]], np.float32),
                week=week, game_id=gid, play_id=pid,
            ))
        print(f"week {week:2d}: cumulative plays={len(plays)}")

    _save(plays, synthetic=False)
    print(f"dropped: too_long={n_dropped_long} no_passer={n_dropped_nopasser}")


def _save(plays, synthetic):
    DATA_DIR.mkdir(exist_ok=True)
    stack = lambda k: np.stack([p[k] for p in plays])
    np.savez_compressed(
        PROC_PATH,
        X=stack("X"), in_mask=stack("in_mask"), Y=stack("Y"), out_mask=stack("out_mask"),
        scored=stack("scored"), role=stack("role"), side=stack("side"), n_out=stack("n_out"),
        ball_land=stack("ball_land"),
        week=np.array([p["week"] for p in plays], np.int16),
        game_id=np.array([p["game_id"] for p in plays], np.int64),
        play_id=np.array([p["play_id"] for p in plays], np.int64),
        synthetic=np.array(synthetic),
    )
    print(f"saved {len(plays)} plays -> {PROC_PATH} ({PROC_PATH.stat().st_size/1e6:.0f} MB)")


# ----------------------------------------------------------------------------
# Synthetic smoke data (same schema; lets train.py run before real data lands)
# ----------------------------------------------------------------------------
def synthetic(n_plays=400, seed=0):
    rng = np.random.default_rng(seed)
    plays = []
    for i in range(n_plays):
        X = np.zeros((P_MAX, T_IN, 6), np.float32)
        in_mask = np.ones((P_MAX, T_IN), bool)
        Y = np.zeros((P_MAX, T_OUT, 2), np.float32)
        out_mask = np.zeros((P_MAX, T_OUT), bool)
        n_out_play = int(rng.integers(5, T_OUT))
        pos = rng.uniform([20, 5], [100, 48], (P_MAX, 2)).astype(np.float32)
        vel = rng.normal(0, 2.5, (P_MAX, 2)).astype(np.float32)
        ball_land = rng.uniform([30, 5], [110, 48]).astype(np.float32)
        for t in range(T_IN):
            vel += rng.normal(0, 0.35, (P_MAX, 2)).astype(np.float32)
            pos += vel * 0.1
            X[:, t, 0:2] = pos
            X[:, t, 2] = np.linalg.norm(vel, axis=1)
            X[:, t, 4] = np.arctan2(vel[:, 0], vel[:, 1])
        p, v = pos.copy(), vel.copy()
        for t in range(n_out_play):
            pull = (ball_land - p) * 0.02
            v += pull + rng.normal(0, 0.3, (P_MAX, 2)).astype(np.float32)
            p += v * 0.1
            Y[:, t] = p
            out_mask[:, t] = True
        scored = np.zeros(P_MAX, bool)
        scored[rng.choice(P_MAX, 6, replace=False)] = True
        role = np.full(P_MAX, 2, np.int8); role[0] = 0; role[1] = 1
        side = np.array([1] * 11 + [0] * 11, np.int8)
        n_out = np.full(P_MAX, n_out_play, np.int16)
        plays.append(dict(X=X, in_mask=in_mask, Y=Y, out_mask=out_mask, scored=scored,
                          role=role, side=side, n_out=n_out, ball_land=ball_land,
                          week=1 + (i % 18), game_id=i, play_id=i))
    _save(plays, synthetic=True)


# ----------------------------------------------------------------------------
# Fixed loading + evaluation (imported by train.py)
# ----------------------------------------------------------------------------
def load():
    """Returns dict of numpy arrays + index arrays train_idx/val_idx.

    NFL_SPLIT selects the split (train = complement of val):
      weeks:17,18       held-out weeks (default; the original frozen split)
      weeks:14-18       week range (matches the real test's week range)
      gamefold:5:0      val = plays with game_id % 5 == 0 (1st-place-style CV)
      none              no validation (train on everything, for final models)
    """
    d = dict(np.load(PROC_PATH, allow_pickle=False))
    spec = os.environ.get("NFL_SPLIT", "weeks:" + ",".join(map(str, VAL_WEEKS)))
    kind, _, arg = spec.partition(":")
    if kind == "weeks":
        if "-" in arg:
            lo, hi = arg.split("-")
            wks = list(range(int(lo), int(hi) + 1))
        else:
            wks = [int(w) for w in arg.split(",") if w]
        val = np.isin(d["week"], wks)
    elif kind == "gamefold":
        n, f = (int(x) for x in arg.split(":"))
        # multiplicative hash: raw game_id % n is biased (ids encode dates)
        h = (d["game_id"].astype(np.uint64) * np.uint64(2654435761)) >> np.uint64(13)
        val = (h % np.uint64(n)).astype(int) == f
    elif kind == "none":
        val = np.zeros(len(d["week"]), bool)
    else:
        raise ValueError(f"bad NFL_SPLIT {spec!r}")
    d["train_idx"] = np.where(~val)[0]
    d["val_idx"] = np.where(val)[0]
    return d


def evaluate(d, preds):
    """Competition metric on the validation split.

    preds: float array [n_val, P_MAX, T_OUT, 2] of ABSOLUTE canonical (x, y).
    Score counts every (frame, coordinate) of players with scored=True and
    out_mask=True, exactly like the leaderboard RMSE.
    """
    vi = d["val_idx"]
    assert preds.shape == (len(vi), P_MAX, T_OUT, 2), f"bad preds shape {preds.shape}"
    m = d["out_mask"][vi] & d["scored"][vi][:, :, None]
    err = preds[m] - d["Y"][vi][m]          # [n_rows, 2]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    np.save(DATA_DIR / "val_preds.npy", preds.astype(np.float32))
    return rmse


def cv_baseline(d):
    """Constant-velocity extrapolation baseline — the score to beat trivially."""
    vi = d["val_idx"]
    X = d["X"][vi]
    last = X[:, :, -1, :]
    vx = last[:, :, 2] * np.sin(last[:, :, 4])
    vy = last[:, :, 2] * np.cos(last[:, :, 4])
    t = (np.arange(T_OUT, dtype=np.float32) + 1) * 0.1
    preds = np.stack([
        last[:, :, 0, None] + vx[:, :, None] * t,
        last[:, :, 1, None] + vy[:, :, None] * t,
    ], -1)
    return evaluate(d, preds)


if __name__ == "__main__":
    if "--synthetic" in sys.argv:
        synthetic()
    else:
        download()
        preprocess()
    d = load()
    print(f"plays={len(d['week'])} train={len(d['train_idx'])} val={len(d['val_idx'])}")
    print(f"constant-velocity baseline val_rmse: {cv_baseline(d):.4f}")

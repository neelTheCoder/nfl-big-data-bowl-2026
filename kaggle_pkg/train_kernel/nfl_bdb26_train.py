"""
Kaggle GPU training kernel — runs server-side (laptop can be off).

Trains the 1st-place-scale recipe on their exact protocol, which is the only
configuration with *verified* out-of-distribution performance (private 0.4634
on the 2025-season test after training on 2023 only):

  stage 0: preprocess raw competition csvs -> /kaggle/working/data/proc.npz
  stage 1: shift screen — train weeks 1-13, validate weeks 14-18 (the test's
           week range) for (a) our lean config and (b) 1st-place scale; logs
           which generalizes better under a within-season temporal shift
  stage 2: 5-fold CV (game_id % 5) x 1st-place-scale config, ~37k steps each,
           best-of-{raw,ema,swa} on the fold's val; saves models/kf{f}.pt

Everything is wall-clock guarded to finish cleanly before Kaggle's session
limit (outputs are only preserved on clean completion).
"""
import json
import os
import shutil
import subprocess
import sys
import time
from glob import glob
from pathlib import Path

T0 = time.time()
# guard: this week's remaining GPU quota (~10.9h) binds before the 12h cap;
# a quota kill loses outputs, so exit with margin
MAX_SEC = float(os.environ.get("MAX_SEC", 10.4 * 3600))
WORK = Path("/kaggle/working")


def elapsed():
    return time.time() - T0


def find_dir(marker, roots=("/kaggle/input",)):
    for root in roots:
        hits = glob(f"{root}/**/{marker}", recursive=True)
        if hits:
            return os.path.dirname(hits[0])
    raise FileNotFoundError(marker)


CODE_DIR = find_dir("prepare.py")
COMP_DIR = find_dir("kaggle_evaluation")
print(f"CODE_DIR={CODE_DIR}\nCOMP_DIR={COMP_DIR}", flush=True)

# fail fast if the assigned GPU has no torch kernel image (e.g. P100/sm_60)
import torch  # noqa: E402

if torch.cuda.is_available():
    try:
        (torch.ones(2, device="cuda") * 2).sum().item()
        print("cuda OK:", torch.cuda.get_device_name(0), flush=True)
    except Exception as e:
        raise SystemExit(f"CUDA broken on this machine shape: {e}")
else:
    print("WARNING: no cuda available; training will use CPU (slow)", flush=True)

# run code from a writable copy (prepare writes next to itself by default)
shutil.copy(f"{CODE_DIR}/prepare.py", WORK / "prepare.py")
shutil.copy(f"{CODE_DIR}/train.py", WORK / "train.py")
os.chdir(WORK)
sys.path.insert(0, str(WORK))

BASE_ENV = dict(
    os.environ,
    NFL_DATA_DIR=str(WORK / "data"),
    NFL_RAW_DIR=COMP_DIR,
)

# ---------------- stage 0: preprocess ----------------
os.environ.update(NFL_DATA_DIR=BASE_ENV["NFL_DATA_DIR"], NFL_RAW_DIR=COMP_DIR)
import prepare  # noqa: E402  (reads NFL_DATA_DIR at import time)

if not (WORK / "data/proc.npz").exists():
    print(f"[{elapsed():6.0f}s] preprocessing...", flush=True)
    prepare.preprocess()
print(f"[{elapsed():6.0f}s] preprocess done", flush=True)

FIRST_PLACE = dict(HP_MULT=64, HP_DIM1=256, HP_N_LAYERS=3, HP_FF_DIM=1024,
                   HP_LR="5e-4", HP_AUX_W="0.8")
LEAN = dict(HP_MULT=32, HP_DIM1=192, HP_N_LAYERS=2, HP_FF_DIM=768,
            HP_LR="1e-3", HP_AUX_W="0.0")

(WORK / "models").mkdir(exist_ok=True)
RESULTS = []


def run(tag, cfg, split, steps, seed=1337):
    env = dict(BASE_ENV, NFL_SPLIT=split, STEP_BUDGET=str(steps), SEED=str(seed),
               EVAL_EVERY_SEC="1200")
    env.update({k: str(v) for k, v in cfg.items()})
    t = time.time()
    r = subprocess.run([sys.executable, "train.py"], env=env,
                       capture_output=True, text=True)
    out = r.stdout
    val = None
    for line in out.splitlines():
        if line.startswith("val_rmse:"):
            val = float(line.split()[-1])
    print(f"[{elapsed():6.0f}s] {tag}: val={val} ({(time.time()-t)/60:.0f} min)", flush=True)
    if val is None:
        print("---- tail ----\n" + "\n".join((out + r.stderr).splitlines()[-15:]), flush=True)
    else:
        RESULTS.append({"tag": tag, "split": split, "steps": steps, "val_rmse": val})
        (WORK / "models/RESULTS.json").write_text(json.dumps(RESULTS, indent=2))
    return val


# ---------------- stage 1: within-season shift screen (opt-in) ----------------
# already run once (shift_lean=0.490, shift_1stplace=0.641); skip by default
if os.environ.get("RUN_SHIFT") == "1":
    SHIFT_STEPS = 12000
    run("shift_lean", LEAN, "weeks:14-18", SHIFT_STEPS)
    if (WORK / "models/last.pt").exists():
        shutil.move(WORK / "models/last.pt", WORK / "models/shift_lean.pt")
    run("shift_1stplace", FIRST_PLACE, "weeks:14-18", SHIFT_STEPS)
    if (WORK / "models/last.pt").exists():
        shutil.move(WORK / "models/last.pt", WORK / "models/shift_1stplace.pt")

# ---------------- stage 2: k-fold CV at 1st-place scale (opt-in) ----------------
# kf0/kf1/kf2 already trained (0.4685/0.4851/0.4905); enable with RUN_FOLDS=1
if os.environ.get("RUN_FOLDS") == "1":
    FOLD_STEPS = int(os.environ.get("FOLD_STEPS", 37000))
    FOLD_START = int(os.environ.get("FOLD_START", 3))
    FOLD_END = int(os.environ.get("FOLD_END", 5))
    FOLD_HEADROOM = float(os.environ.get("FOLD_HEADROOM", 20500))
    for f in range(FOLD_START, FOLD_END):
        if elapsed() > MAX_SEC - FOLD_HEADROOM:
            print(f"[{elapsed():6.0f}s] time guard: stopping before fold {f}", flush=True)
            break
        val = run(f"kfold_{f}", FIRST_PLACE, f"gamefold:5:{f}", FOLD_STEPS, seed=1000 + f)
        if val is not None and (WORK / "models/last.pt").exists():
            shutil.move(WORK / "models/last.pt", WORK / f"models/kf{f}.pt")

# ---------------- stage 3 (round 4): OOD-lever A/B at full scale (opt-in) ----------------
if os.environ.get("RUN_R4AB") == "1":
    AUG_PP = dict(FIRST_PLACE, HP_AUG_ROT_P="1.0", HP_AUG_SHIFT_P="0.9",
                  HP_AUG_XBIAS_STD="2.0", HP_AUG_YBIAS_STD="3.0", HP_AUG_NOISE="1.0")
    DIRECT0 = dict(FIRST_PLACE, HP_DECODE="direct")
    R4_STEPS = int(os.environ.get("R4_STEPS", 31000))
    R4_HEADROOM = R4_STEPS * 0.56 + 900
    for tag, cfg in (("aug2_f0", AUG_PP), ("direct_f0", DIRECT0)):
        if elapsed() > MAX_SEC - R4_HEADROOM:
            print(f"[{elapsed():6.0f}s] time guard: stopping before {tag}", flush=True)
            break
        val = run(tag, cfg, "gamefold:5:0", R4_STEPS, seed=1000)
        if val is not None and (WORK / "models/last.pt").exists():
            shutil.move(WORK / "models/last.pt", WORK / f"models/{tag}.pt")

# ---------------- stage 4 (round 5): DIRECT-decode fold set (default on) ----------------
# direct_f0 (fold 0, DECODE=direct) = 0.4614 CV, the true 1st-place head. Train
# the remaining direct folds for a decode-homogeneous, fold-diverse set to blend
# with the cumsum folds. 31k steps each ~= 4.8h on T4; two fit one session.
if os.environ.get("RUN_DIRECTFOLDS", "1") == "1":
    DIRECT = dict(FIRST_PLACE, HP_DECODE="direct")
    DF_STEPS = int(os.environ.get("DF_STEPS", 31000))
    DF_START = int(os.environ.get("DF_START", 1))
    DF_END = int(os.environ.get("DF_END", 5))
    DF_HEADROOM = float(os.environ.get("DF_HEADROOM", 18500))  # ~5.1h/fold
    for f in range(DF_START, DF_END):
        if elapsed() > MAX_SEC - DF_HEADROOM:
            print(f"[{elapsed():6.0f}s] time guard: stopping before direct fold {f}", flush=True)
            break
        val = run(f"df{f}", DIRECT, f"gamefold:5:{f}", DF_STEPS, seed=2000 + f)
        if val is not None and (WORK / "models/last.pt").exists():
            shutil.move(WORK / "models/last.pt", WORK / f"models/df{f}.pt")

# ---------------- stage 5 (round 5): FULL-DATA final models (opt-in) ----------------
# The biggest untried lever: fold models each drop 20% of data (train on 80%);
# the 1st place's *final* models train on 100% and take EMA weights (train.py's
# no-val path). Each full-data model should beat its 80% fold sibling by ~0.005-
# 0.008. NFL_SPLIT=none => whole dataset, no holdout. Both decodes, 2 seeds.
if os.environ.get("RUN_FULLDATA") == "1":
    FD_STEPS = int(os.environ.get("FD_STEPS", 37000))
    FD_HEADROOM = float(os.environ.get("FD_HEADROOM", 20500))
    FD_SET = [
        ("fd_cumsum_s3", dict(FIRST_PLACE), 3003),
        ("fd_direct_s3", dict(FIRST_PLACE, HP_DECODE="direct"), 3013),
        ("fd_cumsum_s4", dict(FIRST_PLACE), 3004),
        ("fd_direct_s4", dict(FIRST_PLACE, HP_DECODE="direct"), 3014),
    ]
    for tag, cfg, seed in FD_SET:
        if elapsed() > MAX_SEC - FD_HEADROOM:
            print(f"[{elapsed():6.0f}s] time guard: stopping before {tag}", flush=True)
            break
        run(tag, cfg, "none", FD_STEPS, seed=seed)   # val=-1 (no holdout)
        if (WORK / "models/last.pt").exists():
            shutil.move(WORK / "models/last.pt", WORK / f"models/{tag}.pt")

# ---------------- cleanup: keep only models + logs in the output ----------------
shutil.rmtree(WORK / "data", ignore_errors=True)
for junk in ("prepare.py", "train.py"):
    (WORK / junk).unlink(missing_ok=True)
shutil.rmtree(WORK / "__pycache__", ignore_errors=True)
print(f"[{elapsed():6.0f}s] DONE. outputs: "
      f"{sorted(p.name for p in (WORK/'models').iterdir())}", flush=True)
print(json.dumps(RESULTS, indent=2))

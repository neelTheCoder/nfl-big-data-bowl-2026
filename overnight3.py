"""
Round-3 unattended driver: scale up + grow the ensemble.

Round-2 evidence (results.tsv):
  * budget is nearly tapped: r2_best_f0 hit 34,465 steps @7200s ~= 1st-place
    per-model step count, scoring 0.4837
  * MULT=32 (r2_size_f1) scored 0.4801 with only 24,799 steps -> the bigger
    model wins at long budget; short screens were throughput-limited
  * LR=5e-4 gave a real hint at 2400s (0.5042 vs 0.5088) below adopt threshold
  => remaining path to 0.4634 is model SCALE + more ensemble members, toward
     the 1st-place full config (MULT 64, DIM1 256, 3 layers, LR 5e-4).

Plan:
  A. screen scaled configs at SCREEN_BUDGET (measure steps, since scale costs
     throughput on MPS); pick the best scale-up as the round-3 base
  B. train N_FINALS members at FINAL_BUDGET on distinct folds/seeds -> r3_*.pt
  C. ens_eval over r2_* + r3_* (round-2 winners stay in the pool so we can
     only improve) -> greedy subset -> models/ens_*.pt

Run:  cd ~/Desktop/nfl && nohup .venv/bin/python overnight3.py > overnight3.out 2>&1 &
"""
import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = str(ROOT / ".venv/bin/python")
SCREEN_BUDGET = 3000
FINAL_BUDGET = 7200
MAX_HOURS = float(os.environ.get("MAX_HOURS", 13.0))
NOISE = 0.008
SEED = 1337
T0 = time.time()

# round-2 champion single = MULT 32 (0.4801). Screen bigger scale-ups from here.
BASE = dict(MULT=32, DIM1=192, N_LAYERS=2, N_HEADS=8, FF_DIM=768,
            LR=1e-3, WEIGHT_DECAY=1e-4, EMA_DECAY=0.9995, T_USED=20,
            AUG_SHIFT_P=0.675, AUG_ROT_P=0.5, BATCH=64, AUX_W=0.0)

# scale-up candidates (label, overrides on BASE), toward 1st-place full config
SCALE_TRIALS = [
    ("wide_deep_lr", {"DIM1": 256, "N_LAYERS": 3, "LR": 5e-4}),
    ("mult48_wide",  {"MULT": 48, "DIM1": 256, "N_LAYERS": 3, "LR": 5e-4}),
    ("mult32_lr",    {"LR": 5e-4}),
]

# final ensemble members: (tag, extra-overrides on winning base, seed, fold)
N_FINALS = [
    ("r3_a", {}, 1337, 3),
    ("r3_b", {}, 2024, 4),
    ("r3_c", {"MULT": 48, "DIM1": 256, "N_LAYERS": 3, "LR": 5e-4}, 7, 5),
    ("r3_d", {}, 42, 6),
]

RESULTS = ROOT / "results.tsv"
LOG = ROOT / "overnight_log.tsv"
SCREEN_WALL = SCREEN_BUDGET + 500
FINAL_WALL = FINAL_BUDGET + 1000


def sh(cmd, env=None):
    e = dict(os.environ)
    if env:
        e.update({k: str(v) for k, v in env.items()})
    return subprocess.run(cmd, shell=True, cwd=ROOT, env=e,
                          capture_output=True, text=True)


def run_trial(cfg, budget, seed=SEED, fold=-1, eval_every=900):
    env = {f"HP_{k}": v for k, v in cfg.items()}
    env.update(TIME_BUDGET=budget, SEED=seed, HP_FOLD=fold, EVAL_EVERY_SEC=eval_every)
    sh(f"{PY} train.py > run.log 2>&1", env)
    log = (ROOT / "run.log").read_text()
    m = re.search(r"^val_rmse:\s*([\d.]+)", log, re.M)
    if not m:
        return None, ".", ".", "\n".join(log.splitlines()[-6:])
    sm = re.search(r"^num_steps:\s*(\d+)", log, re.M)
    pm = re.search(r"^num_params_M:\s*([\d.]+)", log, re.M)
    return (float(m.group(1)), sm.group(1) if sm else ".",
            pm.group(1) if pm else ".", None)


def short_commit():
    return sh("git rev-parse --short HEAD").stdout.strip()


def logrow(path, cols):
    with open(path, "a") as f:
        f.write("\t".join(str(c) for c in cols) + "\n")


def main():
    print(f"[r3] start {time.strftime('%H:%M')} base={BASE}", flush=True)
    sh("rm -f models/r3_*.pt")

    # ---------- Phase A: scale-up screen ----------
    cfg = dict(BASE)
    best, steps, params, err = run_trial(cfg, SCREEN_BUDGET)
    if best is None:
        print(f"[r3] BASELINE CRASHED:\n{err}", flush=True)
        return
    print(f"[r3] TRIAL 0 base(MULT32)@{SCREEN_BUDGET}s -> {best:.4f} ({steps} steps, {params}M)", flush=True)
    logrow(RESULTS, [short_commit(), SCREEN_BUDGET, f"{best:.6f}", steps, params,
                     "screen-keep", "r3 base MULT32"])

    for i, (label, over) in enumerate(SCALE_TRIALS, 1):
        if time.time() - T0 + SCREEN_WALL + len(N_FINALS) * FINAL_WALL + 1500 > MAX_HOURS * 3600:
            print(f"[r3] time guard: stop screening at {label}", flush=True)
            break
        trycfg = dict(BASE); trycfg.update(over)
        score, steps, params, err = run_trial(trycfg, SCREEN_BUDGET)
        if score is None:
            print(f"[r3] TRIAL {i} {label} CRASH:\n{err}", flush=True)
            logrow(RESULTS, [short_commit(), SCREEN_BUDGET, "99", ".", ".", "crash", f"r3 {label}"])
            continue
        adopt = score < best - NOISE
        if adopt:
            cfg, best = trycfg, score
        print(f"[r3] TRIAL {i} {label} -> {score:.4f} {'ADOPT' if adopt else 'keep'} "
              f"best:{best:.4f} ({steps} steps, {params}M)", flush=True)
        logrow(RESULTS, [short_commit(), SCREEN_BUDGET, f"{score:.6f}", steps, params,
                         "screen-keep" if adopt else "screen-discard", f"r3 {label}"])
    (ROOT / "best_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"[r3] phase A done. base for finals={cfg}", flush=True)

    # ---------- Phase B: ensemble members ----------
    for tag, over, seed, fold in N_FINALS:
        if time.time() - T0 + FINAL_WALL + 1200 > MAX_HOURS * 3600:
            print(f"[r3] time guard: skip final {tag}", flush=True)
            continue
        mcfg = dict(cfg); mcfg.update(over)
        score, steps, params, err = run_trial(mcfg, FINAL_BUDGET, seed=seed, fold=fold)
        if score is None:
            print(f"[r3] FINAL {tag} CRASH:\n{err}", flush=True)
            logrow(RESULTS, [short_commit(), FINAL_BUDGET, "99", ".", ".", "crash", f"r3 final {tag}"])
            continue
        sh(f"cp models/last.pt models/{tag}.pt")
        print(f"[r3] FINAL {tag} seed={seed} fold={fold} -> {score:.4f} ({steps} steps, {params}M)", flush=True)
        logrow(RESULTS, [short_commit(), FINAL_BUDGET, f"{score:.6f}", steps, params,
                         "confirm-keep", f"r3 final {tag} {json.dumps(mcfg)}"])

    # ---------- Phase C: ensemble over round-2 winners + round-3 members ----------
    r = sh(f"ENS_PATTERNS='models/r2_*.pt models/r3_*.pt' {PY} ens_eval.py")
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print(f"[r3] ens_eval FAILED:\n{r.stderr[-2000:]}", flush=True)
    m = re.search(r"^ens_best_rmse:\s*([\d.]+)", r.stdout, re.M)
    if m:
        logrow(RESULTS, [short_commit(), FINAL_BUDGET, m.group(1), ".", ".",
                         "confirm-keep", "r3 ENSEMBLE greedy subset + hflip TTA"])
    print(f"[r3] DONE {(time.time() - T0) / 60:.0f} min total", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print("[r3] CRASHED\n" + traceback.format_exc(), flush=True)

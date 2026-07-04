"""
Round-2 unattended overnight driver.

Research basis (reference/nfl2026-1st-place-train.py, re-read 2026-07-01):
  * their "35 epochs" hides train_data_repeat_factor=6 -> ~37k steps/model;
    our 1800s runs are ~7.8k steps, so budget is still the dominant gap
  * their final ensemble is 5 GroupKFold folds x 3 fold-seeds = 15 models --
    DATA diversity, not seed diversity (our 3-seed gain was only 0.0016)
  * AUX_W=0.8 / LR=5e-4 / model scale were tuned at the 37k-step regime;
    we only ever screened them at ~1.5k-step budgets

Plan:
  A. re-screen budget-dependent knobs (AUX_W, LR, MULT) at SCREEN_BUDGET,
     greedy adoption (> NOISE improvement), all at SEED, FOLD=-1
  B. train 3 diverse finals at FINAL_BUDGET: best cfg + a size-flipped and an
     aux-flipped variant, each on its own data fold (drop 1/8) and seed
  C. ens_eval.py: pool = round-1 checkpoints (r1_*) + tonight's (r2_*),
     greedy subset selection on val, write winners to models/ens_*.pt

Run:  cd ~/Desktop/nfl && nohup .venv/bin/python overnight.py > overnight2.out 2>&1 &
"""
import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = str(ROOT / ".venv/bin/python")
SCREEN_BUDGET = 2400
FINAL_BUDGET = 7200
MAX_HOURS = float(os.environ.get("MAX_HOURS", 11.5))
NOISE = 0.010
SEED = 1337
T0 = time.time()

# champion from round 1 (train.py editable block; best_config.json is stale)
BASE = dict(MULT=16, DIM1=192, N_LAYERS=2, N_HEADS=8, FF_DIM=768,
            LR=1e-3, WEIGHT_DECAY=1e-4, EMA_DECAY=0.9995, T_USED=20,
            AUG_SHIFT_P=0.675, AUG_ROT_P=0.5, BATCH=64, AUX_W=0.0)

# budget-dependent knobs only; each entry is (label, {overrides})
TRIALS = [
    ("AUX_W=0.4", {"AUX_W": 0.4}),
    ("AUX_W=0.8", {"AUX_W": 0.8}),
    ("LR=5e-4",  {"LR": 5e-4}),
    ("MULT=32",  {"MULT": 32}),
]

LOG = ROOT / "overnight_log.tsv"
RESULTS = ROOT / "results.tsv"

# rough wall-clock cost of a run = train budget + interim-eval overhead
SCREEN_WALL = SCREEN_BUDGET + 420
FINAL_WALL = FINAL_BUDGET + 900
PHASE_BC_WALL = 3 * FINAL_WALL + 1500


def sh(cmd, env=None):
    e = dict(os.environ)
    if env:
        e.update({k: str(v) for k, v in env.items()})
    return subprocess.run(cmd, shell=True, cwd=ROOT, env=e,
                          capture_output=True, text=True)


def run_trial(cfg, budget, seed=SEED, fold=-1, eval_every=600):
    env = {f"HP_{k}": v for k, v in cfg.items()}
    env.update(TIME_BUDGET=budget, SEED=seed, HP_FOLD=fold, EVAL_EVERY_SEC=eval_every)
    sh(f"{PY} train.py > run.log 2>&1", env)
    log = (ROOT / "run.log").read_text()
    m = re.search(r"^val_rmse:\s*([\d.]+)", log, re.M)
    if not m:
        return None, ".", ".", "\n".join(log.splitlines()[-5:])
    sm = re.search(r"^num_steps:\s*(\d+)", log, re.M)
    pm = re.search(r"^num_params_M:\s*([\d.]+)", log, re.M)
    return (float(m.group(1)), sm.group(1) if sm else ".",
            pm.group(1) if pm else ".", None)


def short_commit():
    return sh("git rev-parse --short HEAD").stdout.strip()


def logrow(path, cols):
    with open(path, "a") as f:
        f.write("\t".join(str(c) for c in cols) + "\n")


def note(trial, label, score, adopted, cfg, steps=".", params=".", budget=SCREEN_BUDGET):
    txt = "crash" if score is None else f"{score:.4f}"
    print(f"[overnight] TRIAL {trial} {label} -> {txt} {adopted}", flush=True)
    logrow(LOG, [time.strftime('%H:%M'), trial, label, "-", txt, adopted, json.dumps(cfg)])
    logrow(RESULTS, [short_commit(), budget, f"{score:.6f}" if score else "99",
                     steps, params,
                     "screen-keep" if adopted == "ADOPT" else "screen-discard",
                     f"r2 {label}"])


def main():
    print(f"[overnight] r2 start {time.strftime('%H:%M')} base={BASE}", flush=True)
    # keep round-1 ensemble checkpoints in the selection pool as r1_*
    for s in (1337, 2024, 7):
        if (ROOT / f"models/ens_{s}.pt").exists():
            sh(f"cp models/ens_{s}.pt models/r1_{s}.pt")
    sh("rm -f models/r2_*.pt")

    # ---------- Phase A: long-budget re-screen ----------
    cfg = dict(BASE)
    best, steps, params, err = run_trial(cfg, SCREEN_BUDGET)
    if best is None:
        print(f"[overnight] BASELINE CRASHED:\n{err}", flush=True)
        return
    note(0, "baseline@2400s", best, "base", cfg, steps, params)
    scores = {"baseline": best}

    for i, (label, over) in enumerate(TRIALS, 1):
        if time.time() - T0 + SCREEN_WALL + PHASE_BC_WALL > MAX_HOURS * 3600:
            print(f"[overnight] time guard: skipping remaining screens at trial {i}", flush=True)
            break
        trycfg = dict(cfg); trycfg.update(over)
        score, steps, params, err = run_trial(trycfg, SCREEN_BUDGET)
        scores[label] = score
        if score is None:
            note(i, label, None, "no", trycfg)
            print(err, flush=True)
            continue
        adopt = score < best - NOISE
        if adopt:
            cfg, best = trycfg, score
        note(i, label, score, "ADOPT" if adopt else "keep", cfg, steps, params)
    (ROOT / "best_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"[overnight] phase A done best={best:.4f} cfg={cfg}", flush=True)

    # ---------- Phase B: 3 diverse finals ----------
    size_flip = dict(cfg, MULT=32 if cfg["MULT"] == 16 else 16)
    aux_vals = [v for v in (0.0, 0.4, 0.8) if v != cfg["AUX_W"]]
    aux_vals.sort(key=lambda v: scores.get(f"AUX_W={v}", scores["baseline"]) or 99)
    aux_flip = dict(cfg, AUX_W=aux_vals[0])
    members = [
        ("r2_best_f0", cfg, 1337, 0),
        ("r2_size_f1", size_flip, 2024, 1),
        ("r2_aux_f2", aux_flip, 7, 2),
    ]
    for tag, mcfg, seed, fold in members:
        score, steps, params, err = run_trial(mcfg, FINAL_BUDGET, seed=seed,
                                              fold=fold, eval_every=900)
        if score is None:
            print(f"[overnight] FINAL {tag} CRASH:\n{err}", flush=True)
            logrow(RESULTS, [short_commit(), FINAL_BUDGET, "99", ".", ".",
                             "crash", f"r2 final {tag}"])
            continue
        sh(f"cp models/last.pt models/{tag}.pt")
        print(f"[overnight] FINAL {tag} seed={seed} fold={fold} -> {score:.4f}", flush=True)
        logrow(RESULTS, [short_commit(), FINAL_BUDGET, f"{score:.6f}", steps, params,
                         "confirm-keep", f"r2 final {tag} {json.dumps(mcfg)}"])

    # ---------- Phase C: ensemble selection ----------
    r = sh(f"{PY} ens_eval.py")
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print(f"[overnight] ens_eval FAILED:\n{r.stderr[-2000:]}", flush=True)
    m = re.search(r"^ens_best_rmse:\s*([\d.]+)", r.stdout, re.M)
    if m:
        logrow(RESULTS, [short_commit(), FINAL_BUDGET, m.group(1), ".", ".",
                         "confirm-keep", "r2 ENSEMBLE greedy subset + hflip TTA"])
    print(f"[overnight] DONE {(time.time() - T0) / 60:.0f} min total", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print("[overnight] CRASHED\n" + traceback.format_exc(), flush=True)

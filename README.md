# NFL Big Data Bowl 2026 — Prediction, via autoresearch

Predicting where NFL players move while the ball is in the air. Given the
tracking frames up to the throw, predict each targeted player's `(x, y)` for
every frame until the ball lands. Scored by RMSE over all predicted coordinates.

**Best late submission: private RMSE 0.47458** — under the competition's
20th-place cutoff (0.50460); 1st place was 0.46340.

The whole project is a Karpathy-style **autoresearch** loop: model development
run as an experiment loop with a frozen evaluator, one editable training file,
and a fixed wall-clock budget per run, so results are comparable and the loop is
restartable from disk.

## Layout
- `prepare.py` — data preprocess + **frozen** evaluation (competition RMSE) and the validation splits. Never edited during a run.
- `train.py` — the one file the loop edits: model, features, augmentation, loss, decode. Hyperparameters overridable via `HP_*` env vars.
- `ens_eval.py` — ensemble scorer: solo scores + greedy forward selection over checkpoints.
- `kaggle_submit.py` — drives a Kaggle late submission end-to-end over the REST API (dataset → notebook push → run → `submit-notebook` → poll score).
- `kaggle_pkg/notebook/nfl_bdb26_submit.py` — inference notebook run on Kaggle's evaluation API.
- `kaggle_pkg/train_kernel/nfl_bdb26_train.py` — server-side GPU training kernel (runs on Kaggle T4s, no laptop needed).
- `results.tsv` — every tracked experiment: commit, budget, val RMSE, steps, params, verdict.
- `log.md` — the research journal (hypothesis → result → lesson).
- `REPORT.md` / `one_pager.tex` — the write-ups.

Model weights, competition data, and the pulled reference notebooks are
git-ignored (the last for copyright).

## The model in one paragraph
Each of the 22 players is encoded from its last 20 tracking frames (position,
speed, acceleration, direction and orientation as sine/cosine, plus vectors to
the ball's landing spot and the targeted receiver) by a **depthwise 1-D CNN**; a
small stem embeds the static per-play features; a **pre-norm SwiGLU Transformer**
then lets the 22 player tokens attend to one another; a convolutional decoder
emits a per-frame displacement and its variance out to 48 frames. Trained with
Gaussian negative-log-likelihood plus velocity/acceleration auxiliary heads,
heavy geometric augmentation (uniform rotation, mirror, early-start), EMA weight
averaging, and horizontal-flip test-time augmentation. The submission is a 4-way
ensemble, ~4.9M parameters per member.

## Setup
```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install kagglehub torch pandas polars pyarrow
# put a Kaggle API token in ~/.kaggle/ and accept the competition rules
.venv/bin/python prepare.py                  # preprocess + CV baseline
STEP_BUDGET=37000 .venv/bin/python train.py  # train one model
```

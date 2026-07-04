# Kaggle late submission — 0.4784 ensemble

Getting the private-leaderboard score requires running our inference notebook on
Kaggle's servers (the hidden test set is not downloadable). Late submissions on
this competition are still scored on the private test set — they just don't
enter the official (frozen) leaderboard; Kaggle shows you a personal
late-submission board of where you'd have ranked.

## Contents
- `model_ds/` — the Kaggle **dataset** payload: `prepare.py`, `train.py`,
  `models/ens_00.pt` (MULT32) + `models/ens_01.pt` (MULT16) = the 0.4784 ensemble.
- `notebook/nfl_bdb26_submit.py` — the submission **script kernel** (uses the
  competition evaluation API; `serve()` on the competition rerun, local gateway
  otherwise). `kernel-metadata.json` — its metadata.

## Automated path (what we ran)
`python kaggle_submit.py` — drives the whole flow with the bearer token in
`~/.kaggle/access_token`:
1. upload `model_ds/` as dataset `neel999/nfl-bdb26-models`
2. push kernel `neel999/nfl-bdb26-submit`
3. wait for its run to finish
4. `submit-notebook` the kernel version to the competition
5. poll our submissions list for the public/private score

Resumable per stage: `STAGE=2 python kaggle_submit.py`, etc.

## Manual fallback (browser — reliable if the account needs phone verification)
1. **Dataset**: kaggle.com → Datasets → New Dataset → upload the four files in
   `model_ds/` (keep `models/` as a folder) → title "NFL BDB26 Models".
2. **Notebook**: on the competition page → Code → New Notebook → **File → Import**
   `notebook/nfl_bdb26_submit.py` (or paste it). Add data: the competition +
   your `nfl-bdb26-models` dataset. Settings: Internet **off**, GPU off.
3. **Run All** — it should print `loaded 2 checkpoints` and produce
   `submission.parquet` against the example test.
4. **Submit to Competition** (top-right) → wait for scoring → read the
   Private Score on **My Submissions**.

Local sanity check already passed: the same code, via the local
`kaggle_evaluation` gateway, produces a valid 5,837-row submission
(x∈[2.0,118.4], y∈[1.2,52.8]); holdout val RMSE = 0.4784.

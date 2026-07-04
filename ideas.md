# Idea backlog (ordered by expected value ÷ cost)

## From reading the ACTUAL 1st-place code (reference/), 2026-07-01
- Baseline v2 is already their recipe scaled down (MULT 32 vs 64, DIM1 192 vs
  256, FF 768 vs 1024). Try full size in a confirm budget — fewer steps but
  their exact capacity. Their single-config 5-fold ens = 0.46471 private.
- AUX_W=0.8 assumes 210-epoch training; for short budgets the traj term may
  need more weight early — try AUX_W 0.5, or anneal 0.5→0.8.
- We get ~28 epochs in a confirm run vs their 210. Highest-value lever may be
  step speed: MULT=16, BATCH=128 (amortize MPS overhead), fewer head blocks.
- Pass-aligned rot_x/rot_y features exist in their code but their BEST config
  excludes them (0.46471 vs 0.46946 with) — do not add.
- They pick best epoch by val; we pick best of {raw, EMA} at run end. Could
  add cheap best-of-k checkpoint selection at 80/90/100% of budget.
- torch.compile helped them on CUDA; on MPS usually a wash — low priority.

Sources: 1st/2nd/3rd place writeups BDB2026-Prediction + trajectory-forecasting
literature. Baseline already includes: conv dynamic encoder, player
transformer w/ SwiGLU, displacement-from-last-frame decoding, GaussianNLL +
vel/acc aux, rotation + h-flip + early-start augmentation, EMA.

## High priority
1. Predict per-frame deltas + cumsum instead of displacement-from-last-frame
   (2nd place found this clearly better).
2. GRU/LSTM temporal encoder instead of conv stack (2nd place: RNN > conv/transformer
   for the time axis on this task).
3. Constant-velocity residual: decode residuals on top of CV extrapolation
   (physics prior; strong at long horizons, untried by top teams).
4. Aux losses as GaussianNLL instead of SmoothL1 (1st place).
5. LR/schedule/batch sweep: BATCH∈{32,64,128}, LR∈{1e-4..1e-3}; Muon or RAdam.
6. Wider model within budget: D_MODEL 256, layers 2 (3rd place: wide-shallow > narrow-deep).

## Medium
6b. Geometric endpoint prior features (from GNN notebook LB .586): per player,
    deterministic endpoint guess = x + vx*(n_out/10), y + vy*(n_out/10);
    receiver endpoint = ball landing. Feed (endpoint - last_xy) as 2 static
    features. Encodes n_out×velocity interaction the current features lack.
7. Time embedding in decoder (learned per-output-frame embedding added before convs).
8. Loss weighting: time-decay e^(-0.03 t) (2nd/3rd place TemporalHuber) vs flat NLL.
9. More dynamic features: angle_to_ball, distance_to_ball_land, time_elapsed.
10. TTA: h-flip average at eval (3rd place: +small but free at inference).
11. Feed 30 input frames instead of 20 (T_USED).
12. EMA decay sweep 0.995–0.9995; eval both EMA and raw, keep better.
13. Snapshot ensemble within one budget (avg predictions of last k checkpoints).
14. Train final model on train+val for submission after architecture freeze.

## Low / later
15. 2021/2018 BDB tracking data pretraining (2nd/3rd place used it; big lift: download + relabel).
16. Player dropout + random slot shuffle augmentation (3rd place boost).
17. Random input crop augmentation (3rd place).
18. Multi-seed ensembling for the submission notebook (always helps; do at the end).
19. Distance-based attention bias (geometric prior in the player transformer).

## DO NOT RETRY (evidence from top teams)
- Dropout anywhere (3rd place: any dropout hurts pure regression).
- Small-angle-only rotation (uniform 0–360 is better — 1st place).
- Per-step noise, temporal mask + linear fill (3rd place: no help).
- Complex geometric features beyond the basics (2nd/3rd: no help).
- Re-ordering player slots by distance-to-receiver (3rd: no help).
- Narrow-deep transformers (3rd: wide-shallow wins).
- Predicting direction+speed instead of xy deltas (2nd: worse).

# Experiment journal (append-only)

## 2026-07-01 setup
- Scaffold built; smoke test on synthetic data: 60s budget, 710 steps, 1.38M
  params, val_rmse 13.25 vs CV baseline 15.66. Pipeline + independent rescore
  verified on MPS.
- Waiting on Kaggle credentials for real data. Real baseline runs (SEED=1337,
  SEED=2024) are the first two experiments once data lands.

## 2026-07-01 data + reference code landed
- Real data via kagglehub (access token). 14,103 plays; my generic drop rules
  caught exactly the 5 plays 1st place hardcodes — preprocessing validated.
- Constant-velocity baseline: val_rmse 1.6453 (weeks 17-18 holdout).
- Pulled ACTUAL 1st-place notebook code via Kaggle API (reference/). Rewrote
  baseline as faithful scaled clone: AUX_W=0.8 (aux dominates!), z-scored
  features/targets, constant LR 5e-4 RAdam, xy-bias aug, EMA 0.9995.
- MPS profile: step 272ms at BATCH=64 (grouped-conv stem heaviest; full conv
  4x slower — keep grouped). make_batch 9ms — not the bottleneck.
- EMA fix: budget-scaled decay + final best-of{raw,EMA} selection.

## exp0a: baseline v2, seed 1337, screen 420s
- Hypothesis: scaled 1st-place clone lands well under CV baseline.
- Result: val_rmse 0.9445 (raw), EMA 1.1792 (lags at short budget), 1360
  steps ~= 6.9 epochs. Independent rescore matches (0.94445).
- Lesson: raw >> EMA this short; their 0.47 CV needed ~210 epochs — our gap
  is training length, not architecture. Step-efficiency ideas first.

## exp0b: baseline v2, seed 2024, screen 420s
- Result: val_rmse 1.1056 (raw), EMA 1.3811. vs seed 1337: 0.9445.
- Lesson: run-to-run sigma ~0.11 at 420s with raw-final weights — the ±0.005
  noise band was fantasy. Final raw weights are a noisy estimator. Fix first,
  then re-baseline: tail-averaged weights (SWA over last 25%, every 25 steps)
  + best-of{raw, ema, swa} selection = exp1.

## submission path validated (2026-07-01)
- Local kaggle_evaluation gateway run end-to-end with a throwaway checkpoint:
  submission.parquet 5837 rows, id/x/y, sane ranges, exit 0.
  (needed sys.path.insert of the competition dir for kaggle_evaluation.)

## exp1: SWA tail-average + best-of{raw,ema,swa}, seed 1337, 420s
- Hypothesis: final-raw-weights noise is a big chunk of sigma~0.11; averaging
  the tail stabilizes and lifts.
- Result: swa 0.9492 < raw 0.9607 < ema 1.2445; best=swa selected.
- Note: same-seed rerun gave raw 0.9607 vs exp0a's 0.9445 — MPS kernels are
  nondeterministic, so seeds alone don't reproduce; sigma includes backend
  noise. Awaiting seed-2024 pair for new sigma estimate.

## exp1b: SWA, seed 2024, 420s
- Result: raw 1.1675, swa 1.1716 (best raw). Seed 2024 consistently ~0.2
  worse than 1337 across BOTH its runs -> systematic init/data-order effect
  at short budgets, not run noise. Same-seed spread is only ~0.015.
- Lesson: screen same-seed (1337) with +-0.03 band; use seed diversity for
  the final ensemble instead. SWA kept (within-run win on 1337: 0.9492 vs
  0.9607 raw; harmless on 2024).
- Current sigma estimate (same-seed): ~0.015-0.02.

## exp2: BATCH=128, seed 1337, 420s — DISCARD
- Result: 1.2740 (vs 0.9449 best). 128-batch gives ~195 samples/s vs 208 at
  64, so fewer optimizer steps and nothing in return. MPS conv throughput is
  already saturated at batch 64.
- Lesson: on M2, more steps > bigger batches. Try cheaper per-step next
  (MULT=16), not bigger batches.

## exp3: MULT=16, seed 1337, 420s — first run 0.9215 (swa)
- Hypothesis: halving stem width buys more steps at little capacity cost.
- Result: 0.9215 swa (raw 0.9567); ~15% more steps (1471 vs ~1330). Better
  than best 0.9449/0.9492 by 0.023-0.028 -> inside +-0.03 band; rerunning
  same seed per protocol.

## exp3 rerun: 0.9221 — KEEP MULT=16
- Mean 0.9218 vs 0.9469 for MULT=32; consistent across all 4 runs.
- SWA same-seed spread now ~0.003: estimator is stable. New sigma ~0.005;
  tightening informal read of the band, keeping contract at +-0.03 for now.

## exp4: AUX_W=0.5, seed 1337, 420s — 0.7283 (swa) KEEP
- Hypothesis: aux-dominant loss (0.8) is tuned for 210 epochs; short budgets
  need direct trajectory gradient.
- Result: 0.7283 vs 0.9218 — biggest single win so far (-0.19).
- Lesson: loss balance is budget-dependent; sweep further (0.2, 0.35) before
  confirming.

## exp5: AUX_W=0.2 — 0.6873 (raw) KEEP
- Sweep 0.8 -> 0.5 -> 0.2 keeps paying. Their 0.8 is a long-training artifact.
- raw beat swa this run (0.6873 vs 0.6927) — selection layer doing its job.

## exp7: cosine LR decay — 0.6986 DISCARD
- Worse than constant 5e-4 (0.6515) by 0.047. SWA already provides the
  "tail settling" cosine was meant to buy; annealing just slows learning.

## exp8: LR=1e-3 — 0.6313 (swa) KEEP
- Higher LR helps a lot at short budget (more effective progress/step).
- New champion config: MULT16, DIM1 192, 3 layers, AUX_W0, LR1e-3, SWA.

## overnight driver launched
- Added HP_* env overrides (post-editable block) so overnight.py can greedy-
  sweep configs. Driver: baseline -> coordinate descent (LR/MULT/DIM1/layers/
  FF/T_USED/heads/EMA/wd/aug/batch) -> confirm -> 3-seed ensemble into models/.

## overnight run (relaunched 20:59 with steps/params logging)
- Baseline 0.6217 (1825 steps, 2.33M). LR sweep: 7e-4/1.5e-3/2e-3 all within
  noise of 1e-3 -> plateau, keep 1e-3. MULT 24/32 clearly worse (stem cost
  kills step count), 12 within noise -> keep 16. DIM1 256 worse.
- Driver continues: DIM1 128, layers, FF, T_USED, heads, EMA, wd, aug, batch;
  then 900s confirm; then 3-seed x 1800s ensemble -> models/ens_*.pt.

## exp11: geometric endpoint prior features — 0.6048 DISCARD
- Within noise of champion 0.6019; simplicity criterion says no. The model
  already infers the endpoint from ball-land offsets + n_out + velocity.

## exp12: N_LAYERS=2 — 0.5971 (swa) ADOPT
- Within noise of 0.6019 but strictly simpler and faster (1.9M params,
  ~15% quicker steps): simplicity rule -> adopt. 3rd place wide-shallow
  intuition holds.
- Local val 0.597 vs the public GNN notebook's 0.586 LB — close but NOT yet
  under it (and val != LB anyway; late submission is the real check).

## exp13: h-flip TTA — 0.5903 (swa) KEEP
- Same trained model as exp12 (same seed); the delta is pure inference
  averaging. Free at submission time too.
- Champion config: MULT16/DIM1 192/2 layers/FF768, LR1e-3 const, AUX 0,
  cumsum decoding, SWA+selection, hflip TTA. Moving to 900s confirm.

## confirm @900s: 0.5534 (swa)
- Budget doubling 420->900s: 0.590->0.553. Under the GNN notebook's 0.586.
- Launching final phase: 3 seeds x 1800s -> models/ens_*.pt + ensemble eval.

## 2026-07-02 final ensemble
- Seeds @1800s: 1337->0.5156, 2024->0.5148, 7->0.5137 (SWA selected in all).
  Cross-seed spread collapsed from ~0.2 (420s) to ~0.002 (1800s).
- ENSEMBLE (3 seeds, hflip TTA): val_rmse 0.5121.
- Ensemble gain over best single is small (-0.0016): same-config seeds
  converge to near-identical solutions. Next lever = diversity (different
  MULT/DIM1/feature configs per member) and train+val retraining, not more
  seeds.
- Mid-run note: seed 1337 had a transient loss spike (interim 0.62->0.70 at
  t=1200s) that self-recovered; consider GRAD_CLIP 1.0 for long runs.
- Submission notebook: added hflip TTA to predict(); prefers models/ens_*.pt.

## 2026-07-02 01:19 — round 2 kickoff (overnight, fold-diverse)
- Re-read reference/nfl2026-1st-place-train.py: CFG has train_data_repeat_factor=6, so "35 epochs" = ~37k steps/model (~5x our 1800s runs); final ensemble = 5 GroupKFold folds x 3 fold-seeds = 15 models (data diversity, not seed diversity); full scale = MULT 64 / DIM1 256 / FF 1024 / 3 layers; AUX_W=0.8, LR=5e-4 tuned in that regime.
- Implication: our AUX_W=0.0 / LR=1e-3 / small-model adoptions were all screened at ~1.5k steps and may invert at long budgets. MULT=32 lost at 420s purely on steps (1282 vs 1643).
- train.py: added HP_FOLD (drop train idx%8==FOLD; val weeks 17-18 untouched), EVAL_EVERY_SEC env, +AUX_W/LR/FOLD in ckpt config. ens_eval.py added: solo scores + greedy forward selection, writes models/ens_*.pt. Verified: reproduces ens_1337 solo 0.515567 exactly; rejects a garbage 60s model.
- overnight.py rewritten: Phase A re-screen {AUX_W 0.4/0.8, LR 5e-4, MULT 32} @2400s; Phase B 3 finals @7200s = best cfg (seed 1337, fold 0) + size-flip (2024, fold 1) + aux-flip (7, fold 2); Phase C greedy ensemble over r1_* + r2_*. MAX_HOURS=11.5 guard. Launched pid 32613, commit 17e562a.

## 2026-07-02 10:47 — round 2 complete: 0.5121 -> 0.4784
- Phase A re-screen @2400s: baseline(AUX0,LR1e-3) 0.5088; AUX_W=0.4 0.5872; AUX_W=0.8 0.6498 (aux monotonically hurts even at ~9k steps — round-1 AUX_W=0 confirmed correct, not a short-budget artifact); LR=5e-4 0.5042 (hint -0.0046, below 0.01 adopt threshold); MULT=32 0.5130 (tied on fewer steps).
- Phase B finals @7200s on distinct folds: r2_best_f0 (MULT16,fold0) 0.4837 @34465 steps; r2_size_f1 (MULT32,fold1) 0.4801 @24799 steps; r2_aux_f2 (AUX0.4,fold2) 0.4866.
- KEY: MULT=32 beat MULT=16 at long budget with 10k FEWER steps => short screens were throughput-limited; bigger model wins with time. And best_f0 @34k steps ~= 1st-place per-model count => plain budget nearly tapped. Remaining lever = SCALE + more ensemble members.
- Phase C greedy ensemble (pool = r1_* + r2_*): selected r2_size_f1 + r2_best_f0 -> 0.4784. r1 models (0.51) and aux member dropped. models/ens_00.pt(MULT32)+ens_01.pt(MULT16) written; submission notebook globs ens_*.pt, loads per-ckpt config, gateway path proven.
- Round 3 launched (overnight3.py, commit 22021bc): screen scale-ups {wide_deep_lr DIM256/3L/LR5e-4, mult48_wide, mult32_lr} @3000s, then 3-4 finals @7200s on folds 3-6, greedy ensemble over r2_*+r3_* (can only improve on 0.4784). Target 0.4634, gap 0.015.

## 2026-07-02 13:27 — REAL Kaggle private score: 0.57915 (did NOT beat 1st 0.4634)
- Late-submitted the 0.4784 ensemble via kaggle_submit.py (bearer REST: dataset upload + kernel push v2 + submit-notebook form-encoded). Submission 54275816, status complete. public=0.0 (public split is degenerate/0 for everyone), private=0.57915 on full hidden test.
- Private LB top20 = 0.46340..0.50460; ours 0.57915 is 0.0746 below the 20th cutoff -> outside top 20. Full-field rank not exposed by API (view caps at 20; download gives only the all-zeros public LB).
- KEY LESSON: holdout->private gap ballooned 0.013 -> 0.116. Our holdout (weeks 17-18 of the 2023/24 TRAIN seasons) is IN-DISTRIBUTION; hidden test is the 2025 season (OOD). Lean 2-model ensemble overfit same-season holdout; top solutions win on season-shift robustness (heavy aug + 15-model fold ensemble). Next time: validate across a season boundary (train 2023 / val 2024), not held-out weeks of the same seasons.
- Gotchas solved: kaggle CLI import hangs ~130s + Basic-auth 401 (token is Bearer-only); Kaggle mounts inputs at /kaggle/input/datasets/<owner>/<slug> and /kaggle/input/competitions/<slug> (glob-discover, do not hardcode); submit-notebook endpoint needs FORM-encoded body (JSON -> 400 "requires an output FileName").

## 2026-07-02 14:00 — post-mortem plan: attack generalization, not holdout RMSE
- Verified: 5 submissions/day (maxDailySubmissions=5, submissionsDisabled=False). 1 used today. Multiple accounts = Kaggle ToS violation; not needed.
- Verified: train data is 2023 season ONLY (36 csvs); test = 2025 wks14-18 -> two-season shift. Top-20 (0.463-0.505) trained on the same 2023 data => the gap is recipe robustness (15-model fold ensembles, AUX 0.8, 37k steps, MULT 64), not data access.
- Decision: distrust weeks-17/18-tuned choices; adopt the 1st-place recipe wholesale (only config with verified OOD performance). Round-3-as-designed (scale-up selected on same-season val) rejected.
- prepare.py deliberately unfrozen (v2): NFL_SPLIT env (weeks:17,18 default unchanged; weeks:14-18; gamefold:5:f hash-balanced folds 2617-3241; none), NFL_DATA_DIR/NFL_RAW_DIR for Kaggle read-only mounts. train.py: cuda, STEP_BUDGET, no-val->EMA. All smoke-tested locally (default split still 12544/1559).
- Kaggle GPU training kernel (neel999/nfl-bdb26-train): preprocess + shift screen (lean vs 1st-place cfg, train wks1-13 val wks14-18, 12k steps) + 5-fold x 1st-place-scale (4.85M params, 37k steps, best-of on fold val) -> kf0..kf4.pt. 8h guard; runs server-side while laptop is off.

## 2026-07-02 14:20 — train kernel v1 postmortem + v2
- v1 (P100) failed: modern torch on Kaggle has no sm_60 kernels -> "no kernel image available"; script tolerated per-run failures so status=complete with no outputs. Fix: machineShape=NvidiaTeslaT4 on kernels/push + fail-fast cuda op at kernel start.
- Push API gotchas: new-kernel slug is DERIVED FROM TITLE (actual slug: nfl-bdb26-train-5-fold-1st-place-scale); pushing a new version requires that exact slug, else 409 title-in-use.

## 2026-07-03 05:30 UTC — round-3 (Kaggle T4, server-side) results
- Kaggle GPU kernel trained while laptop was closed. 12h limit + 2 shift-screens (34+110min) left time for only 1 of 5 folds.
- RESULTS.json: shift_lean (MULT32/1.9M, AUX_W=0, 12k steps, season-shift weeks14-18 val) = 0.4900; shift_1stplace (MULT64/4.9M, AUX_W=0.8, 12k steps, same split) = 0.6412 (UNDERTRAINED big model, not a real verdict); kfold_0 (MULT64/4.9M, 37k steps, gamefold5:0) = 0.4685.
- Interpretation: the shift screen's apparent "scale is bad" was a step-count confound — at 12k steps the 4.9M model is starved. At full 37k steps the full-scale model reaches 0.4685 (gamefold, in-distribution). We do NOT have big-model @ 37k on the season-shift split, so no clean scale verdict for OOD.
- kf0.pt (4.9M, 0.4685 gamefold) is our best-trained checkpoint. Submitting it single to the real LB (all 5 daily submissions available on 07-03 UTC; prior 0.579 was 07-02).

## 2026-07-03 06:00 UTC — kf0 real LB = 0.53559 (submission 54287916)
- kf0 (full-scale MULT64/4.9M, 37k steps, single model) scored 0.53559 private, beating the r2 2-model lean ensemble (0.57915) by 0.0436. Scale + full training generalizes to the 2025 test better than lean models did — round-3 direction validated on the REAL LB (our same-season holdout had not shown it).
- Standing: 0.579 (r2) -> 0.536 (kf0) | 20th=0.5046 | 1st=0.4634. Gap to 1st now ~0.072; to top-20 ~0.031.
- Next: (B) submit kf0+shift_lean cross-config ensemble to test if ensembling helps on real LB; (C) train remaining full-scale folds for a 5-fold ensemble if GPU quota allows. 4 submissions left today.

## 2026-07-03 06:12 UTC — ensemble B wasted (dataset race) + fix
- Submission B (kf0+shift_lean, ref 54288134) scored 0.53559 — IDENTICAL to kf0 alone. Kernel log: "loaded 1 checkpoints". Root cause: Kaggle processes new dataset versions ASYNC; the submit pipeline pushed+ran the kernel immediately after dataset_upload, so it attached the PREVIOUS version (v3, kf0-only) instead of v4 (kf0+shift_lean). Wasted 1 submission (2/5 used today, both 0.53559).
- FIX: kaggle_submit.py stage_dataset now records prev version, then polls /datasets/view + /datasets/status until currentVersionNumber bumps AND status=="ready" (600s timeout) before returning. Verified dataset v4 now = 27MB (both files), status ready.
- Decision: NOT re-submitting the shift_lean ensemble (low value — lean 12k-step model). Saving submissions for the kf0+kf1+kf2 full-scale fold ensemble now training (kernel v3, ~11h).

## 2026-07-03 11:25 UTC — 3-fold full-scale ensemble = 0.53405 (ref 54306390)
- kf0+kf1+kf2 (all MULT64 37k-step, per-model stats, hflip TTA) scored 0.53405 private — only 0.0015 better than kf0 alone (0.53559). Fold ensembling nearly worthless here: same arch + 80%-overlap data => highly correlated members (matches R2 seed-ensemble gain of 0.0016).
- Fixed real bug en route: submission notebook shared one members stats across all models; now per-model config+stats. Also fixed dataset-version race (kernel loaded stale v; now waits for ready). Both verified (loaded 3 checkpoints).
- STANDING: 0.579 (r2 lean ens) -> 0.53559 (kf0 single full-scale) -> 0.53405 (3-fold). 20th=0.5046, 1st=0.4634.
- DIAGNOSIS: the 0.579->0.536 jump was model SCALE + full 37k-step training (real lever). Ensemble size is tapped (~0.0015/member). Remaining ~0.07 gap to 1st is PER-MODEL generalization: our single full-scale model = 0.536 on 2025 test vs top ~0.47. Our gamefold val 0.4685 -> real 0.536 = 0.067 OOD gap; top solutions have ~0 CV->LB gap. Next real lever = close that gap (heavier augmentation / stronger regularization / possibly missing training data or features), NOT more ensembling. 2 submissions left today.

## 2026-07-03 12:15 UTC — round 4: three-pronged attack on the 0.067 OOD gap
- WHY v3/v4/v5 gained little: v4 was a dataset-race duplicate of v3 (kf0); v5 proved fold-ensembling adds only ~0.0015 (same arch, 80% shared data => correlated errors). All remaining signal is per-model generalization: kf0 CV 0.4685 vs test 0.53559.
- Code audit (subagent) vs 1st-place reference: features/statics/rotation/frame-mapping MATCH; angle-convention differs but is internally consistent (red herring); REAL finds = (1) our Model kept 2nd-place cumsum decoding — true 1st-place head predicts per-frame displacement directly (no error compounding over 48 frames under shift); (2) they clip preds to field at inference (we now do too); (3) they use global stats vs our per-fold (minor).
- Web research (subagent): 1st-place slide deck (Speaker Deck) readable — their two biggest documented wins, time-shift aug (+0.035) and relative ball/receiver coords, we ALREADY have. 3rd place = "PreTrain-FineTune". Train data is 2023 ONLY (verified: no 2024 csvs exist) and test is the 2025 season => two-season shift; no 2024 data to fine-tune on.
- ACTIONS: train.py gains AUG_NOISE (per-frame sensor jitter on x/y/s/a/dir/o) and DECODE knob ("cumsum"|"direct"); notebook gains per-ckpt DECODE default + field clipping. Kernel v4 launched on T4 (dataset v6): aug2_f0 (rot p1.0, shift p0.9, xy-bias 2/3, noise 1.0) vs direct_f0 (DECODE=direct), both full-scale 31k steps gamefold:5:0 seed 1000 — clean A/B against kf0. Quota-guarded (MAX_SEC 10.4h vs ~10.9h remaining; refresh 00:00 UTC).
- parity_test.py running locally: feeds raw weeks-17/18 csvs through the notebook predict() (the exact API path) and compares to predict_val on the same plays/ckpt — the last unchecked failure mode.

## 2026-07-03 20:10 UTC — THE BUG: n_out=0 for unscored players at inference (cost ~0.052)
- Parity test: same kf0 ckpt, same weeks-17/18 plays — direct path 0.461113 vs API path 0.512729 => +0.0516 gap. Array diff: X/masks/ball/role/side identical on 120/120 plays; ONLY n_out mismatches (by 8-17).
- Root cause: prepare.preprocess reads n_out from input-csv num_frames_output (ALL players); the notebook derived it from the test dataframe, which only lists SCORED players -> unscored players got n_out=0, a value seen only on padding slots in training. Poisoned the transformer context on every play.
- Fix: play_arrays now reads num_frames_output from test_input (column verified in API schema); fallback to old mapping if absent. Re-running full parity to verify, then resubmitting the 3-fold ensemble (expect ~0.48 real test, below the 0.5046 top-20 cutoff).
- Implication: the "season-shift OOD gap" was ~77% inference bug. True OOD gap for kf0 is ~0.015, consistent with top teams. The aug++/direct A/B still running remains useful for the residual.

## 2026-07-04 05:46 UTC — PARITY CLEAN after n_out fix + r4 A/B results
- Parity re-verify: API-path 0.460833 vs direct-path 0.461113 = gap -0.000281 (just the new field-clip). The submission path now faithfully matches local eval. n_out bug fully fixed.
- r4 A/B (full-scale, gamefold:5:0, seed 1000, 31k steps): direct_f0 (DECODE=direct, true 1st-place head) val=0.46136; aug2_f0 (heavy rot/shift/bias+noise) val=0.55032. Verdict: direct decode >= cumsum (0.461 vs kf0 0.4685) AND matches reference; heavy augmentation HURTS (reject). direct_f0 also gives decode-diversity for ensembling (decorrelated from cumsum folds).
- Expected real-test impact of n_out fix: prior kf0 single 0.53559 - 0.052 ~= 0.484; 3-fold 0.53405 -> ~0.482. 5 fresh submissions today.

## 2026-07-04 06:00 UTC — DECODE-correct local solos (weeks 17-18, CONTAMINATED ~80%)
- kf2 0.44773, direct_f0 0.45695, kf1 0.45916, kf0 0.46111; equal-weight 4-pool 0.45209. Greedy picks kf2-only because ensembling does not help on this contaminated eval (each model overfit the plays it trained on) — NOT predictive of ensemble value on the true OOD 2025 test. Real submissions are the arbiter.
- Plan (5 subs today): (D) corrected 3-fold cumsum [running]; (E) diverse 4-model kf0+kf1+kf2+direct_f0 (decode+fold diversity, notebook handles mixed DECODE per-model); reserve 3.

## 2026-07-04 06:24 UTC — *** CLEARED TOP 20 *** corrected 3-fold = 0.47490 (ref 54320531)
- Same kf0+kf1+kf2 that scored 0.53405 WITH the n_out bug now scores 0.47490 with it fixed: -0.0591 from one inference fix. Below the 20th-place cutoff (0.50460); 0.0115 from 1st (0.46340).
- Progression: 0.57915 -> 0.53559 -> 0.53405 (all n_out-bugged) -> 0.47490 (fixed). The whole "OOD gap" narrative was ~77% this bug.
- NEXT: 4-model blend kf0+kf1+kf2+direct_f0 (adds direct-decode diversity, decorrelated from the 3 cumsum folds) -> submitting. 4 subs left.

## 2026-07-04 06:48 UTC — 4-model blend = 0.47458 (ref 54321004, new best by 0.0003)
- Adding direct_f0 to the 3 cumsum folds moved 0.47490 -> 0.47458 only. Equal-weight ensembling tapped (members still correlated). 0.0112 from 1st (0.4634). 1st place was SOLO ~0.4634, so the gap is per-model quality, not ensemble size.
- Last submission today: direct_f0 SINGLE (best single by CV 0.4614, never submitted alone) — isolates the direct-decode real-LB score to decide if a full 5-fold DIRECT ensemble can reach 1st. Then overnight: train direct-decode folds (quota refreshes 00:00 UTC).

## 2026-07-04 00:01 UTC — direct_f0 single = 0.47890 (ref 54321562)
- Singles on REAL test ~0.478-0.479 (direct_f0 0.4789); 3-fold 0.4749; 4-model 0.4746. So ensembling DOES help on the true test (~0.004 from 1->3 models) — the contaminated local eval (each model trained on 80% of weeks 17-18) had hidden this by making singles look as good as ensembles. Lesson: trust the real LB over contaminated CV for ensemble decisions.
- direct decode is NOT a better single (0.4789 vs cumsum ensemble 0.4749); its value is diversity in a blend (+0.0003 in 4-model).
- RECALIBRATED path to 1st (0.4634, was SOLO): per-model gap ~0.015 (our singles 0.478 vs their 0.4634). Ensemble curve flattening at ~0.474, so more correlated members ~= +0.002 only. Bigger levers: (1) FULL-DATA models (train on 100%, no fold holdout) ~ +0.005-0.008; (2) direct-fold set for a 6-model blend ~ +0.002-0.003; (3) rotation TTA ~ +0.001-0.002. Combining may reach ~0.465-0.468 — beating 0.4634 is a genuine stretch but in range.
- STANDING: best 0.47458 (top-20, cutoff 0.50460), 0.0112 from 1st. Direct-fold kernel training df1/df2 (~10h). 4 subs left today.

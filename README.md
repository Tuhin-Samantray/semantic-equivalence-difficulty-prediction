# Transformation-Aware Verification Difficulty Prediction

Predict, **before** running an equivalence checker, how expensive it will be to verify that a program `P1` and a transformed version `P2 = transform(P1)` are equivalent. The predictor uses cheap features of the pair (per-program, the delta between P1 and P2, which transformations were applied, and a cheap formula probe). It is evaluated on prediction accuracy and on whether it makes batch verification deliver results sooner.

All programs are small, synthetic, integer-only Python functions (2-4 arguments, constant-trip loops, if/else). Verification is a Z3 bit-vector miter, which is complete here because constant-trip loops are fully unrolled. Results do not automatically transfer to real-world code.

## Headline results

Numbers below come from the committed result files. RandomForest was fixed as the headline model before any results were seen.

### 1. Prediction accuracy (1500 pairs, 300 base programs, grouped splits)

| Set | Features | CV MAE | Test MAE | Test R2 | Test Spearman | Timeout AUC | Overhead/pair* |
|---|---|---|---|---|---|---|---|
| A | P1 only (baseline) | 0.242 | 0.241 | 0.539 | 0.888 | 0.627 | 4 ms |
| B | P1 + P2 | 0.260 | 0.242 | 0.543 | 0.904 | 0.644 | 36 ms |
| C | P1 + Delta | 0.201 | 0.186 | 0.646 | 0.903 | 0.845 | 36 ms |
| D | C + Transform metadata | 0.201 | 0.180 | 0.655 | 0.908 | 0.842 | 36 ms |
| E | D + static probe | 0.148 | 0.129 | 0.828 | 0.950 | 0.966 | 505 ms |
| F | E + dynamic probe | 0.121 | 0.107 | 0.882 | 0.971 | 0.983 | 516 ms |

MAE is on log10(1 + verification ms). *Overhead was measured while building the dataset with 12 parallel workers, so it overstates single-process cost.

- Delta features improve on the P1-only baseline; transformation metadata adds almost nothing on top of delta; concatenating P2 does not help.
- Most of the large gain comes from the static probe (set E): test MAE falls about 46% versus the baseline.
- Set F's dynamic probe is a 20 ms solver run, which reveals part of the answer itself. It is reported separately and is not the headline.
- ExtraTrees has slightly lower CV MAE than RandomForest (set E: 0.133 vs 0.148). RandomForest remains the headline because it was fixed in advance. All four models are in `experiments/results/cv_results.csv` and figure 03.
- The test split has 300 pairs (about 15-20 timeouts), so test AUC is noisy.

### 2. Scheduling benefit: real wall-clock run (one worker, overhead included)

Test pairs were split by base program into 5 disjoint batches of 60. Each policy verified every batch with the real Z3 checker and a 5 s timeout, 3 repeats, random policy order. Predicted policies pay feature extraction and prediction up front. Speedup is the mean over the 5 batches of FIFO time divided by policy time.

| Policy | Mean time per pair | Speedup vs FIFO | Range across batches | Batches better than FIFO | Overhead/batch | Makespan vs FIFO |
|---|---|---|---|---|---|---|
| FIFO | 8.60 s | 1.00x | n/a | n/a | 0 | 1.00x |
| P1-only baseline (A) | 7.16 s | 1.18x | 0.85 - 1.70 | 3 of 5 | 0.11 s | 0.98x |
| Transformation-aware (D) | 4.69 s | 1.59x | 0.43 - 3.37 | 2 of 5 | 0.64 s | 0.93x |

- Ordering by predicted cost delivers results sooner than FIFO, even after paying overhead.
- Set D wins big on timeout-heavy batches (2.3x and 3.4x) and loses on quick batches (0.43x), because its up-front extraction is large against a short batch. The baseline is the safer predictor.
- With only 5 batches, "set D beats the baseline" is **not** established as robust. The supported claim is: better on timeout-heavy batches, worse on quick ones.
- Neither predictor improves makespan. Ordering changes when results arrive, not when the batch finishes.
- Set E was not run in real time. In the replay it lost badly to its overhead (0.39x), because extraction cost more than the batch's verification.

### 3. Scheduling replay (recorded costs, 300 overlapping batches of 60)

| Policy | 1 worker: speedup vs FIFO | 4 workers: speedup vs FIFO |
|---|---|---|
| Oracle (perfect order) | 4.06x | 2.34x |
| P1-only baseline | 1.53x | 1.29x |
| Set D | 1.63x | 0.69x |
| Set E | 0.39x | 0.08x |

The replay batches overlap heavily and are not independent, so its intervals are tighter than the data justify. Dataset overhead timings were taken under CPU contention, so the replay overcharged set D (about 24 ms per pair, versus about 11 ms measured in the real run).

### Figures

Result folders: `experiments/results/` (set E run), `experiments/results_D/` (set D run), `experiments/results_real_clean/` (real wall-clock run).

| Figure | File |
|---|---|
| 01 label distribution | `experiments/results/01_label_distribution.png` |
| 02 cost by transformation | `experiments/results/02_cost_by_transformation.png` |
| 03 model comparison | `experiments/results/03_model_comparison.png` |
| 04 ablation | `experiments/results/04_ablation.png` |
| 05 predicted vs actual | `experiments/results/05_pred_vs_actual.png` |
| 06 feature importance | `experiments/results/06_feature_importance.png` (set D: `results_D/`) |
| 07 scheduling replay, bars | `experiments/results/07_scheduling_replay_bars.png` (set D: `results_D/`) |
| 08 scheduling replay, curves | `experiments/results/08_scheduling_replay_curves.png` (set D: `results_D/`) |
| 09 real wall-clock run | `experiments/results_real_clean/09_real_run.png` |

`experiments/results_real_throttled/` holds a first real run during which Windows energy-saver mode was on for the first repeat. It is kept for transparency and is not used for any result.

## Pipeline

1. **Generate** random programs and apply 1-3 of 10 semantics-preserving transformations (rename, commute, dead_code, const_fold, unroll, strength_reduce, distribute, reassociate, extract_temp, if_flip).
2. **Verify** each pair with a Z3 bit-vector miter (per-program width from 8, 12, 16, 24 or 32 bits, 5 s solve timeout). The label is `log10(1 + build_ms + solve_ms)`.
3. **Extract features:** P1 features, P2 features, delta (differences, ratios, AST edit proxies), transformation metadata, a static probe (miter formula statistics, no solving) and a dynamic probe (20 ms solver run). Each group's extraction time is measured.
4. **Train and evaluate** Ridge, RandomForest, ExtraTrees and HistGradientBoosting on six feature sets (A to F), with grouped 5-fold CV and a grouped 80/20 held-out test, so all pairs of one base program stay together.
5. **Schedule:** replay on recorded costs, then a real wall-clock run.

## Repository layout

```
src/
  generation/           programs.py, transforms.py
  verification/         semantics.py (one interpreter, concrete + Z3), equivalence.py
  ast_extractor/        features.py
  cfg_builder/          cfg.py
  feature_extraction/   probes.py, builder.py
  models/               train.py, scheduling.py, plots.py, realrun.py
experiments/            phase2_sample.py, builddataset.py, train_eval.py,
                        scheduling_replay.py, check_realrun_features.py, real_wallclock.py
experiments/results*/   committed outputs (see Figures)
tests/                  test_transforms.py, test_verify.py
data/                   raw, processed, verification_logs, sample (dataset.csv is not committed)
```

## Setup

Tested on Windows with Python 3.14 and the pinned versions in `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest tests\test_verify.py
```

`tests/test_transforms.py` runs property tests over 150-200 generated programs per transformation and takes several minutes.

## Reproducing the experiments

Run all commands from the repository root with the environment active.

```powershell
python -m experiments.phase2_sample                 # no options: verifies a 20-pair sample and prints timings
python -m experiments.builddataset --n-programs 300 --pairs 5 --workers 12
python -m experiments.train_eval --data data/processed/dataset.csv
python -m experiments.scheduling_replay --pred experiments/results/test_predictions.csv
python -m experiments.real_wallclock --batches 5 --repeats 3 --out experiments\results_real_clean
```

`builddataset` and `scheduling_replay` also accept `--out`, `--seed-offset`/`--seed`, `--timeout-ms`, `--probe-ms`, `--batch-size`, `--n-batches` and `--workers`; pass `--help` for details.

Notes:
- `data/processed/dataset.csv` is gitignored. Verification labels are wall-clock times, so a rebuilt dataset will not match the committed results exactly.
- `train_eval` writes `data/processed/models/model_ours.joblib` and `model_baseline.joblib`, which `real_wallclock` loads. Re-running it with a different `--ours` set replaces the saved model. The committed real-run results used set D as "ours".
- `real_wallclock` runs on one worker and takes about 15 minutes. For meaningful timings, turn off Windows energy-saver, plug in a charger, pause cloud sync if the repo is synced, and close other applications.
- Result CSVs are ignored by `.gitignore` (`*.csv`) and were added with `git add -f`.

## Limitations

- Synthetic, small, integer-only programs; not real code. A real-C pipeline (CBMC, LLVM passes) is future work.
- The real scheduling run has 5 batches; the range across batches is the honest uncertainty, and repeats measure timing noise, not sampling noise. Only one worker and one batch size were run in real time.
- The 300-pair test split has few timeouts (about 15-20), so test AUC and the replay intervals are noisy.
- Dataset timings were measured with 12 parallel workers, so they contain CPU-contention noise.
- The label includes Python-side build time, not solver time alone.
- Feature-extraction overhead is charged sequentially; parallel extraction and miter reuse were not tested.
- Model type was fixed in advance (RandomForest). `--model auto` now selects on the training split only; a check showed it picks ExtraTrees on sets D and E, the same choice as selecting on all data.
- Literature references have not yet been verified.

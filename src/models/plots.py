"""All figures used in the report. Every function saves a PNG and returns its path."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.generation.transforms import TRANSFORM_NAMES  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"FIFO": "#8c8c8c", "Pred-SJF (ours)": "#1f77b4", "Pred-SJF (P1-only baseline)": "#ff7f0e",
          "Oracle-SJF": "#2ca02c", "Longest-first": "#d62728"}


def _save(fig, path: str | Path) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def label_distribution(df: pd.DataFrame, path) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].hist(df["log_cost"], bins=40, color="#1f77b4")
    ax[0].set(xlabel="log10(1 + verification ms)", ylabel="pairs", title="Verification cost distribution")
    counts = df["outcome"].value_counts()
    ax[1].bar(counts.index, counts.values, color=["#2ca02c", "#d62728", "#ff7f0e"][:len(counts)])
    ax[1].set(title="Outcomes", ylabel="pairs")
    for i, v in enumerate(counts.values):
        ax[1].text(i, v, str(v), ha="center", va="bottom")
    return _save(fig, path)


def cost_by_transform(df: pd.DataFrame, path) -> str:
    data, labels, to = [], [], []
    for t in TRANSFORM_NAMES:
        m = df[f"t_{t}"] > 0
        if m.sum() >= 5:
            data.append(df.loc[m, "log_cost"].to_numpy())
            labels.append(f"{t}\n(n={int(m.sum())})")
            to.append(100 * df.loc[m, "is_timeout"].mean())
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].boxplot(data, tick_labels=labels, showfliers=True)
    ax[0].set(ylabel="log10(1 + ms)", title="Verification cost by applied transformation")
    ax[0].tick_params(axis="x", labelsize=7, rotation=45)
    ax[1].bar(range(len(to)), to, color="#d62728")
    ax[1].set_xticks(range(len(to)), [l.split("\n")[0] for l in labels], rotation=45, fontsize=7)
    ax[1].set(ylabel="% timeouts", title="Timeout rate by transformation")
    return _save(fig, path)


def model_comparison(cv: pd.DataFrame, path) -> str:
    models = list(cv["model"].unique())
    sets = list(cv["set"].unique())
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    w = 0.8 / len(models)
    for ai, (metric, title) in enumerate([("mae", "CV MAE (lower is better)"),
                                          ("spearman", "CV Spearman rank corr. (higher is better)")]):
        for mi, m in enumerate(models):
            sub = cv[cv.model == m].set_index("set").loc[sets]
            ax[ai].bar(np.arange(len(sets)) + mi * w, sub[f"{metric}_mean"], w,
                       yerr=sub[f"{metric}_std"], label=m, capsize=2)
        ax[ai].set_xticks(np.arange(len(sets)) + w * (len(models) - 1) / 2, sets)
        ax[ai].set(title=title, xlabel="feature set")
    ax[0].legend(fontsize=8)
    return _save(fig, path)


def ablation(test_metrics: pd.DataFrame, cv_best: pd.DataFrame, path) -> str:
    """Feature-set ablation for the selected model (CV mean +- std, held-out test as dots)."""
    sets = list(cv_best["set"])
    labels = [f"{k}\n{l}" for k, l in zip(cv_best["set"], cv_best["label"])]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, metric, title in zip(ax, ["mae", "r2", "spearman"],
                                ["MAE (lower better)", "R² (higher better)", "Spearman (higher better)"]):
        cols = ["#8c8c8c" if s == "A" else "#1f77b4" for s in sets]
        a.bar(range(len(sets)), cv_best[f"{metric}_mean"], yerr=cv_best[f"{metric}_std"],
              color=cols, capsize=3, label="5-fold group CV")
        a.scatter(range(len(sets)), test_metrics.set_index("set").loc[sets, metric], color="k",
                  zorder=5, label="held-out test", s=18)
        a.set_xticks(range(len(sets)), labels, fontsize=6.5)
        a.set(title=title)
    ax[0].legend(fontsize=8)
    return _save(fig, path)


def pred_vs_actual(y, p_base, p_ours, path, label_ours="ours") -> str:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
    lo, hi = float(min(y.min(), p_base.min(), p_ours.min())), float(max(y.max(), p_base.max(), p_ours.max()))
    for a, p, t in zip(ax, [p_base, p_ours], ["P1-only baseline", label_ours]):
        a.scatter(y, p, s=9, alpha=0.5)
        a.plot([lo, hi], [lo, hi], "k--", lw=1)
        a.set(xlabel="actual log10(1+ms)", title=t)
    ax[0].set_ylabel("predicted log10(1+ms)")
    return _save(fig, path)


def importance(groups: pd.Series, indiv: pd.Series, path, top: int = 15) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 2]})
    g = groups.sort_values()
    ax[0].barh(g.index, g.values, color="#1f77b4")
    ax[0].set(title="Feature-group importance\n(MAE increase when permuted)")
    t = indiv.head(top)[::-1]
    pal = {"p1_": "#8c8c8c", "p2_": "#bcbd22", "d_": "#1f77b4", "t_": "#9467bd",
           "pf_": "#2ca02c", "pd_": "#d62728"}
    col = [next((c for p, c in pal.items() if n.startswith(p)), "k") for n in t.index]
    ax[1].barh(t.index, t.values, color=col)
    ax[1].set(title=f"Top {top} individual features")
    ax[1].tick_params(axis="y", labelsize=7)
    return _save(fig, path)


def scheduling_bars(summary: pd.DataFrame, path) -> str:
    ws = sorted(summary["workers"].unique())
    pols = list(COLORS)
    fig, ax = plt.subplots(1, len(ws), figsize=(6.2 * len(ws), 4.4), squeeze=False)
    for a, k in zip(ax[0], ws):
        s = summary[summary.workers == k].set_index("policy").loc[pols]
        a.bar(range(len(pols)), s["mean_completion_ms"] / 1000, color=[COLORS[p] for p in pols])
        for i, (v, sp) in enumerate(zip(s["mean_completion_ms"] / 1000, s["speedup_mean_completion"])):
            a.text(i, v, f"{sp:.2f}x", ha="center", va="bottom", fontsize=8)
        a.set_xticks(range(len(pols)), [p.replace(" (", "\n(") for p in pols], fontsize=7)
        a.set(title=f"{k} worker(s): mean time-to-result per pair", ylabel="seconds")
    return _save(fig, path)


def scheduling_curves(curves: dict, summary: pd.DataFrame, path) -> str:
    ws = sorted(curves["grids"])
    fig, ax = plt.subplots(1, len(ws), figsize=(6.2 * len(ws), 4.2), squeeze=False)
    for a, k in zip(ax[0], ws):
        for pol in COLORS:
            a.plot(curves["grids"][k] / 1000, curves["mean_curves"][(k, pol)], label=pol, color=COLORS[pol])
        a.set(xlabel="wall-clock (s)", ylabel="fraction of batch verified",
              title=f"Throughput curve, {k} worker(s) (replay)")
    ax[0][0].legend(fontsize=7)
    return _save(fig, path)


def real_run_curves(results: dict, path) -> str:
    """results[policy] = list of completion-time arrays (seconds, one per repeat)."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for pol, runs in results.items():
        for r, done in enumerate(runs):
            d = np.sort(done)
            ax[0].step(d, np.arange(1, len(d) + 1) / len(d), where="post", color=COLORS.get(pol, "k"),
                       alpha=0.9 if r == 0 else 0.35, label=pol if r == 0 else None)
    ax[0].set(xlabel="wall-clock seconds (incl. feature extraction)", ylabel="fraction verified",
              title="REAL run: results over time")
    ax[0].legend(fontsize=7)
    pols = list(results)
    means = [np.mean([np.mean(d) for d in results[p]]) for p in pols]
    ax[1].bar(range(len(pols)), means, color=[COLORS.get(p, "k") for p in pols])
    for i, (p, m) in enumerate(zip(pols, means)):
        ax[1].text(i, m, f"{means[0] / m:.2f}x", ha="center", va="bottom", fontsize=8)
    ax[1].set_xticks(range(len(pols)), [p.replace(" (", "\n(") for p in pols], fontsize=7)
    ax[1].set(ylabel="mean time-to-result (s)", title="REAL run: mean time per pair")
    return _save(fig, path)

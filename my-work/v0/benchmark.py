# benchmark_wstep.py
from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("mpl_drip.custom")

# disable latex for matplotlib
plt.rcParams["text.usetex"] = False
# change font family back to default
plt.rcParams["font.family"] = "sans-serif"

# -------------------------------
# Config (keep tiny => fast)
# -------------------------------
SPECTRA = Path("spectra_anomalies.npz")  # change if needed
K_USE = 8  # None => use min(A.shape) from SVD init; or set an int
MAX_ITERS = 20  # keep small; early-stop will usually trigger sooner
TOL = 1e-6
Q_ROBUST = 1  # your current Q
RCOND = 1e-8
SEEDS = [0, 1, 2]  # 3 runs per arm => 6 fits total


# -------------------------------
# IO helpers
# -------------------------------
def read_npz(p: Path):
    f = np.load(p, allow_pickle=True)
    Y = f["observed"]
    Yt = f["true"]
    W0 = f["weights"]
    x = f["x"]
    # Optional metadata (safe defaults)
    weird_mask = (
        f["weird_row_mask"] if "weird_row_mask" in f else np.zeros(Y.shape[0], bool)
    )
    bad_mask = (
        f["bad_pixel_mask"] if "bad_pixel_mask" in f else np.zeros_like(Y, dtype=bool)
    )
    G_true = f["G_basis"] if "G_basis" in f else None
    return Y, Yt, W0, x, weird_mask, bad_mask, G_true


# -------------------------------
# Your solver components (NumPy/JAX mix)
# -------------------------------
def initial(Y, K_use=None, jitter=0.0, seed=0):
    Y0 = Y
    if jitter > 0:
        rng = np.random.default_rng(seed)
        Y0 = Y + jitter * rng.standard_normal(Y.shape).astype(Y.dtype)
    U, S, VH = jnp.linalg.svd(Y0, compute_uv=True, full_matrices=False)
    if K_use is not None and K_use < S.shape[0]:
        U, S, VH = U[:, :K_use], S[:K_use], VH[:K_use]
    A = U * S
    G = VH
    return A, G


def a_step(G, Y, W, rcond=1e-8):
    GT = G.T  # (M,K)

    def solve_row(y_i, w_i):
        D = jnp.sqrt(w_i)[:, None]
        X = D * GT
        y = (D.squeeze()) * y_i
        a_i, *_ = jnp.linalg.lstsq(X, y, rcond=rcond)
        return a_i

    return jax.vmap(solve_row, in_axes=(0, 0))(Y, W)


def g_step(A, Y, W, rcond=1e-8):
    def solve_col(y_j, w_j):
        D = jnp.sqrt(w_j)[:, None]
        X = D * A
        y = (D.squeeze()) * y_j
        g_j, *_ = jnp.linalg.lstsq(X, y, rcond=rcond)
        return g_j

    G_cols = jax.vmap(solve_col, in_axes=(1, 1))(Y, W)
    return G_cols.T


def w_step(A, G, Y, W_in, Q=1):
    Δ2 = (Y - A @ G) ** 2
    return W_in * Q**2 / (W_in * Δ2 + Q**2)


def reorient(A, G):
    L = A @ G
    U, S, VH = jnp.linalg.svd(L, compute_uv=True, full_matrices=False)
    A_new = U * S
    G_new = VH
    return A_new, G_new


def run_fit(
    Y, W_in, robust=True, max_iters=20, tol=1e-6, rcond=1e-8, Q=1, K_use=None, seed=0
):
    A, G = initial(Y, K_use=K_use, jitter=0.0, seed=seed)
    W = deepcopy(W_in)
    ΔGs = []
    t0 = time.time()
    for it in range(max_iters):
        A_ = a_step(G, Y, W, rcond=rcond)
        G_ = g_step(A_, Y, W, rcond=rcond)
        ΔG = float(jnp.linalg.matrix_norm(G_ - G))
        ΔGs.append(ΔG)
        A_, G_ = reorient(A_, G_)
        if robust:
            W = np.array(w_step(A_, G_, Y, W_in, Q=Q))
        # else keep W as-is (fixed at W_in)
        A, G = A_, G_
        if ΔG < tol:
            break
    wall = time.time() - t0
    return np.array(A), np.array(G), np.array(W), np.array(ΔGs), wall, it + 1


# -------------------------------
# Metrics
# -------------------------------
def rmse_clean(Y_true, A, G, weird_mask, bad_mask):
    Yhat = A @ G
    cmask = (~weird_mask)[:, None] & (~bad_mask)
    if cmask.sum() == 0:
        return np.nan
    d = (Y_true - Yhat)[cmask]
    return float(np.sqrt(np.mean(d**2)))


def subspace_gap(G_true, G_hat, K=None):
    if G_true is None:
        return np.nan, np.nan
    # compare column spaces in R^M (use first K rows of each basis matrix as components)
    X = G_true.T  # (M,Kt)
    Y = G_hat.T  # (M,Kh)
    if K is None:
        K = min(X.shape[1], Y.shape[1])
    X = X[:, :K]
    Y = Y[:, :K]

    def orth(M):
        Q, _ = np.linalg.qr(M, mode="reduced")
        return Q

    U = orth(X)
    Uh = orth(Y)
    s = np.linalg.svd(U.T @ Uh, compute_uv=False)
    s = np.clip(s, 0, 1)
    ang = np.arccos(s)
    return float(np.mean(np.sin(ang))), float(np.max(np.sin(ang)))


def row_scores(Y, A, G, W_in, W_final):
    Yhat = A @ G
    resid = np.sqrt(W_in) * (Y - Yhat)
    score_resid = np.linalg.norm(resid, axis=1)
    eps = 1e-12
    wdrop = 1.0 - W_final / (W_in + eps)
    wdrop = np.where(np.isfinite(wdrop), wdrop, 0.0)
    score_wdrop = np.nanmean(wdrop, axis=1)
    return score_resid, score_wdrop


def auc_binary(scores, labels):
    # Simple, dependency-free ROC-AUC
    s = np.asarray(scores)
    y = np.asarray(labels).astype(int)
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    T = tp[-1]
    F = fp[-1]
    if T == 0 or F == 0:
        return np.nan
    tpr = tp / T
    fpr = fp / F
    # Trapezoid
    auc = np.trapz(tpr, fpr)
    return float(auc), fpr, tpr


def row_scores_2(Y, A, G, W_in, W_final):
    Yhat = A @ G
    R = np.sqrt(W_in) * (Y - Yhat)  # standardized residuals
    s_l2 = np.linalg.norm(R, axis=1)
    s_linf = np.max(np.abs(R), axis=1)
    s_q90 = np.quantile(np.abs(R), 0.90, axis=1)

    eps = 1e-12
    mult = np.where(np.isfinite(W_final), W_final / (W_in + eps), 1.0)
    s_wdrop_mean = 1.0 - np.nanmean(mult, axis=1)
    s_wdrop_frac = np.mean(mult < 0.5, axis=1)

    return {
        "resid_l2": s_l2,
        "resid_linf": s_linf,
        "resid_q90": s_q90,
        "wdrop_mean": s_wdrop_mean,
        "wdrop_frac": s_wdrop_frac,
    }


def roc_curve_np(scores, labels):
    s = np.asarray(scores)
    y = np.asarray(labels).astype(int)
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    T = tp[-1]
    F = fp[-1]
    if T == 0 or F == 0:
        return np.array([0, 1]), np.array([0, 1]), np.nan
    tpr = tp / T
    fpr = fp / F
    auc = float(np.trapz(tpr, fpr))
    # prepend origin
    fpr = np.concatenate([[0.0], fpr])
    tpr = np.concatenate([[0.0], tpr])
    return fpr, tpr, auc


# -------------------------------
# Main experiment
# -------------------------------
Y, Y_true, W_in, x, weird_mask, bad_mask, G_true = read_npz(SPECTRA)

results = []
traces = {}
for robust in [False, True]:
    arm = "no_w" if not robust else "with_w"
    for seed in SEEDS:
        A, G, W, dGs, wall, niter = run_fit(
            Y,
            W_in,
            robust=robust,
            max_iters=MAX_ITERS,
            tol=TOL,
            rcond=RCOND,
            Q=Q_ROBUST,
            K_use=K_USE,
            seed=seed,
        )
        rmse = rmse_clean(Y_true, A, G, weird_mask, bad_mask)
        gap_mean, gap_max = subspace_gap(
            G_true,
            G,
            K=min(G.shape[0], G_true.shape[0]) if G_true is not None else None,
        )
        sr, sw = row_scores(Y, A, G, W_in, W)
        auc_r, fpr_r, tpr_r = auc_binary(sr, weird_mask)
        auc_w, fpr_w, tpr_w = auc_binary(sw, weird_mask)

        results.append(
            dict(
                arm=arm,
                seed=seed,
                rmse_clean=rmse,
                subgap_mean=gap_mean,
                subgap_max=gap_max,
                auc_resid=auc_r,
                auc_wdrop=auc_w,
                wall_s=wall,
                iters=niter,
            )
        )
        traces[(arm, seed)] = dGs

# -------------------------------
# Aggregate + print table
# -------------------------------
import pandas as pd

df = pd.DataFrame(results)
agg = df.groupby("arm").agg(
    rmse_clean_mean=("rmse_clean", "mean"),
    rmse_clean_std=("rmse_clean", "std"),
    subgap_mean=("subgap_mean", "mean"),
    auc_resid_mean=("auc_resid", "mean"),
    auc_wdrop_mean=("auc_wdrop", "mean"),
    wall_s_mean=("wall_s", "mean"),
    iters_mean=("iters", "mean"),
    n=("arm", "count"),
)
print("\nSummary over seeds:")
print(agg)

# -------------------------------
# Plots
# -------------------------------
# 1) ΔG traces (one per arm, best seed by final ΔG)
fig, ax = plt.subplots(1, 2, figsize=(10, 4), layout="compressed", dpi=160)
for j, arm in enumerate(["no_w", "with_w"]):
    # pick representative seed (min final ΔG)
    subset = [(seed, traces[(arm, seed)]) for seed in SEEDS]
    seed_best, tr = min(subset, key=lambda t: t[1][-1])
    ax[j].plot(tr, lw=2)
    ax[j].set_title(f"ΔG trace — {arm} (seed {seed_best})")
    ax[j].set_xlabel("iteration")
    ax[j].set_ylabel("||ΔG||_F")
    ax[j].set_yscale("log")
plt.show()

# 2) RMSE_clean bars
fig, ax = plt.subplots(figsize=(6, 4), layout="compressed", dpi=160)
bars = df.groupby("arm")["rmse_clean"].mean().reindex(["no_w", "with_w"])
ax.bar(bars.index, bars.values)
ax.set_ylabel("RMSE (clean mask)")
ax.set_title("Reconstruction on clean entries")
plt.show()

# 3) Subspace gap bars (skip if truth missing)
if not np.all(np.isnan(df["subgap_mean"])):
    fig, ax = plt.subplots(figsize=(6, 4), layout="compressed", dpi=160)
    bars = df.groupby("arm")["subgap_mean"].mean().reindex(["no_w", "with_w"])
    ax.bar(bars.index, bars.values)
    ax.set_ylabel("Mean sin(principal angles)")
    ax.set_title("Subspace gap to true basis")
    plt.show()

# 4) Oddball detection AUC bars
fig, ax = plt.subplots(figsize=(6, 4), layout="compressed", dpi=160)
a1 = df.groupby("arm")["auc_resid"].mean().reindex(["no_w", "with_w"])
a2 = df.groupby("arm")["auc_wdrop"].mean().reindex(["no_w", "with_w"])
width = 0.35
xloc = np.arange(2)
ax.bar(xloc - width / 2, a1.values, width, label="Residual score")
ax.bar(xloc + width / 2, a2.values, width, label="Weight drop")
ax.set_xticks(xloc, ["no_w", "with_w"])
ax.set_ylim(0, 1.0)
ax.set_ylabel("AUC")
ax.set_title("Weird row detection")
ax.legend()
plt.show()

# 5) Iterations / wall time
fig, ax = plt.subplots(1, 2, figsize=(10, 4), layout="compressed", dpi=160)
for j, metric in enumerate(["iters", "wall_s"]):
    bars = df.groupby("arm")[metric].mean().reindex(["no_w", "with_w"])
    ax[j].bar(bars.index, bars.values)
    ax[j].set_title(metric)
plt.show()


# ROC plots for a representative run per arm
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="compressed", dpi=160)
for j, (arm, robust) in enumerate([("no_w", False), ("with_w", True)]):
    # pick best seed by final ΔG
    candidates = [(seed, traces[(arm, seed)]) for seed in SEEDS]
    seed_best, _ = min(candidates, key=lambda t: t[1][-1])
    A, G, W, *_ = run_fit(
        Y,
        W_in,
        robust=robust,
        max_iters=MAX_ITERS,
        tol=TOL,
        rcond=RCOND,
        Q=Q_ROBUST,
        K_use=K_USE,
        seed=seed_best,
    )
    scores = row_scores_2(Y, A, G, W_in, W)
    # choose which to plot for each arm
    keys = ["resid_l2", "resid_linf", "resid_q90"]
    if robust:
        keys += ["wdrop_mean", "wdrop_frac"]
    for k in keys:
        fpr, tpr, auc = roc_curve_np(scores[k], weird_mask)
        axes[j].plot(fpr, tpr, label=f"{k} (AUC {auc:.2f})", lw=2)
    axes[j].plot([0, 1], [0, 1], "k--", lw=1)
    axes[j].set_xlim(0, 1)
    axes[j].set_ylim(0, 1)
    axes[j].set_xlabel("FPR")
    axes[j].set_ylabel("TPR")
    axes[j].set_title(f"ROC — {arm} (seed {seed_best})")
    axes[j].legend(fontsize=8)
plt.show()


def row_diagnostics(Y, A, G, W_in, W_final, weird_mask, arm_name):
    Yhat = A @ G
    R = np.sqrt(W_in) * (Y - Yhat)
    resid_norm = np.linalg.norm(R, axis=1)

    eps = 1e-12
    mult = np.where(np.isfinite(W_final), W_final / (W_in + eps), 1.0)
    wdrop_mean = 1.0 - np.nanmean(mult, axis=1)

    # Histograms
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=150, layout="compressed")
    ax[0].hist(resid_norm[~weird_mask], bins=30, alpha=0.7, label="normal")
    ax[0].hist(resid_norm[weird_mask], bins=30, alpha=0.7, label="weird")
    ax[0].set_title(f"{arm_name}: residual L2 norms")
    ax[0].legend()

    ax[1].hist(wdrop_mean[~weird_mask], bins=30, alpha=0.7, label="normal")
    ax[1].hist(wdrop_mean[weird_mask], bins=30, alpha=0.7, label="weird")
    ax[1].set_title(f"{arm_name}: mean weight drop")
    ax[1].legend()

    plt.show()

    # Print stats
    print(f"\n{arm_name} diagnostics:")
    for name, arr in [("resid_norm", resid_norm), ("wdrop_mean", wdrop_mean)]:
        mu_norm = float(np.mean(arr[~weird_mask]))
        mu_weird = float(np.mean(arr[weird_mask]))
        print(f"  {name}: mean normal={mu_norm:.3f}, mean weird={mu_weird:.3f}")


# Run diagnostics on the last run from each arm (pick seed 0 for reproducibility)
for arm, robust in [("no_w", False), ("with_w", True)]:
    A, G, W, *_ = run_fit(
        Y,
        W_in,
        robust=robust,
        max_iters=MAX_ITERS,
        tol=TOL,
        rcond=RCOND,
        Q=Q_ROBUST,
        K_use=K_USE,
        seed=0,
    )
    row_diagnostics(Y, A, G, W_in, W, weird_mask, arm)

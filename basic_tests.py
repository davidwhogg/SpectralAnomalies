# robust_wstep_sanity.py
# Minimal, from-scratch verification of the w-step (Stages 0–4)
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Small defaults (fast)
# -----------------------------
SEED = 0
N = 500  # rows (spectra)
M = 240  # columns (pixels)
K_TRUE = 3  # true rank used to generate (not for stage 6)
K_FIT = 4  # rank used in fitting (keep same for stages 0–2; works for 3–4 too)
SIGMA = 0.01  # baseline noise (homoskedastic for 0–2)
Q = 1  # robust scale in chi-units (1..3 reasonable)
MAX_ITERS = 100
TOL = 1e-8
RNG = np.random.default_rng(SEED)


# -----------------------------
# Simple low-rank generator
# -----------------------------
def make_true_basis(M: int, K: int):
    """K smooth rows in feature space (low freq sines/cosines)."""
    x = np.linspace(0, 1, M, dtype=float)
    G = []
    G.append(np.ones_like(x))
    G.append(np.sin(2 * np.pi * 1.0 * x))
    G.append(np.cos(2 * np.pi * 1.0 * x))
    f = 2.0
    while len(G) < K:
        G.append(np.sin(2 * np.pi * f * x))
        if len(G) == K:
            break
        G.append(np.cos(2 * np.pi * f * x))
        f += 1.0
    G = np.stack(G[:K], axis=0)  # (K, M)
    return G


def generate_stage0(N=N, M=M, K=K_TRUE, sigma=SIGMA, seed=SEED):
    """Clean, homoskedastic, no anomalies."""
    rng = np.random.default_rng(seed)
    G = make_true_basis(M, K)
    A = rng.normal(0, 1, size=(N, K))
    Y_true = A @ G
    noise = rng.normal(0, sigma, size=(N, M))
    Y = Y_true + noise
    W_in = np.full((N, M), 1.0 / sigma**2, dtype=float)
    masks = dict(
        row_weird=np.zeros(N, bool),
        spike=np.zeros((N, M), bool),
        bad=np.zeros((N, M), bool),
    )
    return Y, Y_true, W_in, G, masks


def generate_stage1_row_anoms(N=N, M=M, K=K_TRUE, sigma=SIGMA, seed=SEED):
    """Add obvious row anomalies: replace some rows with high-frequency ripples / random flats."""
    Y, Y_true, W_in, G, masks = generate_stage0(N, M, K, sigma, seed)
    rng = np.random.default_rng(seed + 1)

    num_weird = max(6, N // 6)  # ~10 rows
    idx = rng.choice(N, size=num_weird, replace=False)
    x = np.linspace(0, 1, M, dtype=float)
    for i in idx:
        if True:  # rng.random() < 0.5:
            Y_true[i] = 1.0 + 0.25 * np.sin(
                2 * np.pi * 50 * x + rng.uniform(0, 2 * np.pi)
            )
        else:
            Y_true[i] = 1.0 + rng.uniform(-0.3, 0.3)
    Y = Y_true + rng.normal(0, sigma, size=(N, M))
    masks["row_weird"][idx] = True
    return Y, Y_true, W_in, G, masks


def generate_stage2_pixel_spikes(
    N=N, M=M, K=K_TRUE, sigma=SIGMA, seed=SEED, spike_frac=0.01, spike_amp_sigma=10.0
):
    """Clean rows, but inject random pixel spikes (known mask)."""
    Y, Y_true, W_in, G, masks = generate_stage0(N, M, K, sigma, seed)
    rng = np.random.default_rng(seed + 2)
    spike_mask = rng.random((N, M)) < spike_frac
    spikes = np.zeros((N, M), dtype=float)
    spikes[spike_mask] = (
        spike_amp_sigma * sigma * rng.choice([-1.0, 1.0], size=spike_mask.sum())
    )
    Y = Y_true + rng.normal(0, sigma, size=(N, M)) + spikes
    masks["spike"] = spike_mask
    return Y, Y_true, W_in, G, masks


def generate_stage3_bad_pixels(
    N=N, M=M, K=K_TRUE, sigma=SIGMA, seed=SEED, bad_frac=0.005
):
    """Clean data + randomly flagged bad pixels with zero weight and zero observed flux."""
    Y, Y_true, W_in, G, masks = generate_stage0(N, M, K, sigma, seed)
    rng = np.random.default_rng(seed + 3)
    bad_mask = rng.random((N, M)) < bad_frac
    # set observed to 0, weight to 0 at bad pixels
    Y = Y_true + rng.normal(0, sigma, size=(N, M))
    Y = Y.copy()
    W_in = W_in.copy()
    Y[bad_mask] = 0.0
    W_in[bad_mask] = 0.0
    masks["bad"] = bad_mask
    return Y, Y_true, W_in, G, masks


def generate_stage4_hetero_calibrated(N=N, M=M, K=K_TRUE, seed=SEED):
    """Heteroskedastic but calibrated: W_in built from the same Sigma used to draw noise."""
    rng = np.random.default_rng(seed + 4)
    G = make_true_basis(M, K)
    A = rng.normal(0, 1, size=(N, K))
    Y_true = A @ G
    # column shape (e.g., blaze); row scales
    x = np.linspace(0, 1, M)
    sigma_col = 0.01 * (1.0 + 0.6 * (x - 0.5) ** 2)  # varies across pixels
    row_scale = (1.0 + 0.15 * rng.standard_normal((N, 1))).clip(0.7, 1.3)
    Sigma = row_scale * sigma_col[None, :]
    noise = rng.normal(0, 1.0, size=(N, M)) * Sigma
    Y = Y_true + noise
    W_in = 1.0 / (Sigma**2)
    masks = dict(
        row_weird=np.zeros(N, bool),
        spike=np.zeros((N, M), bool),
        bad=np.zeros((N, M), bool),
    )
    return Y, Y_true, W_in, G, masks


# -----------------------------
# Solver (ALS with weighted lstsq)
# -----------------------------
def initial(Y: np.ndarray, K: int):
    U, S, VH = np.linalg.svd(Y, full_matrices=False)
    U, S, VH = U[:, :K], S[:K], VH[:K]
    A = U * S
    G = VH
    return A, G  # A:(N,K), G:(K,M)


def a_step(G, Y, W, rcond=1e-8):
    GT = G.T
    N = Y.shape[0]
    A = np.zeros((N, G.shape[0]), dtype=float)
    for i in range(N):
        wi = W[i]
        D = np.sqrt(wi)[:, None]
        X = D * GT
        y = (D.ravel()) * Y[i]
        Ai, *_ = np.linalg.lstsq(X, y, rcond=rcond)
        A[i] = Ai
    return A


def g_step(A, Y, W, rcond=1e-8):
    K = A.shape[1]
    M = Y.shape[1]
    Gcols = np.zeros((M, K), dtype=float)
    for j in range(M):
        wj = W[:, j]
        D = np.sqrt(wj)[:, None]
        X = D * A
        y = (D.ravel()) * Y[:, j]
        gj, *_ = np.linalg.lstsq(X, y, rcond=rcond)
        Gcols[j] = gj
    return Gcols.T


def w_step(A, G, Y, W_in, Q=Q):
    # Cauchy IRLS in chi-units
    R2 = (np.sqrt(W_in) * (Y - A @ G)) ** 2
    mult = (Q**2) / (Q**2 + R2)
    return W_in * mult


def run_fit(Y, W_in, K=K_FIT, robust=True, max_iters=MAX_ITERS, tol=TOL, Q=Q):
    A, G = initial(Y, K)
    W = W_in.copy()
    dGs = []
    for _ in range(max_iters):
        A_ = a_step(G, Y, W)
        G_ = g_step(A_, Y, W)
        dG = np.linalg.norm(G_ - G)
        dGs.append(dG)
        A, G = A_, G_
        if robust:
            W = w_step(A, G, Y, W_in, Q=Q)
        if dG < tol:
            break
    return A, G, W, np.array(dGs)


# -----------------------------
# Diagnostics (no ROC—just margins)
# -----------------------------
def row_mult(W_after, W_in):
    return (W_after / (W_in + 1e-12)).mean(axis=1)


def resid_norms(Y, A, G, W_in):
    R = np.sqrt(W_in) * (Y - A @ G)
    return np.linalg.norm(R, axis=1)


def stage0_checks(Y, Y_true, W_in):
    print("\n[Stage 0] Clean, no anomalies")
    A0, G0, W0, _ = run_fit(Y, W_in, robust=False)
    A1, G1, W1, _ = run_fit(Y, W_in, robust=True)

    mult = W1 / (W_in + 1e-12)
    med_mult = float(np.median(mult))
    p99_dev = float(np.percentile(np.abs(mult - 1), 99))

    # Theoretical change in weights
    r_med = 0.67448975
    expected_med = Q**2 / (Q**2 + r_med**2)

    tol = 0.03
    assert (expected_med - tol) <= med_mult <= (min(1.0, expected_med + tol)), (
        "Weights changed too much on clean data"
    )
    # assert p99_dev < 0.1, "A few weights deviated more than expected"
    print("  max|mult-1| = %.3e" % np.max(np.abs(mult - 1)))
    # assert np.max(np.abs(mult - 1)) < 1e-3, "Weights changed on clean data!"

    def rmse(A, G):
        return float(np.sqrt(np.mean((Y_true - A @ G) ** 2)))

    rm0, rm1 = rmse(A0, G0), rmse(A1, G1)
    print("  RMSE clean: no_w=%.5f, with_w=%.5f" % (rm0, rm1))
    assert abs(rm0 - rm1) < 5e-4, "RMSE should match on clean data"

    assert np.all(W1 <= W_in + 1e-12), "Weights must not increase"
    print("  PASS ✅")


def stage1_checks(Y, Y_true, W_in, row_mask):
    print("\n[Stage 1] Row anomalies (obvious)")

    A0, G0, W0, _ = run_fit(Y, W_in, robust=False)
    A1, G1, W1, _ = run_fit(Y, W_in, robust=True)

    m1 = row_mult(W1, W_in)
    r0 = resid_norms(Y, A0, G0, W_in)
    r1 = resid_norms(Y, A1, G1, W_in)

    med = lambda a, mask: float(np.median(a[mask]))
    margin_mult = med(m1, ~row_mask) - med(m1, row_mask)
    margin_res0 = med(r0, row_mask) - med(r0, ~row_mask)
    margin_res1 = med(r1, row_mask) - med(r1, ~row_mask)

    print(
        "  row_mult medians (with_w): normals=%.3f, weird=%.3f, margin=%.3f"
        % (med(m1, ~row_mask), med(m1, row_mask), margin_mult)
    )
    print(
        "  resid L2 medians: no_w Δ=%.3f, with_w Δ=%.3f (weird - normal)"
        % (margin_res0, margin_res1)
    )

    assert margin_mult > 0.4, "Row multiplier margin too small—Q or anomalies too weak?"
    assert margin_res1 > 0.0, "Weird rows should have larger residuals (with_w)"

    fig, ax = plt.subplots(1, 2, figsize=(8, 3), dpi=130, layout="compressed")
    ax[0].hist(m1[~row_mask], bins=20, alpha=0.7, label="normal")
    ax[0].hist(m1[row_mask], bins=20, alpha=0.7, label="weird")
    ax[0].set_title("with_w: row mean weight multiplier")
    ax[0].legend()
    ax[1].hist(r1[~row_mask], bins=20, alpha=0.7, label="normal")
    ax[1].hist(r1[row_mask], bins=20, alpha=0.7, label="weird")
    ax[1].set_title("with_w: row residual L2 (chi-units)")
    plt.show()
    print("  PASS ✅")


def stage2_checks(Y, Y_true, W_in, spike_mask):
    print("\n[Stage 2] Pixel spikes (obvious)")
    A1, G1, W1, _ = run_fit(Y, W_in, robust=True)
    mult = W1 / (W_in + 1e-12)

    med_clean = float(np.median(mult[~spike_mask]))
    med_spikes = float(np.median(mult[spike_mask])) if spike_mask.any() else np.nan
    margin = med_clean - med_spikes

    print(
        "  pixel mult medians: clean=%.3f, spikes=%.3f, margin=%.3f"
        % (med_clean, med_spikes, margin)
    )
    print(
        "  row mean multiplier (should stay near 1): median=%.3f"
        % float(np.median(mult.mean(axis=1)))
    )
    assert margin > 0.6, (
        "Pixel multiplier margin too small—spikes not downweighted enough?"
    )
    assert np.all(mult <= 1.0 + 1e-12), "Weights increased somewhere?"

    plt.figure(figsize=(5, 3), dpi=130)
    plt.imshow(mult, aspect="auto", cmap="viridis")
    plt.colorbar(label="weight multiplier")
    plt.title("with_w: W_after / W_in")
    plt.xlabel("pixel")
    plt.ylabel("row")
    plt.tight_layout()
    plt.show()
    print("  PASS ✅")


def stage3_checks(Y, Y_true, W_in, bad_mask):
    print("\n[Stage 3] Bad pixels (weight=0, flux=0)")
    # run robust fit; the zero-weight pixels should remain zero and not break anything
    A1, G1, W1, _ = run_fit(Y, W_in, robust=True)
    assert np.allclose(W1[bad_mask], 0.0), "Bad pixels must remain zero weight"
    assert np.isfinite(A1).all() and np.isfinite(G1).all(), "NaNs/inf in factors"
    print("  #bad pixels:", int(bad_mask.sum()))
    print(
        "  median row mean multiplier:",
        float(np.median((W1 / (W_in + 1e-12)).mean(axis=1))),
    )
    # show the bad mask and weight multiplier
    fig, ax = plt.subplots(1, 2, figsize=(9, 3), dpi=130, layout="compressed")
    ax[0].imshow(bad_mask, aspect="auto", cmap="gray_r")
    ax[0].set_title("bad pixel mask")
    ax[0].set_xlabel("pixel")
    ax[0].set_ylabel("row")
    ax[1].imshow(W1 / (W_in + 1e-12), aspect="auto", cmap="viridis")
    ax[1].set_title("W_after / W_in")
    ax[1].set_xlabel("pixel")
    ax[1].set_ylabel("row")
    plt.show()
    print("  PASS ✅")


def stage4_checks(Y, Y_true, W_in):
    print("\n[Stage 4] Heteroskedastic (calibrated)")
    Qs = [1.0, 2.0, 4.0]

    # Helper: robust scale of standardized residuals
    def robust_sigma_r(A, G):
        R = np.sqrt(W_in) * (Y - A @ G)  # chi-units
        r = R.ravel()
        med = np.median(r)
        mad = np.median(np.abs(r - med)) + 1e-12
        return 1.4826 * mad  # ≈ std if Normal

    r_med_sq = 0.67448975**2  # median(r^2) for N(0,1) ≈ 0.455

    for q in Qs:
        A, G, W, _ = run_fit(Y, W_in, robust=True, Q=q)
        mult = W / (W_in + 1e-12)
        med_mult = float(np.median(mult))
        sigma_r = robust_sigma_r(A, G)

        # Expected median multiplier at this Q and residual scale
        exp_med = (q**2) / (q**2 + (sigma_r**2) * r_med_sq)

        print(
            f"  Q={q:.1f}: median(mult)={med_mult:.3f}  "
            f"expected≈{exp_med:.3f}  sigma_r≈{sigma_r:.3f}"
        )

        # Tolerances: allow ±0.05 around the expectation
        assert abs(med_mult - exp_med) < 0.05, (
            "Median multiplier not consistent with Q and residual scale"
        )
        assert np.all(mult <= 1.0 + 1e-12), "Weights increased somewhere?"

    # One quick histogram at Q=2 for eyeballing
    A2, G2, W2, _ = run_fit(Y, W_in, robust=True, Q=2.0)
    R2 = np.sqrt(W_in) * (Y - A2 @ G2)
    plt.figure(figsize=(5, 3), dpi=130)
    plt.hist(R2.ravel(), bins=40, density=True, alpha=0.8)
    plt.title("Stage 4: standardized residuals (Q=2)")
    plt.xlabel("r")
    plt.ylabel("pdf (approx)")
    plt.tight_layout()
    plt.show()
    print("  PASS ✅")

    # =========================


# Stage 5: Mixed anomalies (same tests as Stage 6)
# =========================


def generate_stage5_mixed(
    N=N,
    M=M,
    K=K_TRUE,
    seed=SEED,
    # anomaly knobs
    row_weird_frac=0.12,  # ~12% rows replaced
    spike_frac=0.01,  # ~1% pixels spiked
    spike_amp_sigma=10.0,  # spike amplitude in baseline sigma units
    bad_frac=0.003,  # some missing pixels
):
    """
    Simple-but-complete toy: low-rank + heteroskedastic noise + row anomalies + pixel spikes + bad pixels.
    Returns (Y, Y_true, W_in, G_true, masks_dict).
    """
    rng = np.random.default_rng(seed)

    # -- true low-rank model --
    G_true = make_true_basis(M, K)
    A_true = rng.normal(0, 1, size=(N, K))
    Y_true = A_true @ G_true

    # -- heteroskedastic sigma --
    x = np.linspace(0, 1, M)
    sigma_col = 0.01 * (1.0 + 0.6 * (x - 0.5) ** 2)  # column shape
    row_scale = (1.0 + 0.10 * rng.standard_normal((N, 1))).clip(0.7, 1.3)
    Sigma = row_scale * sigma_col[None, :]  # (N,M)
    W_in = 1.0 / (Sigma**2)

    # -- row anomalies: replace rows with obvious OOD --
    row_weird = rng.random(N) < row_weird_frac
    x1 = np.linspace(0, 1, M)
    for i in np.where(row_weird)[0]:
        if True:  # rng.random() < 0.5:
            Y_true[i] = 1.0 + 0.25 * np.sin(
                2 * np.pi * 50 * x1 + rng.uniform(0, 2 * np.pi)
            )
        else:
            Y_true[i] = 1.0 + rng.uniform(-0.3, 0.3)

    # -- pixel spikes --
    spike_mask = rng.random((N, M)) < spike_frac
    spikes = np.zeros((N, M), dtype=float)
    spikes[spike_mask] = (
        spike_amp_sigma
        * Sigma[spike_mask]
        * rng.choice([-1.0, 1.0], size=spike_mask.sum())
    )

    # -- draw noise and assemble observed --
    noise = rng.normal(0, 1.0, size=(N, M)) * Sigma
    Y = Y_true + noise + spikes

    # -- bad pixels (weight=0, flux=0) --
    bad_mask = rng.random((N, M)) < bad_frac
    Y = Y.copy()
    W_in = W_in.copy()
    Y[bad_mask] = 0.0
    W_in[bad_mask] = 0.0

    masks = dict(row_weird=row_weird, spike=spike_mask, bad=bad_mask)
    return Y, Y_true, W_in, G_true, masks


# --- metrics used in Stage 5 & Stage 6 ---


def _orthonormal_cols_from_rows(G):
    """Given G (K,M) representing a K-dim row-space, return an orthonormal basis Q (M,K) for that subspace."""
    # column space of G^T equals row space of G
    Q, _ = np.linalg.qr(G.T)  # (M, M)
    return Q[:, : G.shape[0]]


def subspace_gap(G_est, G_true):
    """
    Principal angles between the row-spaces of G_est and G_true.
    Returns mean(sin θ_i) and max(sin θ_i) in [0,1].
    """
    Qe = _orthonormal_cols_from_rows(G_est)
    Qt = _orthonormal_cols_from_rows(G_true)
    s = np.linalg.svd(Qt.T @ Qe, full_matrices=False)[1]  # singular values = cos(θ_i)
    s = np.clip(s, -1.0, 1.0)
    angles = np.arccos(s)
    sins = np.sin(angles)
    return float(np.mean(sins)), float(np.max(sins))


def auc_binary(labels_bool, scores):
    """
    AUC via Mann–Whitney probability: P(score_pos > score_neg) + 0.5*P(tie).
    Works without sklearn; fine for small toy sizes.
    """
    y = np.asarray(labels_bool, dtype=bool)
    s = np.asarray(scores, dtype=float)
    pos = s[y]
    neg = s[~y]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    diff = pos[:, None] - neg[None, :]
    return (np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (pos.size * neg.size)


def rmse_on_mask(Y_true, Yhat, mask):
    d = (Y_true - Yhat)[mask]
    return float(np.sqrt(np.mean(d**2))) if d.size else np.nan


def stage5_checks(Y, Y_true, W_in, G_true, masks, label="Stage 5"):
    # ...
    print(f"[{label}] Mixed anomalies (same tests we’ll use in Stage 6)")
    # Fit both arms
    A0, G0, W0, _ = run_fit(Y, W_in, robust=False, Q=Q)
    A1, G1, W1, _ = run_fit(Y, W_in, robust=True, Q=Q)
    Yhat0, Yhat1 = A0 @ G0, A1 @ G1

    # Clean-entry mask: exclude weird rows, spikes, and bad pixels
    clean_mask = (~masks["row_weird"])[:, None] & (~masks["spike"]) & (~masks["bad"])

    # 1) Clean-entry RMSE (lower is better; with_w should not be worse)
    rm0 = rmse_on_mask(Y_true, Yhat0, clean_mask)
    rm1 = rmse_on_mask(Y_true, Yhat1, clean_mask)
    print(f"  RMSE on clean entries: no_w={rm0:.5f}  with_w={rm1:.5f}")

    # 2) Subspace gap to true basis (smaller is better)
    gap0_mean, gap0_max = subspace_gap(G0, G_true)
    gap1_mean, gap1_max = subspace_gap(G1, G_true)
    print(
        f"  Subspace gap (mean sinθ / max sinθ): no_w={gap0_mean:.3f}/{gap0_max:.3f}   "
        f"with_w={gap1_mean:.3f}/{gap1_max:.3f}"
    )

    # 3) Row-level detection
    mult1_row = (W1 / (W_in + 1e-12)).mean(axis=1)  # row mean multiplier (with_w)
    row_score_resid0 = resid_norms(Y, A0, G0, W_in)  # no_w residuals
    row_score_resid1 = resid_norms(Y, A1, G1, W_in)  # with_w residuals
    row_score_wdrop = 1.0 - mult1_row  # bigger = more downweighted

    auc_row_resid0 = auc_binary(masks["row_weird"], row_score_resid0)
    auc_row_resid1 = auc_binary(masks["row_weird"], row_score_resid1)
    auc_row_wdrop = auc_binary(masks["row_weird"], row_score_wdrop)

    med_mult_norm = float(np.median(mult1_row[~masks["row_weird"]]))
    med_mult_weird = float(np.median(mult1_row[masks["row_weird"]]))
    print(
        f"  Row AUCs — resid(no_w)={auc_row_resid0:.3f}  resid(with_w)={auc_row_resid1:.3f}  "
        f"wdrop(with_w)={auc_row_wdrop:.3f}"
    )
    print(
        f"  Row med multipliers: normals={med_mult_norm:.3f}  weird={med_mult_weird:.3f}  "
        f"margin={med_mult_norm - med_mult_weird:.3f}"
    )

    # 4) Pixel-level detection (exclude bad pixels)
    mult1_pix = W1 / (W_in + 1e-12)
    valid_pix_mask = ~masks["bad"]
    pix_auc = auc_binary(
        masks["spike"][valid_pix_mask], (1.0 - mult1_pix)[valid_pix_mask]
    )
    med_pix_clean = float(np.median(mult1_pix[valid_pix_mask & (~masks["spike"])]))
    med_pix_spike = float(np.median(mult1_pix[valid_pix_mask & (masks["spike"])]))
    print(f"  Pixel AUC — wdrop(with_w)={pix_auc:.3f}")
    print(
        f"  Pixel med multipliers: clean={med_pix_clean:.3f}  spikes={med_pix_spike:.3f}  "
        f"margin={med_pix_clean - med_pix_spike:.3f}"
    )

    # Tiny visual: row multipliers + pixel multipliers heatmap
    fig, ax = plt.subplots(1, 2, figsize=(9, 3), dpi=130, layout="compressed")
    ax[0].hist(mult1_row[~masks["row_weird"]], bins=20, alpha=0.7, label="normal")
    ax[0].hist(mult1_row[masks["row_weird"]], bins=20, alpha=0.7, label="weird")
    ax[0].set_title(f"{label}: row mean multiplier")
    ax[0].legend()
    im = ax[1].imshow(mult1_pix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax[1].set_title(f"{label}: W_after / W_in")
    ax[1].set_xlabel("pixel")
    ax[1].set_ylabel("row")
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    plt.show()

    print(f"  PASS ({label} ran; inspect numbers for improvement with_w vs no_w) ✅")


# Stage 6
# =========================


def _lam_grid(M, lam_min=4000.0, lam_max=6800.0):
    lam = np.geomspace(lam_min, lam_max, M)  # uniform in log-lambda
    x = np.log(lam)
    dx = x[1] - x[0]
    return lam, x, dx


def _linelist_basic():
    # a few strong-ish lines (Balmer + Mg b + Na D-ish)
    return np.array([4101.7, 4340.5, 4861.3, 5167.3, 5172.7, 5183.6, 5890.0, 5896.0])


def _lsf_kernel_sigma_pix(sigma_pix, half_width=6):
    hw = max(1, int(np.ceil(half_width * sigma_pix)))
    t = np.arange(-hw, hw + 1, dtype=float)
    k = np.exp(-0.5 * (t / sigma_pix) ** 2)
    k /= k.sum()
    return k


def _convolve_rows(Y, k):
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), -1, Y)


def _build_spectral_basis(M, K_extra=0, R=None):
    """Return G_true (K x M) with continuum (1,z,z^2) + line template T0 (+ derivs if R is not None) + two masks."""
    lam, x, dx = _lam_grid(M)
    # Continuum (Legendre on [-1,1])
    z = 2 * (x - x.min()) / (x.max() - x.min()) - 1.0
    L0 = np.ones_like(z)
    L1 = z
    L2 = 0.5 * (3 * z**2 - 1)

    # Line template
    ll = _linelist_basic()
    T0 = np.zeros(M)
    for c in ll:
        T0 += -0.45 * np.exp(-0.5 * ((lam - c) / 0.9) ** 2)

    # Optional LSF
    if R is not None:
        sigma_lam = lam.mean() / R  # Δλ FWHM / λ ≈ 1/R
        sigma_loglam = sigma_lam / lam.mean()
        sigma_pix = sigma_loglam / dx
        k = _lsf_kernel_sigma_pix(sigma_pix)
        T0 = _convolve_rows(T0[None, :], k)[0]

    # Derivatives (for RV linearization)
    T1 = np.gradient(T0, dx)
    T2 = np.gradient(T1, dx)

    G_rows = [L0, L1, L2, T0]
    if R is not None:
        G_rows += [T1, T2]  # allow small shifts

    # Optional little extras to keep K similar to your toy
    for j in range(K_extra):
        G_rows.append(
            np.sin(2 * np.pi * (j + 2) * (lam - lam.min()) / (lam.max() - lam.min()))
        )
    G = np.stack(G_rows, axis=0)
    return G, lam


def _expected_clean_median_multiplier(W_in, Y, A, G, Q):
    """For clean data: estimate sigma_r from standardized residuals and return
    expected median multiplier Q^2/(Q^2 + sigma_r^2 * median(r^2))."""
    R = np.sqrt(W_in) * (Y - A @ G)
    r = R.ravel()
    med = np.median(r)
    mad = np.median(np.abs(r - med)) + 1e-12
    sigma_r = 1.4826 * mad  # robust ~ std if Normal
    r_med_sq = 0.67448975**2  # median of r^2 for N(0,1)
    return (Q**2) / (Q**2 + (sigma_r**2) * r_med_sq), sigma_r


# ---- Stage 6 generator with incremental realism ----
def generate_stage6(
    N=N,
    M=M,
    seed=SEED,
    step="6a_lines",
    row_weird_frac=0.10,  # for mixed anomalies
    spike_frac=0.01,
    spike_amp_sigma=10.0,
    bad_frac=0.003,
    R=4000.0,  # resolving power when LSF is used
    rv_sigma_kms=20.0,  # small RV scatter
    poisson_alpha=0.0,  # set >0 in 6e
):
    rng = np.random.default_rng(seed)

    # ---- basis & coefficients
    use_lsf = step in ["6c_lsf", "6d_rv", "6e_poisson", "6f_mixed"]
    include_rv = step in ["6d_rv", "6e_poisson", "6f_mixed"]
    G_true, lam = _build_spectral_basis(M, K_extra=0, R=(R if use_lsf else None))
    K = G_true.shape[0]
    A_true = rng.normal(0, 1, size=(N, K))

    # --- keep derivatives quiet in 6c (no RV) ---
    if step == "6c_lsf":
        # Use a realistic, positive line amplitude and *no* arbitrary T1/T2 mix.
        aT0 = np.abs(1.0 + 0.2 * rng.standard_normal(N))
        A_true[:, 3] = aT0  # T0 amplitude
        A_true[:, 4] *= 0.0  # T1 off
        A_true[:, 5] *= 0.0  # T2 off

    # If RV enabled, encode small shift via T1/T2 coefficients
    if include_rv:
        c_kms = 299792.458
        rv = rv_sigma_kms * rng.standard_normal(N)
        dx = np.log(lam[1]) - np.log(lam[0])
        delta_x = rv / c_kms
        # scale T1,T2 contributions relative to T0 coefficient
        # A rows: [L0 L1 L2 T0 T1 T2]
        aT0 = np.abs(1.0 + 0.2 * rng.standard_normal(N))
        A_true[:, 3] = aT0
        A_true[:, 4] = -aT0 * delta_x
        A_true[:, 5] = 0.5 * aT0 * (delta_x**2)

    Y_true = A_true @ G_true

    # ---- noise model
    # 6a: homoskedastic; 6b+: blaze-like hetero in λ
    if step in ["6a_lines"]:
        Sigma = 0.00001 * np.ones((N, M))
    else:
        x01 = (lam - lam.min()) / (lam.max() - lam.min())
        sigma_col = 0.001 * (1.0 + 0.6 * (x01 - 0.5) ** 2)  # blaze
        row_scale = (1.0 + 0.05 * rng.standard_normal((N, 1))).clip(0.7, 1.3)
        Sigma = row_scale * sigma_col[None, :]

    # 6e+: add Poisson-like component (and keep W_in calibrated)
    if step in ["6e_poisson", "6f_mixed"] and poisson_alpha <= 0.0:
        poisson_alpha = 0.00002
    if step in ["6e_poisson", "6f_mixed"]:
        Sigma = np.sqrt(Sigma**2 + (poisson_alpha**2) * np.clip(Y_true, 1e-6, None))

    noise = rng.normal(0, 1.0, size=(N, M)) * Sigma
    Y = Y_true + noise
    W_in = 1.0 / (Sigma**2)

    # ---- anomalies (only in 6f_mixed)
    masks = dict(
        row_weird=np.zeros(N, bool),
        spike=np.zeros((N, M), bool),
        bad=np.zeros((N, M), bool),
    )
    if step == "6f_mixed":
        # row ripples (true OOD)
        idx = rng.choice(N, size=max(6, int(0.5 + row_weird_frac * N)), replace=False)
        x01 = (lam - lam.min()) / (lam.max() - lam.min())
        for i in idx:
            Y_true[i] = 1.0 + 0.25 * np.sin(
                2 * np.pi * 50 * x01 + rng.uniform(0, 2 * np.pi)
            )
        Y = (
            Y_true + rng.normal(0, 1.0, size=(N, M)) * Sigma
        )  # redraw noise after replacing truth
        masks["row_weird"][idx] = True

        # pixel spikes
        spike_mask = rng.random((N, M)) < spike_frac
        spikes = spike_amp_sigma * Sigma * rng.choice([-1.0, 1.0], size=(N, M))
        Y += np.where(spike_mask, spikes, 0.0)
        masks["spike"] = spike_mask

        # a few bad pixels
        bad_mask = rng.random((N, M)) < bad_frac
        Y[bad_mask] = 0.0
        W_in[bad_mask] = 0.0
        masks["bad"] = bad_mask

    return Y, Y_true, W_in, G_true, masks


def _plot_zoom_around_lines(
    step, Y, Y_true, Yhat0, Yhat1, lam_ang, line_centers, n_rows=3, seed=123
):
    """
    Show continuum-removed zooms around specified line centers.
    lam_ang: (M,) wavelength array in Angstrom (same grid used to generate data)
    line_centers: sequence of wavelengths (Angstrom) to zoom around
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(Y.shape[0], size=n_rows, replace=False)

    # crude continuum: LS onto Legendre {1,z,z^2} in pixel space
    M = Y.shape[1]
    x = np.linspace(-1, 1, M)
    B = np.vstack([np.ones(M), x, 0.5 * (3 * x**2 - 1)]).T  # (M,3)
    BtB_inv = np.linalg.pinv(B.T @ B)

    def cont(y):
        return B @ (BtB_inv @ (B.T @ y))

    fig, axes = plt.subplots(
        n_rows, len(line_centers), figsize=(12, 2.6 * n_rows), dpi=130
    )
    if n_rows == 1:
        axes = axes[None, :]

    for r, i in enumerate(idx):
        y = Y[i]
        yt = Y_true[i]
        y0 = Yhat0[i]
        y1 = Yhat1[i]
        yc, ytc, y0c, y1c = y - cont(y), yt - cont(yt), y0 - cont(y0), y1 - cont(y1)

        for c, (ax, lc) in enumerate(zip(axes[r], line_centers)):
            # nearest pixel to this line center
            j = int(np.argmin(np.abs(lam_ang - lc)))
            lo, hi = max(0, j - 8), min(M, j + 9)
            xx = np.arange(lo, hi)

            ax.plot(
                xx,
                yc[lo:hi],
                color="k",
                lw=1.0,
                label="obs-ctm" if (r == 0 and c == 0) else None,
            )
            ax.plot(
                xx,
                ytc[lo:hi],
                color="C0",
                lw=2,
                alpha=0.7,
                label="true-ctm" if (r == 0 and c == 0) else None,
            )
            ax.plot(
                xx,
                y0c[lo:hi],
                color="C1",
                ls="--",
                label="no_w-ctm" if (r == 0 and c == 0) else None,
            )
            ax.plot(
                xx,
                y1c[lo:hi],
                color="C2",
                ls="--",
                label="with_w-ctm" if (r == 0 and c == 0) else None,
            )
            ax.axvline(j, color="0.8", lw=0.5)
            if r == 0:
                ax.set_title(f"λ≈{int(lc)}Å")
            if r == n_rows - 1:
                ax.set_xlabel("pixel")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle(f"Stage {step}: continuum-removed zooms")
    plt.tight_layout()
    plt.show()


def inspect_stage6_data(step, Y, Y_true, W_in, A0, G0, A1, G1, seed=123):
    """Visual inspection: 5 spectra with truth & reconstructions, plus residuals and SVD spectrum."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(Y.shape[0], size=5, replace=False)

    Yhat0 = A0 @ G0
    Yhat1 = A1 @ G1

    fig, axs = plt.subplots(
        5, 2, figsize=(12, 10), gridspec_kw={"width_ratios": [3, 1]}, dpi=130
    )
    lam = np.arange(Y.shape[1])  # pixel index as x-axis

    for k, i in enumerate(idx):
        # left: spectra
        ax = axs[k, 0]
        ax.plot(lam, Y[i], color="black", lw=1.0, label="observed")
        ax.plot(lam, Y_true[i], color="C0", lw=2, alpha=0.7, label="true")
        ax.plot(lam, Yhat0[i], color="C1", ls="--", label="no_w recon")
        ax.plot(lam, Yhat1[i], color="C2", ls="--", label="with_w recon")
        ax.set_ylabel(f"row {i}")
        if k == 0:
            ax.legend(fontsize=8, loc="best")

        # right: residuals histogram
        r0 = np.sqrt(W_in[i]) * (Y[i] - Yhat0[i])
        r1 = np.sqrt(W_in[i]) * (Y[i] - Yhat1[i])
        axs[k, 1].hist(r0, bins=30, alpha=0.6, label="no_w")
        axs[k, 1].hist(r1, bins=30, alpha=0.6, label="with_w")
        if k == 0:
            axs[k, 1].legend(fontsize=8, loc="best")
        axs[k, 1].set_title("standardized residuals")

    axs[-1, 0].set_xlabel("pixel")
    axs[-1, 1].set_xlabel("r (chi units)")
    fig.suptitle(f"Stage {step}: sample spectra & residuals", y=1.02)
    plt.tight_layout()
    plt.show()

    # Singular value spectrum of the data matrix
    svals = np.linalg.svd(Y, compute_uv=False)
    plt.figure(figsize=(5, 3), dpi=130)
    plt.semilogy(svals[:20], "o-")
    plt.title(f"Stage {step}: singular values of Y")
    plt.xlabel("index")
    plt.ylabel("σ")
    plt.tight_layout()
    plt.show()

    # Build the wavelength grid used in Stage 6
    lam_ang, _, _ = _lam_grid(Y.shape[1])
    line_centers = _linelist_basic()
    _plot_zoom_around_lines(
        step, Y, Y_true, Yhat0, Yhat1, lam_ang, line_centers, n_rows=3, seed=seed
    )


def run_stage6_suite(steps=None, seed=SEED, Q_in=Q):
    if steps is None:
        steps = ["6a_lines", "6b_blaze", "6c_lsf", "6d_rv", "6e_poisson", "6f_mixed"]

    for step in steps:
        print(f"\n=== Stage {step} (tested with Stage-5 metrics) ===")
        Y, Y_true, W_in, G_true, masks = generate_stage6(step=step, seed=seed)

        # Count anomalies present in this step
        n_rows_weird = int(masks["row_weird"].sum())
        n_pix_spikes = int(masks["spike"].sum())
        n_pix_bad = int(masks["bad"].sum())
        print(
            f"  counts — weird_rows={n_rows_weird}, spike_pixels={n_pix_spikes}, bad_pixels={n_pix_bad}"
        )

        # Fit both arms once here so we can also do clean-calibration checks
        A0, G0, W0, _ = run_fit(Y, W_in, robust=False, Q=Q_in)
        A1, G1, W1, _ = run_fit(Y, W_in, robust=True, Q=Q_in)

        # If this step is clean (no anomalies), do the Stage-4-style calibration sanity check
        if n_rows_weird == 0 and n_pix_spikes == 0 and n_pix_bad == 0:
            exp_med, sigma_r = _expected_clean_median_multiplier(W_in, Y, A1, G1, Q_in)
            mult_med = float(np.median(W1 / (W_in + 1e-12)))
            print(
                f"  clean-check — median(mult)={mult_med:.3f}, expected≈{exp_med:.3f}, sigma_r≈{sigma_r:.3f}"
            )
            # light assertion so we catch egregious miscalibration but don’t fail marginal cases
            assert abs(mult_med - exp_med) < 0.06, (
                "Clean median multiplier off expectation"
            )

        # Now run the *same* Stage-5 evaluation (RMSE, subspace, row/pixel AUCs, plots)
        stage5_checks(Y, Y_true, W_in, G_true, masks, label=f"Stage 6: {step}")

        # NEW: Inspect actual spectra and residuals
        inspect_stage6_data(step, Y, Y_true, W_in, A0, G0, A1, G1)


# -----------------------------
# Run all stages
# -----------------------------
# def main():
# Stage 0
# Y0, Y0_true, W0, G0, masks0 = generate_stage0()
# stage0_checks(Y0, Y0_true, W0)

# # Stage 1
# Y1, Y1_true, W1, G1, masks1 = generate_stage1_row_anoms()
# stage1_checks(Y1, Y1_true, W1, masks1["row_weird"])

# # Stage 2
# Y2, Y2_true, W2, G2, masks2 = generate_stage2_pixel_spikes()
# stage2_checks(Y2, Y2_true, W2, masks2["spike"])

# # Stage 3
# Y3, Y3_true, W3, G3, masks3 = generate_stage3_bad_pixels()
# stage3_checks(Y3, Y3_true, W3, masks3["bad"])

# # Stage 4
# Y4, Y4_true, W4, G4, masks4 = generate_stage4_hetero_calibrated()
# stage4_checks(Y4, Y4_true, W4)


# # ----- Stage 5 -----
# Y5, Y5_true, W5, G5_true, masks5 = generate_stage5_mixed()
# stage5_checks(Y5, Y5_true, W5, G5_true, masks5)

# ----- Stage 6 -----
run_stage6_suite(["6f_mixed"])

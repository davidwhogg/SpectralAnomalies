# -*- coding: utf-8 -*-
"""
Generate a toy stellar-like spectral dataset that is strongly low-rank.

It writes an NPZ with the following keys to match your reader:
  - "observed": (N, M) noisy flux
  - "true":     (N, M) noiseless flux
  - "weights":  (N, M) recommended weights (≈ 1/σ^2)
  - "μ":        (N, M) placeholder mean (zeros)
  - "σ":        (N, M) noise sigma used to draw noise
  - "x":        (M,)   wavelength grid (Angstrom)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Params:
    N: int = 1000  # number of spectra (observations)
    M: int = 500  # number of wavelength pixels (features)
    lam_min: float = 4000.0  # Angstrom
    lam_max: float = 6800.0  # Angstrom
    R: float = 4000.0  # resolving power (lambda / FWHM)
    rv_sigma_kms: float = (
        20.0  # per-star RV scatter (km/s), small -> low-rank via derivative bases
    )
    noise_sigma: float = 0.001  # base noise (fractional)
    seed: int = 42
    out: Path = Path("spectra.npz")


def loglam_grid(M: int, lam_min: float, lam_max: float):
    lam = np.geomspace(lam_min, lam_max, M)  # uniform in log-lambda
    x = np.log(lam)  # natural log
    dx = x[1] - x[0]
    return lam, x, dx


def gaussian_kernel_sigma_pix(sigma_pix: float, half_width: int = 6):
    # Discrete Gaussian kernel with std in pixels; truncate at ±half_width*sigma
    hw = max(1, int(np.ceil(half_width * sigma_pix)))
    t = np.arange(-hw, hw + 1, dtype=float)
    k = np.exp(-0.5 * (t / sigma_pix) ** 2)
    k /= k.sum()
    return k


def convolve_lsf(y: np.ndarray, sigma_pix: float):
    if sigma_pix <= 1e-8:
        return y.copy()
    k = gaussian_kernel_sigma_pix(sigma_pix)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), -1, y)


def make_linelist():
    # A small, recognizable optical linelist (Angstrom)
    # Balmer + Mg b triplet + Na D + a few Fe; clipped by grid later.
    return np.array(
        [
            3933.7,  # Ca K (may be out of range depending on lam_min)
            4101.7,
            4340.5,
            4861.3,
            6562.8,  # Hδ, Hγ, Hβ, Hα
            5167.3,
            5172.7,
            5183.6,  # Mg b
            5270.0,
            5328.0,  # Fe blends (approx)
            5890.0,
            5896.0,  # Na D
        ]
    )


def build_basis(params: Params):
    N, M = params.N, params.M
    lam, x, dx = loglam_grid(M, params.lam_min, params.lam_max)

    # Continuum bases: Legendre on [-1,1]
    z = (
        2
        * (np.log(lam) - np.log(params.lam_min))
        / (np.log(params.lam_max) - np.log(params.lam_min))
    ) - 1.0
    L0 = np.ones_like(z)
    L1 = z
    L2 = 0.5 * (3 * z**2 - 1)
    cont = np.stack([L0, L1, L2], axis=0)  # (3, M)

    # Line template T0: sum of Gaussian absorption dips (baseline zero)
    ll = make_linelist()
    in_range = (ll >= params.lam_min) & (ll <= params.lam_max)
    ll = ll[in_range]
    # Per-line intrinsic widths (Angstrom) (pre-LSF), modest variety
    wA = np.full_like(ll, 0.6)  # simple constant; LSF dominates shape anyway
    depth = np.linspace(
        0.08, 0.20, ll.size
    )  # deeper for redder lines, just for variety

    lam_grid = lam[None, :]  # (1, M)
    T0 = np.zeros((1, M), dtype=float)
    for c, sigA, d in zip(ll, wA, depth):
        T0 += -d * np.exp(-0.5 * ((lam_grid - c) / sigA) ** 2)  # negative dips

    T0 = T0[0]
    # Convolve with instrument LSF: sigma_x = 1/(2.355*R) in log-lambda
    sigma_x = 1.0 / (2.355 * params.R)
    sigma_pix = sigma_x / dx
    T0 = convolve_lsf(T0[None, :], sigma_pix)[0]

    # Derivative bases wrt log-λ (for small RV shifts)
    # On log grid, a velocity v corresponds to shift δx ≈ v/c (c in same units).
    T1 = np.gradient(T0, dx)  # first derivative
    T2 = np.gradient(T1, dx)  # second derivative

    # Line-depth "metallicity" sub-templates: split lines into two groups
    mask1 = np.zeros_like(T0)
    mask2 = np.zeros_like(T0)
    mid = 5200.0
    for c in ll:
        if c < mid:
            mask1 += np.exp(-0.5 * ((lam - c) / 1.2) ** 2)
        else:
            mask2 += np.exp(-0.5 * ((lam - c) / 1.2) ** 2)
    # Normalize masks to [0,1] on line regions
    mask1 = mask1 / (mask1.max() + 1e-8)
    mask2 = mask2 / (mask2.max() + 1e-8)

    D1 = T0 * mask1  # emphasize blue-side lines
    D2 = T0 * mask2  # emphasize red-side lines

    # Stack basis rows G: shape (K, M)
    G = np.stack([cont[0], cont[1], cont[2], T0, T1, T2, D1, D2], axis=0)
    return G, lam, x, dx, sigma_pix


def sample_coeffs(params: Params, G):
    rng = np.random.default_rng(params.seed)
    N, M = params.N, params.M
    K = G.shape[0]

    # Continuum coefficients per star: around unity with gentle slopes
    c0 = 1.0 + 0.1 * rng.standard_normal((N, 1))
    c1 = 0.1 * rng.standard_normal((N, 1))
    c2 = 0.05 * rng.standard_normal((N, 1))

    # Line strength ("metallicity-ish") around 1, positive
    s = np.abs(1.0 + 0.2 * rng.standard_normal((N, 1)))

    # Small RV in log-λ units: δx ≈ v/c  (c ≈ 299792.458 km/s)
    c_kms = 299792.458
    rv = params.rv_sigma_kms * rng.standard_normal((N, 1))
    delta_x = rv / c_kms

    # Coeffs on derivative bases for small shift expansion
    # T(x - δ) ≈ T0 - δ T1 + 0.5 δ^2 T2
    a_T0 = s
    a_T1 = -s * delta_x
    a_T2 = 0.5 * s * (delta_x**2)

    # "Metallicity" sub-templates (allow some independent variation)
    m1 = 0.5 * rng.standard_normal((N, 1))
    m2 = 0.5 * rng.standard_normal((N, 1))

    # Stack A to match G rows: [L0, L1, L2, T0, T1, T2, D1, D2]
    A = np.concatenate([c0, c1, c2, a_T0, a_T1, a_T2, m1, m2], axis=1)  # (N, K)

    # Optional tiny extra jitter to prevent perfect rank if desired
    A += 0.00 * rng.standard_normal(A.shape)

    return A, rv.squeeze(-1)


def hetero_noise_sigma(params: Params, lam):
    # Heteroscedastic noise: modestly worse in the blue
    lam_mid = 0.5 * (params.lam_min + params.lam_max)
    shape = 1.0 + 0.3 * ((lam - lam_mid) / (params.lam_max - params.lam_min)) ** 2
    return params.noise_sigma * shape  # (M,)


def generate(params: Params):
    G, lam, x, dx, sigma_pix = build_basis(params)
    A, rv = sample_coeffs(params, G)

    Y_true = A @ G  # (N, M)
    sig_col = hetero_noise_sigma(params, lam)  # (M,)
    # Broadcast per-column sigma to per-pixel sigma; add per-row scaling jitter
    rng = np.random.default_rng(params.seed + 1)
    row_scale = (1.0 + 0.05 * rng.standard_normal((params.N, 1))).clip(0.8, 1.2)
    Sigma = row_scale * sig_col[None, :]  # (N, M)

    noise = rng.standard_normal((params.N, params.M)) * Sigma
    Y_obs = Y_true + noise

    # Weights as inverse variance
    W = 1.0 / (Sigma**2)

    # μ placeholder (zeros), σ actual noise stdev used
    Mu = np.zeros_like(Y_obs)
    out = {
        "observed": Y_obs.astype(np.float32),
        "true": Y_true.astype(np.float32),
        "weights": W.astype(np.float32),
        "μ": Mu.astype(np.float32),
        "σ": Sigma.astype(np.float32),
        "x": lam.astype(np.float32),
    }
    return out


def main():
    p = Params()
    data = generate(p)

    # Plot all spectra
    _, ax = plt.subplots(figsize=(8, 4), layout="compressed")
    ax.pcolormesh(data["observed"].T, vmin=0, cmap="Blues")
    ax.set_xlabel("Spectrum")
    ax.set_ylabel("Pixel")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    np.savez(p.out, **data)

    # Plot some random spectra
    rng = np.random.default_rng(p.seed)
    _, ax = plt.subplots(figsize=(8, 6), layout="compressed")
    for i in rng.choice(p.N, size=5, replace=False):
        ax.plot(data["x"], data["observed"][i], label=f"Spectrum {i}")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")
    ax.legend()
    plt.show()

    print(f"Wrote {p.out} with keys: {list(data.keys())}")
    for k, v in data.items():
        print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    main()

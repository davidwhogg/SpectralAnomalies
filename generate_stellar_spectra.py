# -*- coding: utf-8 -*-
"""
Generate a toy stellar-like spectral dataset that is strongly low-rank.

Writes an NPZ with keys compatible with your reader PLUS extra "truth" arrays:
  - observed: (N, M) noisy flux
  - true:     (N, M) noiseless flux
  - weights:  (N, M) inverse variance
  - μ:        (N, M) zeros
  - σ:        (N, M) noise sigma per pixel
  - x:        (M,)   wavelength grid (Angstrom)

Truth / metadata:
  - A_true:       (N, K) per-spectrum coefficients in the basis rows of G_basis
  - G_basis:      (K, M) basis matrix used to construct spectra
  - basis_names:  (K,)   unicode strings naming each row of G_basis
  - rv_kms:       (N,)   drawn radial velocities (km/s)
  - row_noise_scale: (N,) row-wise noise scale factor used
  - sigma_col:    (M,)   column-wise base sigma before row scaling
  - params_json:  ()     json string of generator parameters
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Params:
    N: int = 200  # number of spectra (observations)
    M: int = 500  # number of wavelength pixels (features)
    lam_min: float = 4000.0  # Angstrom
    lam_max: float = 6800.0  # Angstrom
    R: float = 4000.0  # resolving power (lambda / FWHM)
    rv_sigma_kms: float = 20.0  # per-star RV scatter (km/s)
    noise_sigma: float = 0.01  # base noise (fractional)
    seed: int = 42
    out: Path = Path("spectra.npz").name


def loglam_grid(M: int, lam_min: float, lam_max: float):
    lam = np.geomspace(lam_min, lam_max, M)  # uniform in log-lambda
    x = np.log(lam)  # natural log
    dx = x[1] - x[0]
    return lam, x, dx


def gaussian_kernel_sigma_pix(sigma_pix: float, half_width: int = 6):
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
    return np.array(
        [
            4101.7,
            4340.5,
            4861.3,
            6562.8,  # Balmer Hδ, Hγ, Hβ, Hα
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

    # Line template T0
    ll = make_linelist()
    lam_grid = lam[None, :]
    T0 = np.zeros((1, M), dtype=float)
    wA = np.full(ll.shape, 0.6)
    depth = np.linspace(0.08, 0.20, ll.size)
    for c, sigA, d in zip(ll, wA, depth):
        T0 += -d * np.exp(-0.5 * ((lam_grid - c) / sigA) ** 2)
    T0 = T0[0]

    # Instrument LSF on log-lambda
    sigma_x = 1.0 / (2.355 * params.R)
    _, _, dx = loglam_grid(M, params.lam_min, params.lam_max)
    sigma_pix = sigma_x / dx
    T0 = convolve_lsf(T0[None, :], sigma_pix)[0]

    # Derivatives
    T1 = np.gradient(T0, dx)
    T2 = np.gradient(T1, dx)

    # Line-group depth modifiers
    mask1 = np.zeros_like(T0)
    mask2 = np.zeros_like(T0)
    mid = 5200.0
    for c in ll:
        if c < mid:
            mask1 += np.exp(-0.5 * ((lam - c) / 1.2) ** 2)
        else:
            mask2 += np.exp(-0.5 * ((lam - c) / 1.2) ** 2)
    mask1 = mask1 / (mask1.max() + 1e-8)
    mask2 = mask2 / (mask2.max() + 1e-8)
    D1 = T0 * mask1
    D2 = T0 * mask2

    G = np.stack([cont[0], cont[1], cont[2], T0, T1, T2, D1, D2], axis=0)
    names = np.array(["L0", "L1", "L2", "T0", "T1", "T2", "D1", "D2"], dtype="U2")
    return G, names, lam, dx


def sample_coeffs(params: Params, K: int):
    rng = np.random.default_rng(params.seed)
    N = params.N

    c0 = 1.0 + 0.1 * rng.standard_normal((N, 1))
    c1 = 0.1 * rng.standard_normal((N, 1))
    c2 = 0.05 * rng.standard_normal((N, 1))

    s = np.abs(1.0 + 0.2 * rng.standard_normal((N, 1)))

    c_kms = 299792.458
    rv = params.rv_sigma_kms * rng.standard_normal((N, 1))
    delta_x = rv / c_kms

    a_T0 = s
    a_T1 = -s * delta_x
    a_T2 = 0.5 * s * (delta_x**2)

    m1 = 0.5 * rng.standard_normal((N, 1))
    m2 = 0.5 * rng.standard_normal((N, 1))

    A = np.concatenate([c0, c1, c2, a_T0, a_T1, a_T2, m1, m2], axis=1)
    assert A.shape[1] == K
    return A, rv.squeeze(-1)


def hetero_noise_sigma(params: Params, lam):
    lam_mid = 0.5 * (params.lam_min + params.lam_max)
    shape = 1.0 + 0.3 * ((lam - lam_mid) / (params.lam_max - params.lam_min)) ** 2
    return params.noise_sigma * shape


def generate(params: Params):
    G, names, lam, dx = build_basis(params)
    K = G.shape[0]
    A, rv = sample_coeffs(params, K)

    Y_true = A @ G

    sig_col = hetero_noise_sigma(params, lam)  # (M,)
    rng = np.random.default_rng(params.seed + 1)
    row_scale = (1.0 + 0.05 * rng.standard_normal((params.N, 1))).clip(0.8, 1.2)
    Sigma = row_scale * sig_col[None, :]

    noise = rng.standard_normal((params.N, params.M)) * Sigma
    Y_obs = Y_true + noise
    W = 1.0 / (Sigma**2)
    Mu = np.zeros_like(Y_obs)

    meta = {
        "A_true": A.astype(np.float32),
        "G_basis": G.astype(np.float32),
        "basis_names": names,
        "rv_kms": rv.astype(np.float32),
        "row_noise_scale": row_scale.squeeze(1).astype(np.float32),
        "sigma_col": sig_col.astype(np.float32),
        "params_json": json.dumps(asdict(params)),
    }

    out = {
        "observed": Y_obs.astype(np.float32),
        "true": Y_true.astype(np.float32),
        "weights": W.astype(np.float32),
        "μ": Mu.astype(np.float32),
        "σ": Sigma.astype(np.float32),
        "x": lam.astype(np.float32),
        **meta,
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

    np.savez(p.out, **data)
    print(f"Wrote {p.out} with keys: {list(data.keys())}")
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape} {v.dtype}")
        else:
            print(f"  {k}: (metadata)")


if __name__ == "__main__":
    main()

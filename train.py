from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.style.use("mpl_drip.custom")

# SPECTRA = Path("spectra.npz")
# SPECTRA = Path("spectra_outliers.npz")
SPECTRA = Path("spectra_anomalies.npz")


def read_data(spectra: Path):
    file = jnp.load(spectra)
    return (
        file["observed"],
        file["true"],
        file["weights"],
        file["x"],
    )


def initial(Y):
    U, Σ, VH = jnp.linalg.svd(Y, compute_uv=True, full_matrices=False)
    A = U * Σ
    G = VH
    return A, G


# def a_step(G, Y, W):
#     # G: (K, M); Y, W: (N, M)
#     A_ = jnp.einsum("kj,ij,lj->ikl", G, W, G)  # (N, K, K) where we will batch over i
#     b_ = jnp.einsum("kj,ij,ij->ik", G, W, Y)  # (N, K)
#     return jnp.linalg.solve(A_, b_[..., None]).squeeze(-1)  # (N,K)


# def g_step(A, Y, W):
#     # A: (N, K); Y, W: (N, M)
#     A_ = jnp.einsum("ik,ij,il->jkl", A, W, A)  # (M, K, K) where we will batch over j
#     b_ = jnp.einsum("ik,ij,ij->jk", A, W, Y)  # (M, K)
#     return jnp.linalg.solve(A_, b_[..., None]).squeeze(-1).T  # (M, K)


def a_step(G, Y, W, rcond=1e-8):
    # Solve, for each row i:  min || diag(sqrt(w_i)) (y_i^T - G^T a_i^T) ||
    GT = G.T  # (M,K)

    def solve_row(y_i, w_i):
        D = jnp.sqrt(w_i)[:, None]  # (M,1)
        X = D * GT  # (M,K)
        y = (D.squeeze()) * y_i  # (M,)
        a_i, *_ = jnp.linalg.lstsq(X, y, rcond=rcond)  # (K,)
        return a_i

    return jax.vmap(solve_row, in_axes=(0, 0))(Y, W)  # (N,K)


def g_step(A, Y, W, rcond=1e-8):
    # Solve, for each column j: min || diag(sqrt(w_·j)) (y_·j - A g_·j) ||
    def solve_col(y_j, w_j):
        D = jnp.sqrt(w_j)[:, None]  # (N,1)
        X = D * A  # (N,K)
        y = (D.squeeze()) * y_j  # (N,)
        g_j, *_ = jnp.linalg.lstsq(X, y, rcond=rcond)  # (K,)
        return g_j

    G_cols = jax.vmap(solve_col, in_axes=(1, 1))(Y, W)  # (M,K)
    return G_cols.T


def w_step(A, G, Y, W_in, Q=1e5):
    Δ2 = (Y - A @ G) ** 2
    return W_in * Q**2 / (W_in * Δ2 + Q**2)


def reorient(A, G):
    L = A @ G
    return initial(L)


# @jax.jit
def iteration(A, G, Y, W, W_in):
    A_ = a_step(G, Y, W)
    G_ = g_step(A_, Y, W)
    ΔG = jnp.linalg.matrix_norm(G_ - G)
    A_, G_ = reorient(A_, G_)
    # W_ = w_step(A_, G_, Y, W_in)
    W_ = W
    return A_, G_, W_, ΔG


def _chk(A, G, Y, W, tag):
    N, M = Y.shape
    K = A.shape[1]
    assert A.shape == (N, K), f"{tag}: A {A.shape} != {(N, K)}"
    assert G.shape == (K, M), f"{tag}: G {G.shape} != {(K, M)}"
    assert W.shape == (N, M), f"{tag}: W {W.shape} != {(N, M)}"
    # quick residual + gram SPD sanity
    R = Y - A @ G
    print(f"{tag}: ||R||_F={jnp.linalg.norm(R):.3g}, K={K}, N={N}, M={M}")


def plot_AGYW(A, G, Y, W, show=True):
    fig, ax = plt.subplots(1, 4, figsize=(10, 4), layout="compressed", dpi=200)
    ax[0].imshow(A, cmap="PiYG", vmin=-1, vmax=1)
    ax[1].imshow(G, cmap="PiYG", vmin=-1, vmax=1)
    ax[2].imshow(Y, cmap="PiYG", vmin=-1, vmax=1)
    ax[3].imshow(W, cmap="PiYG", vmin=-1, vmax=1)
    for ax_ in ax.flatten():
        ax_.set_xticks([])
        ax_.set_yticks([])
        ax_.set_aspect(1)
    plt.show()


Y, true, W, *_ = read_data(SPECTRA)
W_in = deepcopy(W)


A, G = initial(Y)
N_ITERATION = 30
ΔGs = []
for _ in tqdm(range(N_ITERATION)):
    A, G, W, ΔG = iteration(A, G, Y, W, W_in)
    ΔGs.append(ΔG)

plt.figure(figsize=(12, 8))
plt.plot(ΔGs)
plt.show()

plt.figure(figsize=(12, 8))
for i in range(10):
    plt.subplot(5, 2, i + 1)
    plt.hist(A[:, i], bins=30, color="C0", alpha=0.7)
    plt.title(f"Component {i + 1}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# plt.scatter(A[:, 0], A[:, 1])
# plt.xlabel("Component 1")
# plt.ylabel("Component 2")
# plt.title("Observations in reduced space")
# plt.show()

# plt.scatter(A[:, 2], A[:, 3])
# plt.xlabel("Component 2")
# plt.ylabel("Component 3")
# plt.title("Observations in reduced space")
# plt.show()

# Compare 5 random spectra to their reconstructed ones using only M basis functions
M = 8
fig, axs = plt.subplots(
    2,
    1,
    figsize=(20, 10),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]},
    layout="compressed",
)

# Plot observed and reconstructed spectra
for i in range(5):
    i += 0
    axs[0].plot(Y[i], label="Observed", c=f"C{i}", lw=3.5)
    axs[0].plot(
        A[i, :M] @ G[:M, :], label=f"Reconstructed (M={M})", linestyle="-", c="k"
    )
axs[0].set_ylabel("Flux")
axs[0].legend(loc="best")
axs[0].set_ylim(0.6, 1.2)

# Plot residuals
for i in range(5):
    i += 0
    axs[1].plot(
        Y[i] - (A[i, :M] @ G[:M, :]), label=f"Residual (M={M})", c=f"C{i}", lw=3.5
    )
axs[1].set_xlabel("Wavelength")
axs[1].set_ylabel("Flux")
axs[1].set_ylim(-0.04, 0.04)

plt.show()

plt.figure(figsize=(12, 8), layout="compressed")
plt.imshow(W - W_in, cmap="RdBu")
plt.colorbar()
plt.show()

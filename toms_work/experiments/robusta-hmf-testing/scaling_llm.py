# benchmark_als_sgd.py
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from robusta_hmf import (
    ALS_RHMF,
    SGD_RHMF,
    FastAffine,
    GaussianLikelihood,
    WeightedAStep,
    WeightedGStep,
)

jax.config.update("jax_enable_x64", True)


# ---------------- utilities ----------------
def make_synthetic(N, D, K, key, noise=1e-2):
    k1, k2, k3 = jax.random.split(key, 3)
    A_true = jax.random.normal(k1, (N, K))
    G_true = jax.random.normal(k2, (D, K))
    Y_true = A_true @ G_true.T
    W_data = jnp.ones_like(Y_true) / noise**2
    Y_true = Y_true + noise * jax.random.normal(k3, (N, D))
    return Y_true, W_data


def rel_recon_error(Y, A, G):
    pred = A @ G.T
    return float(jnp.linalg.norm(Y - pred) / jnp.linalg.norm(Y))


def run_als_once(
    Y,
    W_data,
    K,
    tol=1e-3,
    max_iters=500,
    ridge=1e-6,
    whiten=True,
    key=jax.random.PRNGKey(0),
):
    N, D = Y.shape
    als = ALS_RHMF(
        likelihood=GaussianLikelihood(),
        a_step=WeightedAStep(ridge=ridge),
        g_step=WeightedGStep(ridge=ridge),
        rotation=FastAffine(whiten=whiten, eps=1e-6),
        regulariser=None,
    )
    state = als.init_state(N, D, K, key)

    # No warmup: includes compile
    t0 = time.perf_counter()
    it = 0
    err = rel_recon_error(Y, state.A, state.G)
    while err > tol and it < max_iters:
        state, _ = als.step(Y, W_data, state)
        err = rel_recon_error(Y, state.A, state.G)
        it += 1
    t_end_including_compile = time.perf_counter() - t0

    # Warm start timing: re-init (to avoid bias), warm one step to compile, then time fresh run
    state = als.init_state(N, D, K, key)
    _ = als.step(Y, W_data, state)  # warm compile
    state = als.init_state(N, D, K, key)  # fresh start after compile
    t1 = time.perf_counter()
    it2 = 0
    err2 = rel_recon_error(Y, state.A, state.G)
    while err2 > tol and it2 < max_iters:
        state, _ = als.step(Y, W_data, state)
        err2 = rel_recon_error(Y, state.A, state.G)
        it2 += 1
    t_end_after_warm = time.perf_counter() - t1

    return {
        "time_incl_compile": t_end_including_compile,
        "time_after_warm": t_end_after_warm,
        "iters": it,
        "iters_after_warm": it2,
        "final_err": err,
    }


def run_sgd_once(X, K, tol=1e-3, max_iters=5000, lr=1e-2, key=jax.random.PRNGKey(0)):
    N, D = X.shape
    opt = optax.adam(lr)
    sgd = SGD_RHMF(likelihood=GaussianLikelihood(), opt=opt, regularizer=None)
    state = sgd.init_state(N, D, K, key)

    # No warmup: includes compile
    t0 = time.perf_counter()
    it = 0
    err = rel_recon_error(X, state.A, state.G)
    while err > tol and it < max_iters:
        state, _ = sgd.step(X, state)
        err = rel_recon_error(X, state.A, state.G)
        it += 1
    t_end_including_compile = time.perf_counter() - t0

    # Warm start timing
    state = sgd.init_state(N, D, K, key)
    _ = sgd.step(X, state)  # compile
    state = sgd.init_state(N, D, K, key)
    t1 = time.perf_counter()
    it2 = 0
    err2 = rel_recon_error(X, state.A, state.G)
    while err2 > tol and it2 < max_iters:
        state, _ = sgd.step(X, state)
        err2 = rel_recon_error(X, state.A, state.G)
        it2 += 1
    t_end_after_warm = time.perf_counter() - t1

    return {
        "time_incl_compile": t_end_including_compile,
        "time_after_warm": t_end_after_warm,
        "iters": it,
        "iters_after_warm": it2,
        "final_err": err,
    }


# ---------------- benchmark driver ----------------
def benchmark_grid(
    K=4,
    sizes_N=(200, 400, 800),
    sizes_D=(200, 400, 800),
    tol=1e-3,
    als_max=500,
    sgd_max=5000,
    sgd_lr=1e-2,
    noise=1e-2,
    seed=0,
):
    key0 = jax.random.PRNGKey(seed)
    num = len(sizes_N) * len(sizes_D)
    grid_keys = jax.random.split(key0, num)

    results = {
        "ALS": {
            "incl": np.zeros((len(sizes_N), len(sizes_D))),
            "warm": np.zeros((len(sizes_N), len(sizes_D))),
            "iters": np.zeros((len(sizes_N), len(sizes_D))),
        },
        "SGD": {
            "incl": np.zeros((len(sizes_N), len(sizes_D))),
            "warm": np.zeros((len(sizes_N), len(sizes_D))),
            "iters": np.zeros((len(sizes_N), len(sizes_D))),
        },
        "meta": {"Ns": sizes_N, "Ds": sizes_D, "K": K, "tol": tol},
    }

    for iN, N in enumerate(sizes_N):
        for iD, D in enumerate(sizes_D):
            flat_idx = iN * len(sizes_D) + iD
            key = grid_keys[flat_idx]
            Y, W_data = make_synthetic(N, D, K, key, noise=noise)

            als_stats = run_als_once(Y, W_data, K, tol=tol, max_iters=als_max, key=key)
            sgd_stats = run_sgd_once(
                X, K, tol=tol, max_iters=sgd_max, lr=sgd_lr, key=key
            )

            results["ALS"]["incl"][iN, iD] = als_stats["time_incl_compile"]
            results["ALS"]["warm"][iN, iD] = als_stats["time_after_warm"]
            results["ALS"]["iters"][iN, iD] = als_stats["iters_after_warm"]

            results["SGD"]["incl"][iN, iD] = sgd_stats["time_incl_compile"]
            results["SGD"]["warm"][iN, iD] = sgd_stats["time_after_warm"]
            results["SGD"]["iters"][iN, iD] = sgd_stats["iters_after_warm"]

            print(
                f"N={N:5d} D={D:5d} | ALS {als_stats['time_after_warm']:.2f}s/{int(als_stats['iters_after_warm'])} iters | "
                f"SGD {sgd_stats['time_after_warm']:.2f}s/{int(sgd_stats['iters_after_warm'])} iters"
            )

    return results


# ---------------- plotting ----------------
def plot_square_scaling(results):
    Ns = np.array(results["meta"]["Ns"])
    Ds = np.array(results["meta"]["Ds"])
    # pick diagonal N==D points
    diag = []
    for n in Ns:
        if n in Ds:
            diag.append((np.where(Ns == n)[0][0], np.where(Ds == n)[0][0], n))
    if not diag:
        return
    idxN, idxD, sizes = zip(*diag)
    als = results["ALS"]["warm"][idxN, idxD]
    sgd = results["SGD"]["warm"][idxN, idxD]

    plt.figure(figsize=(7, 4))
    plt.plot(sizes, als, "o-", label="ALS (time after warmup)")
    plt.plot(sizes, sgd, "o--", label="SGD (time after warmup)")
    plt.xlabel("Problem size (N = D)")
    plt.ylabel("Time to convergence (s)")
    plt.title(
        f"Scaling at fixed K={results['meta']['K']} (tol={results['meta']['tol']})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_heatmaps(results):
    Ns = np.array(results["meta"]["Ns"])
    Ds = np.array(results["meta"]["Ds"])
    A = results["ALS"]["warm"]
    G = results["SGD"]["warm"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    im0 = axes[0].imshow(
        A, origin="lower", aspect="auto", extent=[Ds[0], Ds[-1], Ns[0], Ns[-1]]
    )
    axes[0].set_title("ALS time (s) after warmup")
    axes[0].set_xlabel("D")
    axes[0].set_ylabel("N")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        G, origin="lower", aspect="auto", extent=[Ds[0], Ds[-1], Ns[0], Ns[-1]]
    )
    axes[1].set_title("SGD time (s) after warmup")
    axes[1].set_xlabel("D")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Convergence time heatmaps (K={results['meta']['K']}, tol={results['meta']['tol']})"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ---------------- run ----------------
if __name__ == "__main__":
    # Config
    K = 4
    sizes_N = (200, 400, 800, 2000, 8000, 15000)  # bump up/down as your machine allows
    sizes_D = (200, 400, 800, 2000, 5000)
    tol = 1e-5
    als_max = 5000
    sgd_max = 5000
    sgd_lr = 1e-2
    noise = 0.01

    results = benchmark_grid(
        K=K,
        sizes_N=sizes_N,
        sizes_D=sizes_D,
        tol=tol,
        als_max=als_max,
        sgd_max=sgd_max,
        sgd_lr=sgd_lr,
        noise=noise,
        seed=0,
    )
    plot_square_scaling(results)
    plot_heatmaps(results)

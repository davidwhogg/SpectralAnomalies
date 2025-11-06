import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from collect_data import get_data
from robusta_hmf import Robusta
from robusta_hmf.rhmf_hogg import RHMF

jax.config.update("jax_enable_x64", True)
plt.style.use("mpl_drip.custom")
rng = np.random.default_rng(42)

# Benchmarking config
N_VALS = [500, 1000, 5000, 10000]
K_VALS = [2, 5, 10]

# Model config
ROBUST_SCALE = 2.0
MAX_ITER = 1000
CONV_TOL = 1e-3
# CONV_STRATEGY = "max_frac_G"
CONV_STRATEGY = "rel_frac_loss"
INIT_STRATEGY = "svd"


# ==== Read the GAIA RVS spectra data ====

Y, W, spec_λ, bp_rp, abs_mag_G = get_data(
    thresh_bp_rp=1.2,
    thresh_abs_mag=1.2,
    clip_edge_pix=20,
)
print("\n================================\n")


def benchmark_als(K, Y, W, N):
    model_als = Robusta(
        rank=K,
        method="als",
        robust_scale=ROBUST_SCALE,
        conv_strategy=CONV_STRATEGY,
        conv_tol=CONV_TOL,
        init_strategy=INIT_STRATEGY,
        target="G",
        whiten=True,
    )
    time_start = time.time()
    model_als.fit(
        Y=jnp.copy(Y[:N, :]),
        W=jnp.copy(W[:N, :]),
        max_iter=MAX_ITER,
        conv_check_cadence=1,
    )
    time_end = time.time()
    return time_end - time_start


def benchmark_sgd(K, Y, W, N):
    model_sgd = Robusta(
        rank=K,
        method="sgd",
        robust_scale=ROBUST_SCALE,
        conv_strategy=CONV_STRATEGY,
        conv_tol=CONV_TOL,
        init_strategy=INIT_STRATEGY,
        target="G",
        whiten=True,
    )
    time_start = time.time()
    model_sgd.fit(
        Y=jnp.copy(Y[:N, :]),
        W=jnp.copy(W[:N, :]),
        max_iter=MAX_ITER,
    )
    time_end = time.time()
    return time_end - time_start


def benchmark_hogg(K, Y, W, N):
    model_hogg = RHMF(
        rank=K,
        nsigma=ROBUST_SCALE,
    )
    model_hogg.set_training_data(Y[:N, :], weights=W[:N, :])
    time_start = time.time()
    model_hogg.train(maxiter=MAX_ITER, tol=CONV_TOL)
    time_end = time.time()
    return time_end - time_start


als_times = jnp.zeros((len(N_VALS), len(K_VALS)))
sgd_times = jnp.zeros((len(N_VALS), len(K_VALS)))
hogg_times = jnp.zeros((len(N_VALS), len(K_VALS)))

for i, N in enumerate(N_VALS):
    for j, K in enumerate(K_VALS):
        print(f"Benchmarking N={N}, K={K}...")
        print("  Running Robusta ALS...")
        als_time = benchmark_als(K, Y, W, N)
        jax.block_until_ready(als_time)
        print("  Running Robusta SGD...")
        sgd_time = benchmark_sgd(K, Y, W, N)
        jax.block_until_ready(sgd_time)
        print("  Running Hogg RHMF...")
        hogg_time = benchmark_hogg(K, Y, W, N)
        jax.block_until_ready(hogg_time)
        als_times = als_times.at[i, j].set(als_time)
        sgd_times = sgd_times.at[i, j].set(sgd_time)
        hogg_times = hogg_times.at[i, j].set(hogg_time)
        print(
            f"  Robusta ALS time: {als_time:.2f} s, Robusta SGD time: {sgd_time:.2f} s, Hogg RHMF time: {hogg_time:.2f} s"
        )

print("\nBenchmarking complete!\n")
np.savez(
    "benchmark_results.npz",
    N_VALS=N_VALS,
    K_VALS=K_VALS,
    als_times=np.array(als_times),
    sgd_times=np.array(sgd_times),
    hogg_times=np.array(hogg_times),
)

# Plot the K=2 results
plt.figure(figsize=[8, 5], dpi=100, layout="compressed")
plt.plot(N_VALS, als_times[:, 0], lw=2, marker="o", label="Robusta ALS")
plt.plot(N_VALS, sgd_times[:, 0], lw=2, marker="o", label="Robusta SGD")
plt.plot(N_VALS, hogg_times[:, 0], lw=2, marker="o", label="Hogg RHMF")
plt.xlabel("Number of Spectra (N)")
plt.ylabel("Time (s)")
# plt.xscale("log")
# plt.yscale("log")
plt.title("Benchmarking Results (K=2)")
plt.legend()
plt.show()

# Plot the K=5 results
plt.figure(figsize=[8, 5], dpi=100, layout="compressed")
plt.plot(N_VALS, als_times[:, 1], lw=2, marker="o", label="Robusta ALS")
plt.plot(N_VALS, sgd_times[:, 1], lw=2, marker="o", label="Robusta SGD")
plt.plot(N_VALS, hogg_times[:, 1], lw=2, marker="o", label="Hogg RHMF")
plt.xlabel("Number of Spectra (N)")
plt.ylabel("Time (s)")
# plt.xscale("log")
# plt.yscale("log")
plt.title("Benchmarking Results (K=5)")
plt.legend()
plt.show()

# Plot the K=10 results
plt.figure(figsize=[8, 5], dpi=100, layout="compressed")
plt.plot(N_VALS, als_times[:, 2], lw=2, marker="o", label="Robusta ALS")
plt.plot(N_VALS, sgd_times[:, 2], lw=2, marker="o", label="Robusta SGD")
plt.plot(N_VALS, hogg_times[:, 2], lw=2, marker="o", label="Hogg RHMF")
plt.xlabel("Number of Spectra (N)")
plt.ylabel("Time (s)")
# plt.xscale("log")
# plt.yscale("log")
plt.title("Benchmarking Results (K=10)")
plt.legend()
plt.show()

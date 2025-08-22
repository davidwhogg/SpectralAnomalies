import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

plt.style.use("mpl_drip.custom")

rng = default_rng(seed=0)


def gaussian(x, μ, σ):
    return np.exp(-0.5 * ((x - μ) / σ) ** 2)
    # return (1 / (σ * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - μ) / σ) ** 2)


# Dataset
N_DATA = 500
LINE_SIGMA = 0.02
μ_values = rng.uniform(0 + 2 * LINE_SIGMA, 1 - 2 * LINE_SIGMA, N_DATA)
σ_values = np.repeat(LINE_SIGMA, N_DATA)


# Spectra
N_PIXELS = 200
NOISE_SIGMA = 0.2
x = np.linspace(0, 1, N_PIXELS)
true_spectra = np.array([gaussian(x, μ, σ) for μ, σ in zip(μ_values, σ_values)])
noise = rng.normal(0, NOISE_SIGMA, true_spectra.shape)
observed_spectra = true_spectra + noise

fig, ax = plt.subplots(figsize=(8, 4), layout="compressed")
c = ax.pcolormesh(observed_spectra.T, vmin=0, cmap="Blues")
ax.set_xlabel("Spectrum")
ax.set_ylabel("Pixel")
ax.set_xticks([])
ax.set_yticks([])
plt.show()


np.savez(
    "spectra.npz",
    observed=observed_spectra,
    true=true_spectra,
    μ=μ_values,
    σ=σ_values,
    x=x,
)

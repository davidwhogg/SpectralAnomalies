import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

plt.style.use("mpl_drip.custom")

rng = default_rng(seed=0)


def gaussian(x, μ, σ):
    return np.exp(-0.5 * ((x - μ) / σ) ** 2)
    # return (1 / (σ * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - μ) / σ) ** 2)


# Dataset
N_DATA = 200
LINE_SIGMA = 0.05
peak_values = rng.uniform(0 + 2 * LINE_SIGMA, 1 - 2 * LINE_SIGMA, N_DATA)
σ_values = np.repeat(LINE_SIGMA, N_DATA)


# Spectra
N_PIXELS = 50
NOISE_SIGMA = 0.2
x = np.linspace(0, 1, N_PIXELS)
true_spectra = np.array(
    [p * gaussian(x, 0.5, σ) for p, σ in zip(peak_values, σ_values)]
)
weights = np.ones_like(true_spectra) / NOISE_SIGMA**2
noise = rng.normal(scale=weights**-2)
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
    weights=weights,
    x=x,
)

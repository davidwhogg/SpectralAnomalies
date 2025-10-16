# v0.py
# Fit anything

import matplotlib.pyplot as plt
import numpy as np
from collect import (
    TARGET_ID,
    compute_abs_mag,
    find_source_index,
    nans_mask,
    read_meta,
    read_spectra,
)
from robusta_hmf.rhmf_hogg import RHMF

plt.style.use("mpl_drip.custom")
rng = np.random.default_rng(0)


# TARGET_ID = 1303220968849834112

# Read meta and spectra
source_id, bp_rp, phot_g_mean_mag, parallax = read_meta()
spec_λ, spec_source_id, spec_flux, spec_u_flux = read_spectra()
abs_mag_G = compute_abs_mag(phot_g_mean_mag, parallax)
target_idx = find_source_index(source_id, TARGET_ID)
# target_idx = 56792
target_spec_idx = find_source_index(spec_source_id, TARGET_ID)
print(
    f"Sanity check source IDs: {source_id[target_idx]} {spec_source_id[target_spec_idx]}"
)

# Find things that are similar to the target in colour-magnitude space, in an ellipse
target_bp_rp = bp_rp[target_idx]
target_abs_mag_G = abs_mag_G[target_idx]
THRESH_BP_RP = 0.2
THRESH_ABS_MAG = 0.1
bp_rp_diff = np.abs(bp_rp - target_bp_rp)
abs_mag_diff = np.abs(abs_mag_G - target_abs_mag_G)
ellipse_mask = (bp_rp_diff / THRESH_BP_RP) ** 2 + (
    abs_mag_diff / THRESH_ABS_MAG
) ** 2 < 1
rectangle_mask = (bp_rp_diff < THRESH_BP_RP) & (abs_mag_diff < THRESH_ABS_MAG)
similar_mask = ellipse_mask  # or rectangle_mask
print(f"Found {np.sum(similar_mask)} similar sources")

# Plot the CMD with the target and similar sources highlighted
fig, ax = plt.subplots(dpi=100, figsize=[18, 7], layout="compressed")
ax.scatter(bp_rp, abs_mag_G, s=0.5, alpha=0.05, c="C0", zorder=0)
ax.scatter(
    bp_rp[target_idx],
    abs_mag_G[target_idx],
    s=50,
    alpha=1,
    c="C1",
    label="Target",
    zorder=2,
)
ax.scatter(
    bp_rp[similar_mask],
    abs_mag_G[similar_mask],
    s=5,
    alpha=0.05,
    c="C2",
    label="Similar",
    zorder=1,
)
ax.set_ylabel("Absolute Mag G band")
ax.set_xlabel("BP - RP")
ypad_fac = 11
xpad_fac = 15
# ypad_fac = 2
# xpad_fac = 2
y_ext = THRESH_ABS_MAG * ypad_fac
x_ext = THRESH_BP_RP * xpad_fac
ax.set_ylim(target_abs_mag_G + y_ext, target_abs_mag_G - y_ext)
# ax.set_xlim(target_bp_rp - x_ext, target_bp_rp + x_ext)
ax.set_xlim(-0.5, 3.5)
ax.legend()
plt.show()

# Get the spectra of the similar sources
similar_spec_idxs = np.sort(
    [find_source_index(spec_source_id, sid) for sid in source_id[similar_mask]]
)
similar_specs = spec_flux[similar_spec_idxs]
similar_ids = spec_source_id[similar_spec_idxs]
similar_bp_rp = bp_rp[similar_mask]
similar_abs_mag_G = abs_mag_G[similar_mask]
target_similar_idx = find_source_index(similar_ids, TARGET_ID)


# Plot the target spectrum and some random similar spectra
fig, ax = plt.subplots(figsize=[18, 7], dpi=100, layout="compressed")
ax.plot(
    spec_λ, similar_specs[target_similar_idx], lw=1, alpha=1, c="C3", label="Target"
)
for i in rng.choice(similar_specs.shape[0], size=40, replace=False):
    ax.plot(spec_λ, similar_specs[i], lw=1, alpha=0.5, c="C2", label="Similar")
ax.set_xlabel(r"$\lambda$ [nm]")
ax.set_ylabel("Normalised Flux")
ax.set_xlim(849.5, 850.5)
ax.set_ylim(0.8, 1.00)
ax.set_title(f"Gaia DR3 {TARGET_ID} and similar sources in CM-space")
plt.show()

# Data to fit
Y = similar_specs.copy()
W = 1.0 / spec_u_flux[similar_spec_idxs] ** 2
spec_nans_mask = nans_mask([Y, W])
Y[~spec_nans_mask] = np.nan
W[~spec_nans_mask] = np.nan
Y = np.nan_to_num(Y)
W = np.nan_to_num(W)

rhmf = RHMF(rank=30, nsigma=2.0)
rhmf.set_training_data(Y, weights=W)
rhmf.train(tol=1e-4)

plt.figure()
plt.title("Residuals")
plt.imshow(rhmf.resid(), aspect="auto", cmap="RdBu", vmin=-0.1, vmax=0.1)
plt.colorbar()
plt.show()

plt.figure(dpi=100, figsize=[18, 7], layout="compressed")
plt.title("Weights")
plt.imshow(rhmf.W / rhmf.input_W, aspect="auto", cmap="viridis", vmin=0, vmax=1)
plt.colorbar()
plt.show()

plt.figure(dpi=100, figsize=[18, 7], layout="compressed")
plt.plot(spec_λ, (rhmf.W / rhmf.input_W).mean(axis=0), lw=2)
plt.show()

# Histogram of the weights (number of spectra with significant downweighting anywhere)
plt.figure(dpi=100, figsize=[9, 7], layout="compressed")
plt.hist(
    np.sum((rhmf.W / rhmf.input_W) < 0.5, axis=1),
    bins=30,
    histtype="step",
    color="C2",
    lw=2,
)
plt.xlabel("Number of downweighted pixels")
plt.ylabel("Number of spectra")
# plt.yscale("log")
plt.show()

# Get source IDs of the most downweighted spectra and also how much target pixels were downweighted
n_downweighted = np.sum((rhmf.W / rhmf.input_W) < 0.5, axis=1)
most_downweighted_idxs = np.argsort(n_downweighted)[-5:][::-1]
print("Most downweighted spectra:")
for i in most_downweighted_idxs:
    sid = similar_ids[i]
    n_dw = n_downweighted[i]
    print(f"  Source ID {sid} with {n_dw} downweighted pixels")
target_dw = np.sum((rhmf.W[target_spec_idx] / rhmf.input_W[target_spec_idx]) < 0.5)
print(f"Target spectrum had {target_dw} downweighted pixels")

# Plot the most downweighted spectra
fig, ax = plt.subplots(figsize=[18, 7], dpi=100, layout="compressed")
ax.plot(spec_λ, spec_flux[target_spec_idx], lw=1, alpha=1, c="C3", label="Target")
for i in most_downweighted_idxs:
    sid = source_id[similar_mask][i]
    n_dw = n_downweighted[i]
    ax.plot(
        spec_λ,
        similar_specs[i],
        lw=1,
        alpha=0.5,
        c="C2",
        label=f"Similar {sid} ({n_dw} downweighted)",
    )
ax.set_xlabel(r"$\lambda$ [nm]")
ax.set_ylabel("Normalised Flux")
# ax.set_xlim(849.5, 850.5)
ax.set_ylim(0.8, 1.00)
ax.set_title(f"Gaia DR3 {TARGET_ID} and most downweighted similar sources in CM-space")
ax.legend()
plt.show()

# Plot all spectra on CMD, coloured by number of downweighted pixels
fig, ax = plt.subplots(dpi=100, figsize=[10, 7], layout="compressed")
sc = ax.scatter(
    similar_bp_rp,
    similar_abs_mag_G,
    c=n_downweighted,
    s=50,
    alpha=1,
    cmap="viridis",
    vmin=0,
    vmax=np.max(n_downweighted),
)
ax.set_ylabel("Absolute Mag G band")
ax.set_xlabel("BP - RP")
ypad_fac = 2
xpad_fac = 2
y_ext = THRESH_ABS_MAG * ypad_fac
x_ext = THRESH_BP_RP * xpad_fac
ax.set_ylim(target_abs_mag_G + y_ext, target_abs_mag_G - y_ext)
ax.set_xlim(target_bp_rp - x_ext, target_bp_rp + x_ext)
ax.legend()
cbar = plt.colorbar(sc)
cbar.set_label("Number of downweighted pixels")
plt.show()

# Plot some random spectra and their fits
fitted_specs = rhmf.synthesis()
residuals = rhmf.resid()
n_to_plot = 5
fig, ax = plt.subplots(
    n_to_plot, 1, figsize=[18, 3 * n_to_plot], dpi=100, layout="compressed"
)
for i, idx in enumerate(rng.choice(Y.shape[0], size=n_to_plot, replace=False)):
    ax[i].plot(spec_λ, Y[idx], lw=1, alpha=1, c="C2", label="Data")
    ax[i].plot(spec_λ, fitted_specs[idx], lw=1, alpha=1, c="C3", label="Fit")
    # ax[i].plot(spec_λ, residuals[idx], lw=1, alpha=1, c="C4", label="Residual")
    ax[i].set_ylabel("Normalised Flux")
    ax[i].set_ylim(0.7, 1.1)
    ax[i].set_title(f"Gaia DR3 {source_id[similar_mask][idx]}")
    ax[i].legend()
ax[-1].set_xlabel(r"$\lambda$ [nm]")
plt.show()

# Plot the target spectrum and its fit
fig, ax = plt.subplots(figsize=[18, 7], dpi=100, layout="compressed")
ax.plot(spec_λ, Y[target_similar_idx], lw=1, alpha=1, c="C2", label="Data")
ax.plot(spec_λ, fitted_specs[target_similar_idx], lw=1, alpha=1, c="C3", label="Fit")
# ax.plot(spec_λ, residual_target, lw=1, alpha=1, c="C4", label="Residual")
ax.set_xlabel(r"$\lambda$ [nm]")
ax.set_ylabel("Normalised Flux")
ax.set_ylim(0.7, 1.1)
ax.set_title(f"Gaia DR3 {similar_ids[target_similar_idx]}")
ax.legend()
plt.show()

# Plot the spectra of the 5 most downweighted sources and their fits
fig, ax = plt.subplots(5, 1, figsize=[18, 3 * 5], dpi=100, layout="compressed")
for i, idx in enumerate(most_downweighted_idxs):
    sid = source_id[similar_mask][idx]
    n_dw = n_downweighted[idx]
    ax[i].plot(spec_λ, similar_specs[idx], lw=1, alpha=1, c="C2", label="Data")
    ax[i].plot(
        spec_λ,
        fitted_specs[idx],
        lw=1,
        alpha=1,
        c="C3",
        label="Fit",
    )
    # ax[i].plot(spec_λ, residuals[idx], lw=1, alpha=1, c="C4", label="Residual")
    ax[i].set_ylabel("Normalised Flux")
    ax[i].set_ylim(0.7, 1.1)
    ax[i].set_title(f"Gaia DR3 {sid} ({n_dw} downweighted pixels)")
    ax[i].legend()
ax[-1].set_xlabel(r"$\lambda$ [nm]")
plt.show()

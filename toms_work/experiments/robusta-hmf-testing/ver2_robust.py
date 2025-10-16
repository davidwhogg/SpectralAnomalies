import jax
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
from robusta_hmf.convergence import ConvergenceTester
from robusta_hmf.frame import OptFrame
from robusta_hmf.initialisation import Initialiser
from robusta_hmf.rhmf import ALS_RHMF, SGD_RHMF

plt.style.use("mpl_drip.custom")
rng = np.random.default_rng(0)

# NOTE: LESSONS SO FAR
# - Absolutely do not use whitening on either A or G. It just makes things worse.
#   - REVISED: nevermind it's totally fine for BOTH ALS and SGD in both cases!
# - Definitely rotate A, not G.
#   - REVISED: ALSO fine to do either in both cases!
# - No need to use the Block-SGD thingy, jointly optimising A and G seems fine.
# - Adafactor seems to work better than Adam.
# - Having the end pixels have many nans weights messes up the SGD optimisation. Clip them.


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
THRESH_BP_RP = 0.5
THRESH_ABS_MAG = 0.5
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
# Count the number of NaNs and zeros in W, and negatives
# n_nans = np.sum(np.isnan(W))
# n_zeros = np.sum(W == 0)
# n_neg = np.sum(W < 0)
# print(f"Number of NaNs in W: {n_nans}")
# print(f"Number of zeros in W: {n_zeros}")
# print(f"Number of negatives in W: {n_neg}")

# Plot the number of nan weights per pixel
plt.figure(figsize=[8, 4], dpi=100, layout="compressed")
plt.plot(np.sum(np.isnan(W), axis=0))
plt.xlabel("Pixel index")
plt.ylabel("Number of nan weights")
plt.title("Number of nan weights per pixel")
plt.show()

# Plot the number of zeros in the data (Y) per pixel
plt.figure(figsize=[8, 4], dpi=100, layout="compressed")
plt.plot(np.sum(Y == 0, axis=0))
plt.xlabel("Pixel index")
plt.ylabel("Number of zero fluxes")
plt.title("Number of zero fluxes per pixel")
plt.show()

spec_nans_mask = nans_mask([Y, W])
Y[~spec_nans_mask] = np.nan
W[~spec_nans_mask] = np.nan
Y = np.nan_to_num(Y)
W = np.nan_to_num(W)

clip_edge_pix = 20
l_ind, u_ind = clip_edge_pix, Y.shape[1] - clip_edge_pix
Y = Y[:, l_ind:u_ind]
W = W[:, l_ind:u_ind]
spec_λ = spec_λ[l_ind:u_ind]


OPT_TYPE = "sgd"  # "sgd" or "als"

# The below will be hidden from users in the future, with optional overrides of course
if OPT_TYPE == "sgd":
    als_hmf = SGD_RHMF(learning_rate=1e-3, whiten=False, target="G")
    opt = als_hmf.opt
    ROT_CADENCE = 10
    conv_strategy = "rel_frac_loss"
elif OPT_TYPE == "als":
    als_hmf = ALS_RHMF(whiten=False, target="G")
    opt = None
    ROT_CADENCE = 1
    conv_strategy = "rel_frac_loss"
else:
    raise Exception("whoops")

RANK = 7

conv_tester = ConvergenceTester(strategy=conv_strategy, tol=1e-3)
init = Initialiser(N=Y.shape[0], M=Y.shape[1], K=RANK, strategy="svd")

frame = OptFrame(method=als_hmf, conv_tester=conv_tester)

init_state = init.execute(seed=0, Y=Y, opt=opt)

N_ITER = 1000
CONV_CADENCE = 20

plot_state, loss_history = frame.run(
    Y=Y,
    W=W,
    init_state=init_state,
    rotation_cadence=ROT_CADENCE,
    conv_check_cadence=CONV_CADENCE,
    max_iter=N_ITER,
)

# Plot the inferred basis vectors
plt.figure(figsize=[8, 8], dpi=100, layout="compressed")
for k in range(RANK):
    plt.plot(spec_λ, plot_state.G[:, k], lw=2, label=f"Basis {k}")
plt.xlabel(r"$\lambda$ [nm]")
plt.ylabel("Basis flux")
# plt.ylim(-0.05, 0.05)
plt.legend()
plt.show()

plt.figure(figsize=[8, 3], dpi=100, layout="compressed")
for k in range(RANK):
    plt.plot(plot_state.A[:, k], lw=2, label=f"Basis {k}")
plt.ylabel("Coefficients")
plt.legend()
plt.show()

plt.figure()
plt.plot(loss_history)
plt.yscale("log")
plt.ylabel("Loss")
plt.xlabel("Iteration")
plt.show()

# Reconstruct the spectra
Y_rec = plot_state.A @ plot_state.G.T
# Plot the target spectrum and its reconstruction
fig, ax = plt.subplots(figsize=[9, 5], dpi=100, layout="compressed")
ax.plot(
    spec_λ,
    similar_specs[target_similar_idx][l_ind:u_ind],
    lw=2,
    alpha=1,
    c="C0",
    label="Target",
)
ax.plot(
    spec_λ,
    Y_rec[target_similar_idx],
    lw=2,
    alpha=1,
    ls="--",
    c="C1",
    label="Reconstruction",
)
ax.set_xlabel(r"$\lambda$ [nm]")
ax.set_ylabel("Normalised Flux")
ax.set_title(f"Gaia DR3 {TARGET_ID} and HMF reconstruction")
ax.legend()
plt.show()

# 3 panel plot of 3 random spectra and their reconstructions
rand_idxs = rng.choice(Y.shape[0], size=3, replace=False)
fig, axs = plt.subplots(3, 1, figsize=[9, 9], dpi=100, layout="compressed", sharex=True)
for i, ax in zip(rand_idxs, axs):
    ax.plot(
        spec_λ,
        similar_specs[i][l_ind:u_ind],
        lw=2,
        alpha=1,
        c="C0",
        label="Data",
    )
    ax.plot(
        spec_λ,
        Y_rec[i],
        lw=2,
        alpha=1,
        ls="--",
        c="C1",
        label="Reconstruction",
    )
    ax.set_ylabel("Normalised Flux")
    ax.legend()
axs[-1].set_xlabel(r"$\lambda$ [nm]")
fig.suptitle(f"HMF reconstruction of Gaia DR3 {TARGET_ID} and similar sources", y=1.1)
plt.show()

# Plot the robust weights
robust_weights = als_hmf.likelihood.weights_irls(
    Y=Y,
    W_data=W,
    A=plot_state.A,
    G=plot_state.G,
)
plt.figure(figsize=[8, 4], dpi=100, layout="compressed")
plt.imshow(robust_weights, aspect="auto", origin="lower")
plt.colorbar(label="Robust Weight")
plt.title("Robust Weights")
plt.xlabel(r"Pixel Index")
plt.ylabel("Spectrum Index")
plt.show()

# Print some diagnostics that check for mode collapse
AtA = plot_state.A.T @ plot_state.A
GtG = plot_state.G.T @ plot_state.G
print(np.linalg.eigvalsh(AtA / np.trace(AtA)))
print(np.linalg.eigvalsh(GtG / np.trace(GtG)))

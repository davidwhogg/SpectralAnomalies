import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from collect_data import get_data
from robusta_hmf import Robusta
from robusta_hmf.rhmf_hogg import RHMF
from utils import principal_angles

jax.config.update("jax_enable_x64", True)
plt.style.use("mpl_drip.custom")
rng = np.random.default_rng(42)


# ==== Read the GAIA RVS spectra data ====

Y, W, spec_λ, bp_rp, abs_mag_G = get_data(
    thresh_bp_rp=0.3,  # NOTE: Increase these thresholds to get more data
    thresh_abs_mag=0.3,
    clip_edge_pix=20,  # NOTE: Don't clip much less or we get edge effects due to shitty spectra shit I don't understand that kills the optimisation
)
print("\n================================\n")

# ==== Model config ====

# Model setup options shared for (a.i) Hogg code and (b.i) Robusta ALS (b.ii) Robusta SGD
RANK = 3
ROBUST_SCALE = 2.0
MAX_ITER = 1000

# NOTE: For ALS, I'll match Hogg's convergence criterion. For SGD, that tends to cause early exits so I do something based on change in loss instead.
conv_tol = 1e-3

# ==== Fitting ====

# Build the models
model_als = Robusta(
    rank=RANK,
    method="als",
    robust_scale=ROBUST_SCALE,
    conv_strategy="max_frac_G",
    conv_tol=conv_tol,
    init_strategy="svd",
    rotation="fast",
    target="G",
    whiten=True,
)
model_sgd = Robusta(
    rank=RANK,
    method="sgd",  # NOTE: Although I called this SGD, I'm not using mini-batches, so it's just full-batch first order optimisation
    robust_scale=ROBUST_SCALE,
    conv_strategy="rel_frac_loss",
    conv_tol=conv_tol,
    init_strategy="svd",
    rotation="fast",
    target="G",
    whiten=True,
)
model_hogg = RHMF(
    rank=RANK,
    nsigma=ROBUST_SCALE,
)
model_hogg.set_training_data(Y, weights=W)

# Train the models
print("Training Robusta ALS...")
state_als, loss_history_als = model_als.fit(
    Y=Y, W=W, max_iter=MAX_ITER, conv_check_cadence=1
)
print("\nTraining Robusta SGD...")
state_sgd, loss_history_sgd = model_sgd.fit(Y=Y, W=W, max_iter=MAX_ITER)
print("\nTraining Hogg RHMF...")
init_state = model_als._initialiser.execute(Y=Y)
model_hogg.A = jnp.array(init_state.A.T)
model_hogg.G = jnp.array(init_state.G.T)
model_hogg.train(maxiter=MAX_ITER, tol=conv_tol)

# Plot the loss histories for both Robusta models
plt.figure(figsize=[8, 5], dpi=100, layout="compressed")
plt.plot(loss_history_als, lw=2, label="Robusta ALS")
plt.plot(loss_history_sgd, lw=2, label="Robusta SGD")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("Loss Histories")
plt.legend()
plt.show()

# NOTE: As we see in the loss history plots, the SGD model plateaus at higher loss than the ALS. I think in practice this wouldn't really matter for
# the following reason. Here, I'm initialising everything from the same SVD. In practice, one would fit with ALS to a tractable subset of the data, take the basis vectors, solve for the coefficients for all data, then initialise SGD from there.

print("\n================================\n")

# ==== Analysis ====

# Let's compare the basis vectors
basis_als = model_als.basis_vectors()
basis_sgd = model_sgd.basis_vectors()
basis_hogg = jnp.array(model_hogg.G.T)


def get_sign_abs_max(x):
    return jnp.sign(x[jnp.argmax(jnp.abs(x))])


fig, axes = plt.subplots(
    RANK, 1, dpi=100, figsize=[8, 3 * RANK], layout="compressed", sharex=True
)
for k in range(RANK):
    axes[k].plot(
        spec_λ,
        basis_als[:, k] / get_sign_abs_max(basis_als[:, k]),
        lw=2,
        alpha=0.7,
        label="Robusta ALS",
    )
    axes[k].plot(
        spec_λ,
        basis_sgd[:, k] / get_sign_abs_max(basis_sgd[:, k]),
        lw=2,
        alpha=0.7,
        label="Robusta SGD",
    )
    axes[k].plot(
        spec_λ,
        basis_hogg[:, k] / get_sign_abs_max(basis_hogg[:, k]),
        lw=2,
        alpha=0.7,
        label="Hogg RHMF",
    )
    if k == RANK - 1:
        axes[k].legend()
        axes[k].set_xlabel(r"$\lambda$ [nm]")
    axes[k].set_ylabel("Flux")
plt.suptitle("Inferred Basis Vectors Comparison")
plt.show()

# Same plot but for the coefficients
coeffs_als = model_als.coefficients()
coeffs_sgd = model_sgd.coefficients()
coeffs_hogg = jnp.array(model_hogg.A)

fig, axes = plt.subplots(
    RANK, 1, dpi=100, figsize=[8, 3 * RANK], layout="compressed", sharex=True
)
for k in range(RANK):
    axes[k].plot(
        coeffs_als[:, k],
        lw=2,
        alpha=0.7,
        label="Robusta ALS",
    )
    axes[k].plot(
        coeffs_sgd[:, k],
        lw=2,
        alpha=0.7,
        label="Robusta SGD",
    )
    axes[k].plot(
        coeffs_hogg[:, k],
        lw=2,
        alpha=0.7,
        label="Hogg RHMF",
    )
    if k == RANK - 1:
        axes[k].legend()
        axes[k].set_xlabel("Spectrum Index")
    axes[k].set_ylabel("Value")
plt.suptitle("Inferred Coefficients Comparison")
plt.show()

# Calculate principal angles between each pair of basis vector subspaces
angles_als_sgd = principal_angles(basis_als, basis_sgd, degrees=True)
angles_als_hogg = principal_angles(basis_als, basis_hogg, degrees=True)
angles_sgd_hogg = principal_angles(basis_sgd, basis_hogg, degrees=True)

# Plot the principal angles
plt.figure(figsize=[8, 5], dpi=100, layout="compressed")
plt.plot(
    range(1, len(angles_als_sgd) + 1),
    angles_als_sgd,
    marker="o",
    lw=2,
    label="Robusta ALS vs Robusta SGD",
)
plt.plot(
    range(1, len(angles_als_hogg) + 1),
    angles_als_hogg,
    marker="o",
    lw=2,
    label="Robusta ALS vs Hogg RHMF",
)
plt.plot(
    range(1, len(angles_sgd_hogg) + 1),
    angles_sgd_hogg,
    marker="o",
    lw=2,
    label="Robusta SGD vs Hogg RHMF",
)
plt.xlabel("Principal Angle Index")
plt.xticks([1, 2, 3])
plt.ylabel("Principal Angle (degrees)")
plt.title("Principal Angles Between Inferred Basis Vector Subspaces")
plt.legend()
plt.show()

# Plot the reconstructions for a few random spectra
rand_idxs = rng.choice(Y.shape[0], size=3, replace=False)
Y_rec_als = model_als.synthesize()
Y_rec_sgd = model_sgd.synthesize()
Y_rec_hogg = jnp.array(model_hogg.synthesis())
fig, axs = plt.subplots(3, 1, figsize=[9, 9], dpi=100, layout="compressed", sharex=True)
for i, ax in zip(rand_idxs, axs):
    ax.plot(
        spec_λ,
        Y[i],
        lw=2,
        alpha=1,
        c="k",
        label="Data",
    )
    ax.plot(
        spec_λ,
        Y_rec_als[i],
        lw=2,
        alpha=1,
        ls="--",
        c="C1",
        label="Robusta ALS Recon",
    )
    ax.plot(
        spec_λ,
        Y_rec_sgd[i],
        lw=2,
        alpha=1,
        ls=":",
        c="C2",
        label="Robusta SGD Recon",
    )
    ax.plot(
        spec_λ,
        Y_rec_hogg[i],
        lw=2,
        alpha=1,
        ls="-.",
        c="C3",
        label="Hogg RHMF Recon",
    )
    ax.set_ylabel("Normalised Flux")
    ax.set_title(f"Spectrum Index {i} and Reconstructions")
    if i == rand_idxs[-1]:
        ax.set_xlabel(r"$\lambda$ [nm]")
        ax.legend()
plt.show()

# Plot and compare the robust weights for the three models
W_als = model_als.robust_weights(Y, W)
W_sgd = model_sgd.robust_weights(Y, W)
W_hogg = jnp.array(model_hogg.W) / W

fig, axes = plt.subplots(
    3, 1, dpi=100, figsize=[10, 12], layout="compressed", sharex=True
)
im0 = axes[0].imshow(
    W_als,
    aspect="auto",
    cmap="viridis",
    vmin=0,
    vmax=1,
)
axes[0].set_title("Robusta ALS Robust Weights")
plt.colorbar(im0, ax=axes[0], label="Robust Weight")
im1 = axes[1].imshow(
    W_sgd,
    aspect="auto",
    cmap="viridis",
    vmin=0,
    vmax=1,
)
axes[1].set_title("Robusta SGD Robust Weights")
plt.colorbar(im1, ax=axes[1], label="Robust Weight")
im2 = axes[2].imshow(
    W_hogg,
    aspect="auto",
    cmap="viridis",
    vmin=0,
    vmax=1,
)
axes[2].set_title("Hogg RHMF Robust Weights")
plt.colorbar(im2, ax=axes[2], label="Robust Weight")
axes[2].set_xlabel("Pixel Index")
plt.show()

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from robusta_hmf import (
    ALS_RHMF,
    SGD_RHMF,
    GaussianLikelihood,
    JointOptimiser,
    L2Regularizer,
    Regularizer,
    Reorienter,
    StudentTLikelihood,
    WeightedAStep,
    WeightedGStep,
)

# --- Assume you've already defined:
# RHMFState, GaussianLikelihood, L2Regularizer, WeightedAStep,
# WeightedGStep, Reorienter, ALS_RHMF, SGD_RHMF
# (from the refactored "most jax-pure" version with eqx.tree_at)


# ---------------- Synthetic dataset ----------------
def make_synthetic(N=50, D=40, K=3, noise=0.1, key=jax.random.PRNGKey(0)):
    k1, k2, k3 = jax.random.split(key, 3)
    A_true = jax.random.normal(k1, (N, K))
    G_true = jax.random.normal(k2, (D, K))
    X_true = A_true @ G_true.T
    X_noisy = X_true + noise * jax.random.normal(k3, (N, D))
    return X_true, X_noisy


# ---------------- Helper: error metric ----------------
def rel_error(X, A, G):
    pred = A @ G.T
    return jnp.linalg.norm(X - pred) / jnp.linalg.norm(X)


# ---------------- Main demo ----------------
if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    N, D, K = 60, 45, 4
    X_true, X = make_synthetic(N, D, K, noise=0.05, key=key)

    # ---- ALS model ----
    als_model = ALS_RHMF(
        likelihood=GaussianLikelihood(),
        a_step=WeightedAStep(ridge=1e-6),
        g_step=WeightedGStep(ridge=1e-6),
        reorienter=Reorienter(whiten=True, eps=1e-6),
        regularizer=L2Regularizer(weight=1e-6),
    )
    als_state = als_model.init_state(N, D, K, key)

    # Run ALS iterations
    for i in range(50):
        als_state, loss = als_model.step(X, als_state)
        if i % 10 == 0:
            err = rel_error(X_true, als_state.A, als_state.G)
            print(f"[ALS] iter {i:03d} | loss {loss:.4f} | rel_error {err:.4f}")

    # ---- SGD model ----
    opt = optax.adam(1e-2)
    sgd_model = SGD_RHMF(
        likelihood=GaussianLikelihood(),
        opt=opt,
        regularizer=L2Regularizer(weight=1e-6),
    )
    sgd_state = sgd_model.init_state(N, D, K, key)

    # Run SGD iterations
    for i in range(200):  # needs more steps
        sgd_state, loss = sgd_model.step(X, sgd_state)
        if i % 20 == 0:
            err = rel_error(X_true, sgd_state.A, sgd_state.G)
            print(f"[SGD] iter {i:03d} | loss {loss:.4f} | rel_error {err:.4f}")

    # ---- Final check ----
    als_err = rel_error(X_true, als_state.A, als_state.G)
    sgd_err = rel_error(X_true, sgd_state.A, sgd_state.G)
    print("\n=== Final Comparison ===")
    print(f"ALS rel error: {als_err:.4e}")
    print(f"SGD rel error: {sgd_err:.4e}")
    print(f"Difference   : {abs(als_err - sgd_err):.4e}")


# --- Align SGD basis G to ALS basis G ---
def orthogonal_procrustes(X, Y):
    # Find R that minimises ||X - YR||_F with R orthogonal
    U, _, Vt = jnp.linalg.svd(Y.T @ X, full_matrices=False)
    R = U @ Vt
    return Y @ R


# --- Orthogonal Procrustes (rotation only) ---
U, _, Vt = jnp.linalg.svd(sgd_state.G.T @ als_state.G, full_matrices=False)
R = U @ Vt
G_rot = sgd_state.G @ R  # rotated SGD basis, [D,K]

# --- Optimal diagonal rescaling, one per component ---
num = jnp.sum(als_state.G * G_rot, axis=0)  # <g_als, g_rot>
den = jnp.sum(G_rot * G_rot, axis=0) + 1e-12  # ||g_rot||^2
S_diag = num / den  # per-component scales
G_sgd_aligned_scaled = G_rot * S_diag  # [D,K]

# --- Plot comparison ---
fig, axes = plt.subplots(K, 1, figsize=(8, 2 * K), sharex=True)
for k in range(K):
    axes[k].plot(als_state.G[:, k], label="ALS", color="C0")
    axes[k].plot(
        G_sgd_aligned_scaled[:, k], "--", label="SGD (aligned+scaled)", color="C1"
    )
    axes[k].set_ylabel(f"Comp {k}")
    if k == 0:
        axes[k].legend()
plt.xlabel("Feature index (D)")
plt.suptitle(
    "Basis functions comparison (ALS vs SGD)\n(orthogonal alignment + per-component scaling)"
)
plt.tight_layout()
plt.show()


for k in range(K):
    corr = jnp.corrcoef(als_state.G[:, k], G_sgd_aligned_scaled[:, k])[0, 1]
    print(f"Comp {k}: corr = {corr:.4f}")

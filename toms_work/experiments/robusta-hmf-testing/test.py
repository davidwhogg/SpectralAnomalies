import jax
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

key = jax.random.PRNGKey(42)
N, D, K = 6000, 4500, 4
X = jax.random.normal(key, (N, D))

print("=== ALS (Gaussian) with reorientation (whiten) ===")
als_model = ALS_RHMF(
    likelihood=GaussianLikelihood(),
    a_step=WeightedAStep(ridge=1e-6),
    g_step=WeightedGStep(ridge=1e-6),
    reorienter=Reorienter(whiten=True, eps=1e-6),
    regularizer=Regularizer(),  # no penalty
)
als_state = als_model.init_state(N, D, K, key)
for i in range(5):
    als_state, als_loss = als_model.step(X, als_state)
    print(f"ALS iter {i:02d} | loss {als_loss:.4f}")

print("\n=== SGD (Student-t, nu=5) with Adam, L2 regularizer ===")
opt = optax.adam(1e-2)
# Initialise opt state with correct param shapes
dummy_params = (X[:, :K], X[:K, :K])  # shapes only
opt_state = opt.init(dummy_params)
sgd_model = SGD_RHMF(
    likelihood=StudentTLikelihood(nu=5.0, scale=1.0),
    opt=opt,  # pass the Optax transformation, not JointOptimiser
    regularizer=L2Regularizer(weight=1e-3),
)
sgd_state = sgd_model.init_state(N, D, K, key)
for i in range(5):
    sgd_state, sgd_loss = sgd_model.step(X, sgd_state)
    print(f"SGD  iter {i:02d} | loss {sgd_loss:.4f}")

# new.py
# Tom and GPT-5 Make RHMF go brrrr
import equinox as eqx
import jax
import jax.numpy as jnp
import optax


# ---------------- State ----------------
class RHMFState(eqx.Module):
    A: jnp.ndarray
    G: jnp.ndarray
    it: int  # = eqx.field(default=0)


# ---------------- Likelihood Base + Implementations ----------------
# TODO: Check Equinox convention for abstract base classes
class Likelihood(eqx.Module):
    """Abstract base for likelihoods."""

    def weights(self, X, A, G):
        raise NotImplementedError

    def loss(self, X, A, G):
        raise NotImplementedError


class GaussianLikelihood(Likelihood):
    """Gaussian = least squares."""

    def weights(self, X, A, G):
        return jnp.ones_like(X)

    def loss(self, X, A, G):
        pred = A @ G.T
        resid = X - pred
        return jnp.sum(resid**2)


class CauchyLikelihood(Likelihood):
    """Cauchy (Student-t with nu=1)."""

    scale: float = 1.0

    def weights(self, X, A, G):
        pred = A @ G.T
        resid = X - pred
        return 1.0 / (1.0 + (resid / self.scale) ** 2)

    def loss(self, X, A, G):
        pred = A @ G.T
        resid = X - pred
        return jnp.sum(jnp.log1p((resid / self.scale) ** 2))


class StudentTLikelihood(Likelihood):
    """General Student-t."""

    nu: float = 3.0
    scale: float = 1.0

    def weights(self, X, A, G):
        pred = A @ G.T
        resid = (X - pred) / self.scale
        return (self.nu + 1.0) / (self.nu + resid**2)

    def loss(self, X, A, G):
        pred = A @ G.T
        resid = (X - pred) / self.scale
        # Negative log-likelihood up to constants
        return jnp.sum(0.5 * (self.nu + 1.0) * jnp.log1p(resid**2 / self.nu))


# ---------------- Regularizers ----------------
class Regularizer(eqx.Module):
    """Abstract base; default is no regularization."""

    def __call__(self, state: RHMFState) -> float:
        return 0.0


class L2Regularizer(Regularizer):
    weight: float = 1e-3

    def __call__(self, state: RHMFState) -> float:
        return self.weight * (jnp.sum(state.A**2) + jnp.sum(state.G**2))


# ---------------- ALS Steps ----------------


class WeightedAStep(eqx.Module):
    ridge: float = eqx.field(static=True, default=1e-6)

    def __call__(self, X, state: RHMFState, W):
        G = state.G

        def solve_row(x_i, w_i):
            s = jnp.sqrt(w_i)
            Gw = G * s[:, None]
            M = Gw.T @ Gw + self.ridge * jnp.eye(G.shape[1], dtype=G.dtype)
            b = Gw.T @ (s * x_i)
            return jnp.linalg.solve(M, b)

        A_new = jax.vmap(solve_row)(X, W)  # [N, K]
        return eqx.tree_at(lambda s: s.A, state, A_new)


class WeightedGStep(eqx.Module):
    ridge: float = eqx.field(static=True, default=1e-6)

    def __call__(self, X, state: RHMFState, W):
        A = state.A

        def solve_col(x_j, w_j):
            s = jnp.sqrt(w_j)
            Aw = A * s[:, None]
            M = Aw.T @ Aw + self.ridge * jnp.eye(A.shape[1], dtype=A.dtype)
            b = Aw.T @ (s * x_j)
            return jnp.linalg.solve(M, b)

        G_new = jax.vmap(solve_col)(X.T, W.T)  # [D, K]
        return eqx.tree_at(lambda s: s.G, state, G_new)


# ---------------- Reorientation (non-lazy) ----------------
class Reorienter(eqx.Module):
    whiten: bool = eqx.field(static=True, default=True)
    eps: float = eqx.field(static=True, default=1e-6)

    def __call__(self, state: RHMFState):
        A = state.A
        K = A.shape[1]
        C = A.T @ A + self.eps * jnp.eye(K, dtype=A.dtype)
        evals, V = jnp.linalg.eigh(C)

        if self.whiten:
            invsqrt = 1.0 / jnp.sqrt(jnp.maximum(evals, self.eps))
            sqrtv = jnp.sqrt(jnp.maximum(evals, self.eps))
            R = V @ (invsqrt[:, None] * V.T)  # V Λ^{-1/2} V^T
            Rinverse = V @ (sqrtv[:, None] * V.T)
        else:
            R, Rinverse = V, V.T

        A_new = A @ R
        G_new = state.G @ Rinverse
        state = eqx.tree_at(lambda s: s.A, state, A_new)
        state = eqx.tree_at(lambda s: s.G, state, G_new)
        return state


# ---------------- SGD Optimiser ----------------
class JointOptimiser(eqx.Module):
    opt: optax.GradientTransformation
    opt_state: optax.OptState

    def __call__(
        self, X, state: RHMFState, likelihood: Likelihood, regularizer: Regularizer
    ):
        def loss_fn(params, X):
            A, G = params
            base = likelihood.loss(X, A, G)
            reg = regularizer(state.replace(A=A, G=G))
            return base + reg

        params = (state.A, state.G)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(params, X)
        updates, new_opt_state = self.opt.update(grads, self.opt_state, params)
        new_params = optax.apply_updates(params, updates)
        A_new, G_new = new_params
        new_state = state.replace(A=A_new, G=G_new, it=state.it + 1)
        return new_state, JointOptimiser(self.opt, new_opt_state), loss


# ---------------- Orchestrators (branch-free steps) ----------------
class ALS_RHMF(eqx.Module):
    likelihood: Likelihood
    a_step: WeightedAStep
    g_step: WeightedGStep
    reorienter: Reorienter
    regularizer: Regularizer = Regularizer()

    def init_state(self, N, D, K, key):
        k1, k2 = jax.random.split(key)
        A = jax.random.normal(k1, (N, K))
        G = jax.random.normal(k2, (D, K))
        return RHMFState(A=A, G=G, it=0)

    @eqx.filter_jit
    def step(self, X, state: RHMFState):
        W = self.likelihood.weights(X, state.A, state.G)
        state = self.a_step(X, state, W)
        state = self.g_step(X, state, W)
        state = self.reorienter(state)
        loss = self.likelihood.loss(X, state.A, state.G) + self.regularizer(state)
        state = eqx.tree_at(lambda s: s.it, state, state.it + 1)
        return state, loss


class SGDState(eqx.Module):
    A: jnp.ndarray
    G: jnp.ndarray
    it: int
    opt_state: optax.OptState


class SGD_RHMF(eqx.Module):
    likelihood: Likelihood
    opt: optax.GradientTransformation
    regularizer: Regularizer = Regularizer()

    def init_state(self, N, D, K, key):
        k1, k2 = jax.random.split(key)
        A = jax.random.normal(k1, (N, K))
        G = jax.random.normal(k2, (D, K))
        opt_state = self.opt.init((A, G))
        return SGDState(A=A, G=G, it=0, opt_state=opt_state)

    @eqx.filter_jit
    def step(self, X, state: SGDState):
        def loss_fn(params, X):
            A, G = params
            return self.likelihood.loss(X, A, G) + self.regularizer(
                RHMFState(A=A, G=G, it=state.it)
            )

        params = (state.A, state.G)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(params, X)
        updates, opt_state = self.opt.update(grads, state.opt_state, params)
        A_new, G_new = optax.apply_updates(params, updates)

        state = eqx.tree_at(lambda s: s.A, state, A_new)
        state = eqx.tree_at(lambda s: s.G, state, G_new)
        state = eqx.tree_at(lambda s: s.opt_state, state, opt_state)
        state = eqx.tree_at(lambda s: s.it, state, state.it + 1)
        return state, loss

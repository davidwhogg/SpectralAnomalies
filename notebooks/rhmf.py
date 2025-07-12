"""
# Robust Heteroskedastic Matrix Factorization
An iteratively reweighted least-squares version of HMF.

## Author:
- **David W. Hogg** (NYU) (MPIA) (Flatiron)

## License:
Copyright 2025 the author. All code is licensed for re-use under the MIT License.
"""

import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

class RHMF():
    def __init__(self, rank, nsigma, A=None, G=None, tol=1.e-4):
        self.K = int(rank)
        self.nsigma = float(nsigma)
        self.Q2 = self.nsigma ** 2
        self.A = A
        self.G = G
        self.tol = tol
        self.trained = False

    def train(self, data, weights):
        """
        # inputs:
        `data`:     (N, M) array of observations.
        `weights`:  (N, M) units of (and equivalent to) inverse uncertainty variances.

        # comments:
        - Checks convergence with the g-step only.
        """
        self.trained = False
        assert jnp.all(jnp.isfinite(data))
        assert jnp.all(jnp.isfinite(weights))
        self.Y = jnp.array(data)
        self.input_W = jnp.array(weights)
        assert self.Y.shape == self.input_W.shape
        self.N, self.M = self.Y.shape
        self.converged = False
        self.n_iter = 0
        self._initialize()
        print("train(): before starting:", self.objective(), self.original_objective())
        while not self.converged:
            self._A_step()
            self._G_step()
            self._affine()
            self._update_W()
            self.n_iter += 1
            if self.n_iter % 100 == 0:
                print("train(): after iteration", self.n_iter, ":",
                      self.objective(), self.original_objective())
        print("train(): converged at iteration", self.n_iter, ":",
              self.objective(), self.original_objective())
        self.trained = True

    def test(self, ystar, wstar):
        """
        # inputs:
        `ystar`:     (M, ) array for one observation.
        `wstar`:     (M, ) units of (and equivalent to) inverse uncertainty variances.

        # outputs:
        `synth`:     (M, ) synthetic spectrum.

        # comments:
        - Checks convergence with the a-step only.
        """
        assert self.trained
        assert jnp.all(np.isfinite(ystar))
        assert jnp.all(np.isfinite(wstar))
        assert ystar.shape == (self.M, )
        assert wstar.shape == (self.M, )
        self.converged = False
        self.n_iter = 0
        w = 1. * wstar
        a = jnp.zeros(self.K)
        while not self.converged:
            foo = self.one_star_objective(ystar, w, a)
            a = self._one_star_A_step(ystar, w)
            bar = self.one_star_objective(ystar, w, a)
            if foo - bar < self.tol:
                self.converged = True
            w = self._update_one_star_W(ystar, wstar, a)
            self.n_iter += 1
        print("test(): converged at iteration:", self.n_iter, ":",
              self.one_star_objective(ystar, w, a),
              self.one_star_objective(ystar, wstar, a))
        return self.one_star_synthesis(a)

    def synthesis(self):
        return self.A.T @ self.G

    def one_star_synthesis(self, a):
        return a @ self.G

    def resid(self):
        return self.Y - self.synthesis()

    def one_star_resid(self, y, a):
        return y - self.one_star_synthesis(a)

    def objective(self):
        if self.A is None or self.G is None:
            return jnp.Inf
        return jnp.sum(self.W * self.resid() ** 2)

    def one_star_objective(self, y, w, a):
        return jnp.sum(w * self.one_star_resid(y, a) ** 2)

    def original_chi(self):
        return jnp.sqrt(self.input_W) * self.resid()

    def original_objective(self):
        return jnp.sum(self.input_W * self.resid() ** 2)

    def _initialize(self):
        """
        # bugs:
        - Consider switching SVD to a fast PCA implementation?
        """
        self.W = 1. * self.input_W # copy not reference
        if self.A is None:
            if self.G is None:
                print("_initialize(): initializing with an SVD")
                u, s, v = jnp.linalg.svd(self.Y, full_matrices=False) # maybe slow
                self.A = (u[:,:self.K] * s[:self.K]).T
                self.G = v[:self.K,:]
            else:
                print("_initialize(): initializing with an a-step")
                self._A_step()
        else:
            if self.G is None:
                print("_initialize(): initializing with a g-step")
                self._G_step()
            else:
                print("_initialize(): both A and G provided; initialization unnecessary")
        assert self.A.shape == (self.K, self.N)
        assert self.G.shape == (self.K, self.M)

    def _one_star_A_step(self, y1, w1):
        XTCinvX = self.G * w1 @ self.G.T
        XTCinvY = self.G * w1 @ y1
        return jnp.linalg.solve(XTCinvX, XTCinvY)

    def _one_star_G_step(self, y1, w1):
        XTCinvX = self.A * w1 @ self.A.T
        XTCinvY = self.A * w1 @ y1
        return jnp.linalg.solve(XTCinvX, XTCinvY)

    def _A_step(self):
        foo = self.objective()
        self.A = jax.vmap(self._one_star_A_step)(self.Y, self.W).T
        bar = self.objective()
        if foo < bar:
            print("_A_step(): ERROR: objective got worse", foo, bar)

    def _G_step(self):
        foo = self.objective()
        self.G = jax.vmap(self._one_star_G_step)(self.Y.T, self.W.T).T
        bar = self.objective()
        if foo < bar:
            print("_G_step(): ERROR: objective got worse", foo, bar)
        if foo - bar < self.tol:
            self.converged = True

    def _affine(self):
        """
        # bugs:
        - Consider switching SVD to a fast PCA implementation?
        """
        u, s, v = jnp.linalg.svd(self.A.T @ self.G, full_matrices=False)
        self.A = (u[:,:self.K] * s[:self.K]).T
        self.G = v[:self.K,:]

    def _update_W(self):
        self.W = self.input_W * self.Q2 / (self.input_W * self.resid() ** 2 + self.Q2)

    def _update_one_star_W(self, y, w, a):
        return w * self.Q2 / (w * self.one_star_resid(y, a) ** 2 + self.Q2)

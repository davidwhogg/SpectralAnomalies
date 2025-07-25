"""A
# Robust Heteroskedastic Matrix Factorization
An iteratively-reweighted-least-squares (IRLS) version of HMF.

## Author:
- **David W. Hogg** (NYU) (MPIA) (Flatiron)

## License:
Copyright 2025 the author.
This code is licensed for re-use under the *MIT License*.
See the file `LICENSE` for details.

## Comments:
- Currently converges training step on the maximum (squared) change in the g-step update.
- Currently converges test step on the maximum (squared) change in the a-step update.

## Bugs:
- Needs a way to save and restore a model, like repr? or pickle?
- Needs a set of unit tests.
- Needs a set of functional tests.
- Not packaged into a proper package.
- Is there any theory about the convergence or stability of IRLS?
"""

import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

class RHMF():
    def __init__(self, rank, nsigma, A=None, G=None):
        self.K = int(rank)
        self.nsigma = float(nsigma)
        self.Q2 = self.nsigma ** 2
        self.A = A
        self.G = G
        self.Y = None
        self.input_W = None
        self.trained = False

    def set_training_data(self, data, weights):
        assert jnp.all(jnp.isfinite(data))
        assert jnp.all(jnp.isfinite(weights))
        self.Y = jnp.array(data)          # copy, I hope
        self.input_W = jnp.array(weights) # copy, I hope
        self.N, self.M = self.Y.shape
        assert self.Y.shape == self.input_W.shape
        self.A = None         # because data just changed
        self.trained = False  # because data just changed
        
    def train(self, maxiter=jnp.inf, tol=1.e-5):
        """
        # inputs:
        `data`:     (N, M) array of observations.
        `weights`:  (N, M) units of (and equivalent to) inverse uncertainty variances.

        # comments:
        - Checks convergence with the g-step only.

        # bugs:
        - Should `raise` not `assert` right?
        """
        if self.Y is None or self.input_W is None:
            print("train(): ERROR: No training data.")
            assert False
        self.tol = tol
        self.converged = False
        self.n_iter = 0
        self._initialize()
        print("train(): before starting:", self.objective(), self.original_objective())
        while not self.converged:
            self._update_W()
            self._A_step()
            self._update_W()
            self._G_step()
            self._affine()
            self.n_iter += 1
            if not self._all_tests_pass():
                print("train(): WARNING: failed tests after iteration", self.n_iter)
                self.converged = True
            if self.n_iter % 100 == 0:
                print(f"train(): after iteration {self.n_iter}:",
                      self.objective(), self.original_objective())
            if self.n_iter >= maxiter:
                print("train(): WARNING: stopping at maximum iteration, not true convergence")
                self.converged = True
        print("train(): finished at iteration", self.n_iter, ":",
              self.objective(), self.original_objective())
        self.trained = True

    def test(self, ystar, wstar, maxiter=100, verbose=False, tol=1.e-5):
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
        assert jnp.all(jnp.isfinite(ystar))
        assert jnp.all(jnp.isfinite(wstar))
        assert ystar.shape == (self.M, )
        assert wstar.shape == (self.M, )
        self.converged = False
        self.n_iter = 0
        w = 1. * wstar
        a = jnp.zeros(self.K)
        while not self.converged:
            da = self._one_element_step(self.G, ystar - self.one_star_synthesis(a), w)
            a += da
            if (jnp.max(da * da) / jnp.mean(a * a)) < tol: # input tol not self.tol
                self.converged = True
            w = self._update_one_star_W(ystar, wstar, a)
            self.n_iter += 1
            if self.n_iter >= maxiter:
                print("test(): WARNING: stopping at maximum iteration, not true convergence")
                self.converged = True
        if verbose:
            print("test(): converged at iteration:", self.n_iter, ":",
                  jnp.max(da * da), jnp.mean(a * a),
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
            return jnp.inf
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

    def _one_element_step(self, matrix, y1, w1):
        return jnp.linalg.solve(matrix * w1 @ matrix.T,
                                matrix * w1 @ y1)

    def _A_step(self):
        """
        ## notes:
        - Works on residuals to reduce dynamic ranges for everything.
          (It is not obvious that this helps with anything.)
        """
        if self.A is None:
            self.A = jnp.zeros((self.K, self.N))
        dY = self.resid()
        dA = jax.vmap(self._one_element_step, in_axes=(None, 0, 0))(self.G, dY, self.W).T
        self.A += dA

    def _G_step(self):
        """
        ## notes:
        - Works on residuals to reduce dynamic ranges for everything.
          (It is not obvious that this helps with anything.)
        - Converges on an estimate of fractional change in G (not the objective function).
        """
        dY = self.resid()
        dG = jax.vmap(self._one_element_step, in_axes=(None, 0, 0))(self.A, dY.T, self.W.T).T
        self.G += dG
        if self.n_iter % 5 == 0:
            print(f"_G_step() at iteration {self.n_iter + 1}: maximum fractional squared G adjustment is:",
                  jnp.max(dG * dG) / jnp.mean(self.G * self.G))
        if (jnp.max(dG * dG) / jnp.mean(self.G * self.G)) < self.tol:
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

    def _all_tests_pass(self):
        boo = True
        foo = jnp.sum(jnp.isnan(self.A))
        if foo > 0:
            print(f"WARNING: {foo} elements of A matrix went bad", flush=True)
            boo = False
        foo = jnp.sum(jnp.isnan(self.G))
        if foo > 0:
            print(f"WARNING: {foo} elements of G matrix went bad", flush=True)
            boo = False
        foo = jnp.sum(jnp.isnan(self.W))
        if foo > 0:
            print(f"WARNING: {foo} elements of W matrix went bad", flush=True)
            boo = False
        return boo

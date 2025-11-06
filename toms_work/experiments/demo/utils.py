import numpy as np


# NOTE: LLM generated function
def principal_angles(
    A: np.ndarray,
    B: np.ndarray,
    *,
    assume_orthonormal: bool = False,
    atol: float = 1e-10,
    rtol: float = 1e-8,
    degrees: bool = False,
) -> np.ndarray:
    """
    Compute principal angles θ_i ∈ [0, π/2] between the column-spaces of A and B.

    Parameters
    ----------
    A, B : (n×kA), (n×kB)
        Basis matrices whose columns span the subspaces.
    assume_orthonormal : bool
        If True, treat columns of A and B as already orthonormal and skip SVD/QR.
    atol, rtol : float
        Thresholds for rank determination when extracting orthonormal bases.
    degrees : bool
        If True, return angles in degrees. Otherwise radians.

    Returns
    -------
    theta : (m,) np.ndarray
        Principal angles sorted ascending, where m = min(rank(A), rank(B)).
    """
    if not assume_orthonormal:
        # Orthonormal bases via SVD (robust if A/B may be rank-deficient)
        UA, sA, _ = np.linalg.svd(A, full_matrices=False)
        UB, sB, _ = np.linalg.svd(B, full_matrices=False)
        tolA = max(atol, rtol * sA[0]) if sA.size else atol
        tolB = max(atol, rtol * sB[0]) if sB.size else atol
        rA = int((sA > tolA).sum())
        rB = int((sB > tolB).sum())
        QA = UA[:, :rA]
        QB = UB[:, :rB]
    else:
        QA, QB = A, B

    if QA.size == 0 or QB.size == 0:
        return np.array([], dtype=A.dtype)

    # Principal angles from singular values of QAᵀ QB
    s = np.linalg.svd(QA.T @ QB, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)  # numerical safety
    theta = np.arccos(s)  # in radians
    if degrees:
        theta = np.degrees(theta)
    return theta

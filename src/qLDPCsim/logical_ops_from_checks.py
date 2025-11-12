# logical_ops_from_checks.py
"""
Compute logical X and Z operators for a CSS quantum code defined by (Hx, Hz).
The code assumes binary matrices over GF(2).

We find:
 - basis vectors of the X-stabilizer space,
 - basis vectors of the Z-stabilizer space,
 - and basis vectors for the logical operator spaces (mod stabilizers).

"""

import numpy as np
from typing import Tuple, List


def gf2_nullspace(A: np.ndarray) -> np.ndarray:
    """
    Return a basis for the nullspace of A (mod 2), as rows.
    """
    A = A.copy() % 2
    m, n = A.shape
    rref = A.copy()
    pivots = []
    row = 0
    for col in range(n):
        # Find a pivot
        pivot = None
        for r in range(row, m):
            if rref[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        # Swap
        if pivot != row:
            rref[[pivot, row]] = rref[[row, pivot]]
        pivots.append(col)
        # Eliminate below and above
        for r in range(m):
            if r != row and rref[r, col] == 1:
                rref[r, :] ^= rref[row, :]
        row += 1
        if row == m:
            break

    free_cols = [c for c in range(n) if c not in pivots]
    null_basis = []
    for free in free_cols:
        v = np.zeros(n, dtype=int)
        v[free] = 1
        for i, p in enumerate(pivots):
            v[p] = rref[i, free]
        null_basis.append(v)
    return np.array(null_basis, dtype=int)


def logical_ops_from_css(Hx: np.ndarray, Hz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given CSS parity-check matrices Hx, Hz (binary),
    return (logical_X_ops, logical_Z_ops) as binary matrices
    whose rows correspond to logical operators.

    For a [[n,k]] code:
      k = n - rank(Hx) - rank(Hz)
    """

    n = Hx.shape[1]
    # Find nullspaces (over GF2)
    nx = gf2_nullspace(Hx)  # X operators commuting with X checks
    nz = gf2_nullspace(Hz)  # Z operators commuting with Z checks

    # Remove overlaps with stabilizer space
    # For CSS, stabilizers are rows of Hx (Z-checks) and Hz (X-checks)
    # Logical X space = null(Hx) / rowspace(Hz)
    # Logical Z space = null(Hz) / rowspace(Hx)

    def row_reduce_mod2(A):
        A = A.copy() % 2
        m, n = A.shape
        r = 0
        for c in range(n):
            piv = None
            for i in range(r, m):
                if A[i, c] == 1:
                    piv = i
                    break
            if piv is None:
                continue
            A[[r, piv]] = A[[piv, r]]
            for i in range(m):
                if i != r and A[i, c] == 1:
                    A[i] ^= A[r]
            r += 1
        return A[:r]

    def remove_dependents(space: np.ndarray, stabilizers: np.ndarray) -> np.ndarray:
        if len(space) == 0:
            return space
        space = space % 2
        stabilizers = row_reduce_mod2(stabilizers)
        independent = []
        for v in space:
            breakpoint()
            combo = np.vstack([stabilizers, v])
            reduced = row_reduce_mod2(combo)
            # if rank increases, v adds a new independent direction
            if len(reduced) > len(stabilizers):
                independent.append(v)
                stabilizers = reduced
        return np.array(independent, dtype=int)

    logical_X = remove_dependents(nx, Hz)
    breakpoint()
    logical_Z = remove_dependents(nz, Hx)
    return logical_X, logical_Z



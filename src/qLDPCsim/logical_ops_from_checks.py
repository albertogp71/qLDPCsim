"""
Copyright (c) 2025, Alberto G. Perotti
All rights reserved.

Compute logical X and Z operators for a CSS quantum code defined by (Hx, Hz).
The code assumes binary matrices over GF(2).

We find:
 - basis vectors of the X-stabilizer space,
 - basis vectors of the Z-stabilizer space,
 - and basis vectors for the logical operator spaces (mod stabilizers).

"""

import numpy as np
import gf2math
from typing import Tuple

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
    nx = gf2math.nullSpace(Hx)  # X operators commuting with X checks
    nz = gf2math.nullSpace(Hz)  # Z operators commuting with Z checks

    # Remove overlaps with stabilizer space
    # For CSS, stabilizers are rows of Hx (Z-checks) and Hz (X-checks)
    # Logical X space = null(Hx) / rowspace(Hz)
    # Logical Z space = null(Hz) / rowspace(Hx)


    logical_X = remove_dependents(nx, Hz)
    logical_Z = remove_dependents(nz, Hx)
    return logical_X, logical_Z


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
        combo = np.vstack([stabilizers, v])
        reduced = row_reduce_mod2(combo)
        # if rank increases, v adds a new independent direction
        if len(reduced) > len(stabilizers):
            independent.append(v)
            stabilizers = reduced
    return np.array(independent, dtype=int)

"""
Copyright (c) 2025, Alberto G. Perotti
All rights reserved.

Compute logical X and Z operator bases for general CSS codes.

Given parity-check matrices Hx and Hz satisfying Hx @ Hz.T = 0 mod 2,
this module computes:
    - stabilizer generators (rows of Hx, Hz)
    - logical operator bases for X and Z

Author: ChatGPT (2025)
"""

import numpy as np
from itertools import product

# -----------------------------------------------------------
# GF(2) linear algebra utilities
# -----------------------------------------------------------
def gf2_rank(A: np.ndarray) -> int:
    """Compute rank of binary matrix A over GF(2)."""
    A = A.copy() % 2
    m, n = A.shape
    rank = 0
    row = 0
    for col in range(n):
        if row >= m:
            break
        pivot = np.where(A[row:, col] == 1)[0]
        if len(pivot) == 0:
            continue
        pivot = pivot[0] + row
        A[[row, pivot]] = A[[pivot, row]]
        for i in range(m):
            if i != row and A[i, col] == 1:
                A[i, :] ^= A[row, :]
        row += 1
        rank += 1
    return rank


def gf2_null_space(A: np.ndarray) -> np.ndarray:
    """Compute null space of A over GF(2) as binary array (columns = basis vectors)."""
    A = A.copy() % 2
    m, n = A.shape
    pivots = []
    r = 0
    for c in range(n):
        pivot = np.where(A[r:, c] == 1)[0]
        if len(pivot) == 0:
            continue
        pivot = pivot[0] + r
        A[[r, pivot]] = A[[pivot, r]]
        for i in range(m):
            if i != r and A[i, c] == 1:
                A[i, :] ^= A[r, :]
        pivots.append(c)
        r += 1
        if r == m:
            break
    free_vars = [i for i in range(n) if i not in pivots]
    null_basis = []
    for fv in free_vars:
        x = np.zeros(n, dtype=int)
        x[fv] = 1
        for i, c in enumerate(pivots):
            if A[i, fv] == 1:
                x[c] = 1
        null_basis.append(x)
    if len(null_basis) == 0:
        return np.zeros((n, 0), dtype=int)
    return np.array(null_basis).T


def gf2_rowspace_basis(A: np.ndarray) -> np.ndarray:
    """Return independent rowspace basis of A over GF(2)."""
    A = A.copy() % 2
    m, n = A.shape
    pivots = []
    r = 0
    for c in range(n):
        pivot = np.where(A[r:, c] == 1)[0]
        if len(pivot) == 0:
            continue
        pivot = pivot[0] + r
        A[[r, pivot]] = A[[pivot, r]]
        for i in range(m):
            if i != r and A[i, c] == 1:
                A[i, :] ^= A[r, :]
        pivots.append(c)
        r += 1
        if r == m:
            break
    return A[:r, :]


# -----------------------------------------------------------
# CSS logical operator computation
# -----------------------------------------------------------
def logical_ops_css(Hx: np.ndarray, Hz: np.ndarray):
    """
    Compute logical X and Z operator bases for CSS code.

    Args:
        Hx : (r_x, n) binary ndarray, X-type parity-check matrix
        Hz : (r_z, n) binary ndarray, Z-type parity-check matrix
             must satisfy (Hx @ Hz.T) % 2 == 0.

    Returns:
        logical_X_ops : (k, n) binary ndarray
        logical_Z_ops : (k, n) binary ndarray
        k : number of logical qubits
    """
    Hx = Hx % 2
    Hz = Hz % 2
    n = Hx.shape[1]
    assert Hz.shape[1] == n, "Hx and Hz must have same number of columns"

    # Check commutativity
    if np.any((Hx @ Hz.T) % 2):
        raise ValueError("Hx and Hz must satisfy Hx @ Hz.T = 0 mod 2 (CSS condition)")

    rank_x = gf2_rank(Hx)
    rank_z = gf2_rank(Hz)
    k = n - rank_x - rank_z
    print(f"n={n}, rank(Hx)={rank_x}, rank(Hz)={rank_z}, logical qubits k={k}")

    # Compute null spaces
    Cx = gf2_null_space(Hx)   # X-code = ker(Hx)
    Cz = gf2_null_space(Hz)   # Z-code = ker(Hz)

    # Rowspaces (stabilizers)
    Sx = gf2_rowspace_basis(Hx)
    Sz = gf2_rowspace_basis(Hz)

    # Logical X = Cz / rowspace(Hx)
    # Logical Z = Cx / rowspace(Hz)
    logical_X_ops = coset_basis(Cz, Sx)
    logical_Z_ops = coset_basis(Cx, Sz)

    return logical_X_ops, logical_Z_ops, k


# -----------------------------------------------------------
# Utility: compute basis for quotient space A / B
# -----------------------------------------------------------
def coset_basis(A_basis: np.ndarray, B_basis: np.ndarray):
    """
    Compute independent coset representatives of A / B
    where A = span(A_basis), B = span(B_basis).
    Small-scale brute force, suitable for pedagogical or small codes.
    """
    if A_basis.size == 0:
        return np.zeros((0, B_basis.shape[1]), dtype=int)

    n = A_basis.shape[0]
    dimA = A_basis.shape[1]
    dimB = B_basis.shape[0] if B_basis.ndim > 1 else 1
    if dimA > 15:
        raise ValueError("Too large to brute-force cosets")

    A_vecs = [(np.sum([b[i] * A_basis[:, i] for i in range(A_basis.shape[1])], axis=0) % 2)
              for b in product([0, 1], repeat=A_basis.shape[1])]
    B_vecs = [(np.sum([b[i] * B_basis[i, :] for i in range(B_basis.shape[0])], axis=0) % 2)
              for b in product([0, 1], repeat=gf2_rank(B_basis))]

    logical_vecs = []
    for v in A_vecs:
        if not any(np.all((v + w) % 2 == 0) for w in B_vecs):
            logical_vecs.append(v)
    return np.unique(np.array(logical_vecs), axis=0)


# -----------------------------------------------------------
# Example: Steane [[7,1,3]] code (non dual trivial case)
# -----------------------------------------------------------
if __name__ == "__main__":
    Hx = np.array([
        [1,0,0,1,0,1,1],
        [0,1,0,1,1,0,1],
        [0,0,1,0,1,1,1]
    ], dtype=int)

    Hz = Hx.copy()  # For Steane, Hx=Hz

    LX, LZ, k = logical_ops_css(Hx, Hz)
    print("\nLogical X operators:\n", LX)
    print("\nLogical Z operators:\n", LZ)

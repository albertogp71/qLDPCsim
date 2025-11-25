"""
Copyright (c) 2025, Alberto G. Perotti
All rights reserved.

Some useful GF2 math functions.
"""

import numpy as np



def nullSpace(A: np.ndarray) -> np.ndarray:
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






def rowBasis(M: np.ndarray) -> np.ndarray:
    """
    Return an array whose rows form a basis of the row space of matrix M.
    Zero rows and dependent rows are removed. 
    Order is row-echelon.
    """
    A = (np.asarray(M, dtype=np.uint8) & 1).copy()
    m, n = A.shape
    r = 0
    for c in range(n):
        # Find pivot row with 1 in column c at or below r
        piv = None
        for i in range(r, m):
            if A[i, c] == 1:
                piv = i
                break
        if piv is None:
            continue
        # Swap into row r
        if piv != r:
            A[[r, piv]] = A[[piv, r]]
        # Eliminate all other 1s in column c
        for i in range(m):
            if i != r and A[i, c] == 1:
                A[i] ^= A[r]
        r += 1
        if r == m:
            break
    if r == 0:
        return np.zeros((0, n), dtype=np.uint8)
    return A[:r].copy()



def rank(A: np.ndarray) -> int:
    """
    Compute the rank of a binary matrix A over GF(2).
    
    Parameters
    ----------
    A : np.ndarray
        A 2D binary matrix.

    Returns
    -------
    int
        Rank of A over GF(2).
    """
    A = (np.asarray(A, dtype=np.uint8) & 1).copy()
    m, n = A.shape
    Arank = 0
    row = 0

    for col in range(n):
        # Find pivot in column col at or below row
        pivot = None
        for r in range(row, m):
            if A[r, col]:
                pivot = r
                break
        if pivot is None:
            # No pivot in this column
            continue

        # Swap pivot row into place
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]

        # Eliminate other rows having a 1 in this column
        for r in range(m):
            if r != row and A[r, col]:
                A[r] ^= A[row]

        row += 1
        Arank += 1
        if row == m:
            break

    return Arank




def systematic_form(H):
    """
    Transform a full-rank GF(2) parity-check matrix H (r x n)
    into systematic form [I_r | A] using column permutations only.

    Returns:
        H_sys  -- systematic matrix [I_r | A]
        perm  -- permutation vector of length n such that H_sys = H[:, perm]
    """

    H = (np.array(H, dtype=np.uint8) & 1).copy()
    r, n = H.shape

    # Keep track of column permutations
    perm = np.arange(n, dtype=int)
    pivot_row = 0

    for col in range(n):
        if pivot_row == r:
            break  # all pivots found

        # Find row >= pivot_row that has a 1 in column 'col'
        pivot = None
        for row in range(pivot_row, r):
            if H[row, col] == 1:
                pivot = row
                break

        if pivot is None:
            continue  # no pivot here

        # Swap rows to bring pivot into pivot_row
        if pivot != pivot_row:
            H[[pivot, pivot_row]] = H[[pivot_row, pivot]]

        # Eliminate other rows
        for row in range(r):
            if row != pivot_row and H[row, col] == 1:
                H[row] ^= H[pivot_row]

        # Now pivot at (pivot_row, col). If the pivot column is not in the
        # correct place in systematic form, swap columns.
        if col != pivot_row:
            # swap columns col <-> pivot_row
            H[:, [col, pivot_row]] = H[:, [pivot_row, col]]
            perm[[col, pivot_row]] = perm[[pivot_row, col]]

        pivot_row += 1

    # Check rank
    if pivot_row < r:
        raise ValueError("Matrix is not full-rank; cannot form systematic representation.")

    return H, perm



"""
Copyright (c) 2025, Alberto G. Perotti
All rights reserved.

Library of Parity Check Matricx (PCM) pairs (Hx, Hz) of quantum CSS codes.

- Shor (9-qubit) code [1]
- Steane (7-qubit) code [2]
- Laflamme (5-qubit) code [3]
- QC-LDPC lifted codes from [4].

REFERENCES
[1] Phys. Rev. A 52, R2493(R). https://doi.org/10.1103%2FPhysRevA.52.R2493.
[2] https://doi.org/10.1098/rspa.1996.0136.
[3] https://arxiv.org/abs/quant-ph/9602019
[4] Quantum 6, 767 (2022).
[5] DOI: 10.1109/TIT.2004.838370.
"""

import numpy as np
from typing import Tuple

def shor_code() -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (Hx, Hz) parity-check matrices for the 9-qubit Shor code (CSS).
    The Shor code is constructed by concatenating three 3-qubit repetition codes
    in both Z and X bases.
    """

    # Z-checks (detect X errors) — three intra-block parity checks:
    # (0,1), (1,2); (3,4),(4,5); (6,7),(7,8)
    Hz = np.array([
        [1,1,0, 0,0,0, 0,0,0],
        [0,1,1, 0,0,0, 0,0,0],
        [0,0,0, 1,1,0, 0,0,0],
        [0,0,0, 0,1,1, 0,0,0],
        [0,0,0, 0,0,0, 1,1,0],
        [0,0,0, 0,0,0, 0,1,1]], dtype=int)

    # X-checks (detect Z errors) — three “across-block” parity checks at each of the 3 positions:
    # Compare block1-block2, and block2-block3:
    Hx = np.array([
        [1,1,1, 1,1,1, 0,0,0],
        [0,0,0, 1,1,1, 1,1,1]], dtype=int)

    return Hx, Hz


def steane_code() -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (Hx, Hz) for the [[7,1,3]] Steane code.

    """
    H = np.array([
        [1,0,0,1,0,1,1],
        [0,1,0,1,1,0,1],
        [0,0,1,0,1,1,1]], dtype=int)
    # In the usual CSS form, Hx checks Z errors, Hz checks X errors:
    Hx = H.copy()
    Hz = H.copy()
    return Hx, Hz



def qc_ldpc_tanner_code() -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (Hx, Hz) for the quasi-cyclic Tanner LDPC code from [5].

    Returns:
        Hx, Hz : binary parity-check matrices after lifting
    """

    def expand_base(B: np.ndarray, L: int) -> np.ndarray:
        m_b, n_b = B.shape
        H = np.zeros((m_b * L, n_b * L), dtype=int)
        I = np.eye(L, dtype=int)
        for i in range(m_b):
            for j in range(n_b):
                shift = B[i, j]
                if shift >= 0:
                    H[i*L:(i+1)*L, j*L:(j+1)*L] = np.roll(I, shift, axis=1)
        return H

    L = 31
    B = np.array([
        [ 1,  2,  4,  8, 16],
        [ 5, 10, 20,  9, 18],
        [25, 19,  7, 14, 28]], dtype=int)

    Btc = L - np.transpose(B)
    m_b, n_b = B.shape
    Bx = -1 + np.concat((np.kron(B+1, np.identity(n_b)), np.kron(np.identity(m_b), Btc+1)), axis=1)
    Bz = -1 + np.concat((np.kron(np.identity(n_b), B+1), np.kron(Btc+1, np.identity(m_b))), axis=1)

    Hx = expand_base(Bx, L)
    Hz = expand_base(Bz, L)

    return Hx, Hz






def qc_ldpc_lifted_code(family: str = "LP04", 
                         index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (Hx, Hz) for the quasi-cyclic lifted product (LP) LDPC codes from [4].

    Returns:
        Hx, Hz : binary parity-check matrices after lifting
    """

    def expand_base(B: np.ndarray, L: int) -> np.ndarray:
        m_b, n_b = B.shape
        H = np.zeros((m_b * L, n_b * L), dtype=int)
        I = np.eye(L, dtype=int)
        for i in range(m_b):
            for j in range(n_b):
                shift = B[i, j]
                if shift >= 0:
                    H[i*L:(i+1)*L, j*L:(j+1)*L] = np.roll(I, shift, axis=1)
        return H



    match family:
        case "LP04":
            if index == 0:
                L, dmin = 7, 10
                B = np.array([
                    [0, 0, 0, 0],
                    [0, 1, 2, 5],
                    [0, 6, 3, 1]], dtype=int)
            elif index == 1:
                L, dmin = 9, 12
                B = np.array([
                    [0, 0, 0, 0],
                    [0, 1, 6, 7],
                    [0, 4, 5, 2]], dtype=int)
            elif index == 2:
                L, dmin = 17, 18
                B = np.array([
                    [0,  0,  0,  0],
                    [0,  1,  2, 11],
                    [0,  8, 12, 13]], dtype=int)
            elif index == 3:
                L, dmin = 19, 20
                B = np.array([
                    [0,  0,  0,  0],
                    [0,  2,  6,  9],
                    [0, 16,  7, 11]], dtype=int)
            else:
                raise ValueError("qc_ldpc_lifted_codes: index out of bounds for code family LP04.")

        case "LP118":
            if index == 0:
                L, dmin = 16, 12
                B = np.array([
                    [0,  0,  0,  0,  0],
                    [0,  2,  4,  7, 11],
                    [0,  3, 10, 14, 15]], dtype=int)
            elif index == 1:
                L, dmin = 21, 16
                B = np.array([
                    [0,  0,  0,  0,  0],
                    [0,  4,  5,  7, 17],
                    [0, 14, 18, 12, 11]], dtype=int)
            elif index == 2:
                L, dmin = 30, 20
                B = np.array([
                    [0,  0,  0,  0,  0],
                    [0,  2, 14, 24, 25],
                    [0, 16, 11, 14, 13]], dtype=int)
            else:
                raise ValueError("qc_ldpc_lifted_codes: index out of bounds for code family LP118.")
        case _:
            raise ValueError("qc_ldpc_lifted_codes: unrecognized code family.")

    Btc = L - np.transpose(B)
    m_b, n_b = B.shape
    Bx = -1 + np.concat((np.kron(B+1, np.identity(n_b)), np.kron(np.identity(m_b), Btc+1)), axis=1)
    Bz = -1 + np.concat((np.kron(np.identity(n_b), B+1), np.kron(Btc+1, np.identity(m_b))), axis=1)

    Hx = expand_base(Bx, L)
    Hz = expand_base(Bz, L)

    return Hx, Hz


# --- If run as script, print the shapes as a quick check ---
if __name__ == "__main__":
    print("Shor code Hx, Hz shapes:", shor_code()[0].shape, shor_code()[1].shape)
    print("Steane code Hx, Hz shapes:", steane_code()[0].shape, steane_code()[1].shape)
    Hx_l, Hz_l = qc_ldpc_lifted_code()
    print("QC-LDPC lifted Hx, Hz shapes:", Hx_l.shape, Hz_l.shape)

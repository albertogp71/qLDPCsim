# qLDPClib.py

"""
Predefined quantum CSS codes via parity-check matrix pairs (Hx, Hz).

- Shor (9-qubit) code
- Steane (7-qubit) code
- QC-LDPC lifted codes from [1].

REFERENCES
[1] Quantum 6, 767 (2022).
"""

import numpy as np
from typing import Tuple

def shor_code() -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (Hx, Hz) parity-check matrices for the 9-qubit Shor code (CSS).
    The Shor code is constructed by concatenating three 3-qubit repetition codes
    in both Z and X bases.
    """
    # The Shor code is often described as:
    # - Three blocks of 3 qubits; within each block, there is a Z-type repetition code to detect bit-flips,
    # - Then an X-type repetition code across the three blocks to detect phase-flips.
    #
    # One common representation:
    # For bit-flip (Z checks) we have 3 checks per block:
    #   Z1 Z2, Z2 Z3 in each block => total 3 × 2 = 6 Z-checks
    # For phase-flip (X checks) we have two checks across blocks for each position:
    #   X on qubit pos j in block1 vs block2, and block2 vs block3, for j = 1,2,3 => 3×2 = 6 X-checks
    #
    # But many rows are redundant; a minimal independent set yields fewer checks.
    #
    # For simplicity, here is a commonly used reduced form:
    # We index qubits as 0..8, grouped in blocks [0,1,2], [3,4,5], [6,7,8].

    # Z-checks (detect X errors) — three intra-block parity checks:
    # (0,1), (1,2); (3,4),(4,5); (6,7),(7,8)
    Hz = np.array([
        [1,1,0, 0,0,0, 0,0,0],
        [0,1,1, 0,0,0, 0,0,0],
        [0,0,0, 1,1,0, 0,0,0],
        [0,0,0, 0,1,1, 0,0,0],
        [0,0,0, 0,0,0, 1,1,0],
        [0,0,0, 0,0,0, 0,1,1],
    ], dtype=int)

    # X-checks (detect Z errors) — three “across-block” parity checks at each of the 3 positions:
    # Compare block1-block2, and block2-block3, for each position j:
    Hx = np.array([
        [1,0,0, 1,0,0, 0,0,0],
        [0,0,0, 1,0,0, 1,0,0],
        [0,1,0, 0,1,0, 0,0,0],
        [0,0,0, 0,1,0, 0,1,0],
        [0,0,1, 0,0,1, 0,0,0],
        [0,0,0, 0,0,1, 0,0,1],
    ], dtype=int)

    return Hx, Hz


def steane_code() -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (Hx, Hz) for the [[7,1,3]] Steane code.

    The Steane code is CSS built from the classical [7,4,3] Hamming code.
    The classical parity-check matrix H (3×7) is:

        H = [[1,0,0,1,0,1,1],
             [0,1,0,1,1,0,1],
             [0,0,1,0,1,1,1]]

    Then for the quantum CSS form:
    - Hx = H (for Z-error checks),
    - Hz = H (for X-error checks),
      i.e. symmetrical in Steane’s case.
    """
    H = np.array([
        [1,0,0,1,0,1,1],
        [0,1,0,1,1,0,1],
        [0,0,1,0,1,1,1],
    ], dtype=int)
    # In the usual CSS form, Hx checks Z errors, Hz checks X errors:
    Hx = H.copy()
    Hz = H.copy()
    return Hx, Hz


def qc_ldpc_lifted_codes(family: str = "LP04", 
                         index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (Hx, Hz) for the quasi-cyclic lifted LDPC codes from [1].

    Returns:
        Hx, Hz : binary parity-check matrices after lifting
    """
    # --- Example small protograph (toy) ---
    # Suppose the paper gives base matrices Bx and Bz, dimension m_base × n_base,
    # where each entry is either -1 or a shift index in [0, L-1].
    # Here we set up a toy base:
    Bx = np.array([
        [0, 2, -1],
        [1, -1, 3],
    ], dtype=int)
    Bz = np.array([
        [ -1, 1, 0 ],
        [ 2, 3, -1 ],
    ], dtype=int)

    L = 5  # example circulant size (must be set according to paper’s code parameters)

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
                B = np.array[
                    [0, 0, 0, 0],
                    [0, 1, 2, 5],
                    [0, 6, 3, 1]]
            elif index == 1:
                L, dmin = 9, 12
                B = np.array[
                    [0, 0, 0, 0],
                    [0, 1, 6, 7],
                    [0, 4, 5, 2]]
            elif index == 2:
                L, dmin = 17, 18
                B = np.array[
                    [0,  0,  0,  0],
                    [0,  1,  2, 11],
                    [0,  8, 12, 13]]
            elif index == 3:
                L, dmin = 19, 20
                B = np.array[
                    [0,  0,  0,  0],
                    [0,  2,  6,  9],
                    [0, 16,  7, 11]]
            else:
                raise ValueError("qc_ldpc_lifted_codes: index out of bounds for code family LP04.")

        case "LP118":
            if index == 0:
                L, dmin = 16, 12
                B = np.array[
                    [0,  0,  0,  0,  0],
                    [0,  2,  4,  7, 11],
                    [0,  3, 10, 14, 15]]
            elif index == 1:
                L, dmin = 21, 16
                B = np.array[
                    [0,  0,  0,  0,  0],
                    [0,  4,  5,  7, 17],
                    [0, 14, 18, 12, 11]]
            elif index == 2:
                L, dmin = 30, 20
                B = np.array[
                    [0,  0,  0,  0,  0],
                    [0,  2, 14, 24, 25],
                    [0, 16, 11, 14, 13]]
            else:
                raise ValueError("qc_ldpc_lifted_codes: index out of bounds for code family LP118.")
        case _:
            raise ValueError("qc_ldpc_lifted_codes: unrecognized code family.")

    Btc = L - np.transpose(B)
    m_b, n_b = B.shape
    Bx = np.concat((np.kron(B, np.identity(n_b)), np.kron(np.identity(m_b), Btc)), axis=1)
    Bz = np.concat((np.kron(np.identity(n_b), B), np.kron(Btc, np.identity(m_b))), axis=1)

    Hx = expand_base(Bx, L)
    Hz = expand_base(Bz, L)

    return Hx, Hz


# --- If run as script, print the shapes as a quick check ---
if __name__ == "__main__":
    print("Shor code Hx, Hz shapes:", shor_code()[0].shape, shor_code()[1].shape)
    print("Steane code Hx, Hz shapes:", steane_code()[0].shape, steane_code()[1].shape)
    Hx_l, Hz_l = qc_ldpc_lifted_code()
    print("QC-LDPC lifted Hx, Hz shapes:", Hx_l.shape, Hz_l.shape)

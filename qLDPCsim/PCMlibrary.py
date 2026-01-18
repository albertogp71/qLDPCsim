"""
Copyright (c) 2025, Alberto G. Perotti
All rights reserved.

Library of Parity Check Matricx (PCM) pairs (Hx, Hz) of quantum CSS codes.

- Shor (9-qubit) code [1]
- Steane (7-qubit) code [2]
- Bicycle code
- QC-LDPC Tanner code [4]
- QC-LDPC lifted codes from [4].

REFERENCES
[1] P. Shor, "Scheme for reducing decoherence in quantum computer memory", 
    Phys. Rev. A 52, R2493(R). https://doi.org/10.1103%2FPhysRevA.52.R2493.
[2] A. Steane, "Multiple Particle Interference and Quantum Error Correction",
    Online: https://arxiv.org/abs/quant-ph/9601029 .
[3] R. Laflamme, C. Miquel, J.P. Paz, W. H. Zurek, "Perfect Quantum Error 
    Correction Code", Online: https://arxiv.org/abs/quant-ph/9602019 .
[4] N. Raveendran, N. Rengaswamy, F. Rozpędek, A. Raina, L. Jiang, B. Vasić, 
    "Finite Rate QLDPC-GKP Coding Scheme that Surpasses the CSS Hamming Bound",
    Quantum 6, 767 (2022).
[5] R.M. Tanner; D. Sridhara; A. Sridharan; T.E. Fuja; D.J. Costello, "LDPC
    block and convolutional codes based on circulant matrices", IEEE Trans.
    Inf. Theory, vol. 50, no. 12, Dec. 2004. DOI: 10.1109/TIT.2004.838370.
[6] D.J.C. MacKay, G. Mitchison, P.L. McFadden, "Sparse-Graph Codes for Quantum 
    Error-Correction", Online: https://arxiv.org/pdf/quant-ph/0304161 .
[7] P. Panteleev, G. Kalachev, "Degenerate Quantum LDPC Codes With Good Finite 
    Length Performance", Quantum 5, 585.
"""

import numpy as np
from typing import Tuple



def shor_code() -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (Hx, Hz) parity-check matrices for the 9-qubit Shor code [1].
    The Shor code is constructed by concatenating three 3-qubit repetition 
    codes in both Z and X bases.
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

    Hx = H.copy()
    Hz = H.copy()
    return Hx, Hz



def bicycle_code() -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (Hx, Hz) for the bicycle code [6].
    """
    c = np.zeros((1,73))
    c[0,[2,8,15,19,20,34,42,44,72]] = 1     # From [6], Figure 9. The indices form
                                            # a perfect difference set of size 73.
    C = np.concatenate([np.roll(c,i,axis=1) for i in range(c.shape[1])], axis=0)
    H0 = np.concatenate([C, C.transpose()], axis=1)
    Hx = H0.copy()
    Hz = H0.copy()
    return Hx, Hz



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



def qc_ldpc_tanner_code() -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (Hx, Hz) for the quasi-cyclic Tanner LDPC code from [5].
    Returns:
        Hx, Hz : binary parity-check matrices.
    """

    L = 31
    B = np.array([
        [ 1,  2,  4,  8, 16],
        [ 5, 10, 20,  9, 18],
        [25, 19,  7, 14, 28]], dtype=int)

    Btc = L - np.transpose(B)
    m_b, n_b = B.shape
    Bx = -1 + np.concatenate((np.kron(B+1, np.identity(n_b)), np.kron(np.identity(m_b), Btc+1)), axis=1)
    Bz = -1 + np.concatenate((np.kron(np.identity(n_b), B+1), np.kron(Btc+1, np.identity(m_b))), axis=1)

    Hx = expand_base(Bx, L)
    Hz = expand_base(Bz, L)

    return Hx, Hz



def qc_ldpc_lifted_code(family: str = "LP04", 
                         index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (Hx, Hz) for the quasi-cyclic lifted product (LP) LDPC codes from [4].

    Returns:
        Hx, Hz : binary parity-check matrices.
    """

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
    Bx = -1 + np.concatenate((np.kron(B+1, np.identity(n_b)), np.kron(np.identity(m_b), Btc+1)), axis=1)
    Bz = -1 + np.concatenate((np.kron(np.identity(n_b), B+1), np.kron(Btc+1, np.identity(m_b))), axis=1)

    Hx = expand_base(Bx, L)
    Hz = expand_base(Bz, L)

    return Hx, Hz




def PK(code: str = "A1" 
       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    The QLDPC codes defined in [7].

    Parameters
    ----------
    code : str
        The type of code.
        "An" with n = 1,...,6 gives the generlized bycicle codes.
        "Bn" with n = 1,...,3 gives the generlized hypergraph product codes.

    Returns
    -------
        Hx, Hz : binary parity-check matrices.
    """



    match code[0]:
        case 'A':
            match code:
                case "A1":
                    l = 127                                         # Circulant size
                    a = np.zeros((l))
                    b = np.zeros((l))
                    a[[0,15,20,28,66]] = 1
                    b[[0,58,59,100,121]] = 1
                case "A2":
                    l = 63                                          # Circulant size
                    a = np.zeros((l))
                    b = np.zeros((l))
                    a[[0,1,14,16,22]] = 1
                    b[[0,3,13,20,42]] = 1
                case "A3":
                    l = 24                                          # Circulant size
                    a = np.zeros((l))
                    b = np.zeros((l))
                    a[[0,2,8,15]] = 1
                    b[[0,2,12,17]] = 1
                case "A4":
                    l = 23                                          # Circulant size
                    a = np.zeros((l))
                    b = np.zeros((l))
                    a[[0,5,8,12]] = 1
                    b[[0,1,5,7]] = 1
                case "A5":
                    l = 90                                          # Circulant size
                    a = np.zeros((l))
                    b = np.zeros((l))
                    a[[0,28,80,89]] = 1
                    b[[0,2,21,25]] = 1
                case "A6":
                    l = 450                                         # Circulant size
                    a = np.zeros((l))
                    b = np.zeros((l))
                    a[[0,97,372,425]] = 1
                    b[[0,50,265,390]] = 1
                case _:
                    raise ValueError("PK: unrecognized code family An.")
            A = np.stack([np.roll(a, i) for i in np.arange(l)], axis=1)      # Form circulant matrix
            B = np.stack([np.roll(b, i) for i in np.arange(l)], axis=1)      # Form circulant matrix
        case 'B':
            match code:
                case "B1":
                    l = 7
                    L = 63
                    a = -np.ones((l))
                    a[0:3] = [27, 54, 0]
                    A = expand_base(np.stack([np.roll(L-a, i) for i in np.arange(l)], axis=1), L)       # Expand base with L-a as expand_base() rolls on axis=1
                    b = np.zeros((L))
                    b[[0,1,6]] = 1
                    B = np.kron(np.eye(l), np.stack([np.roll(b, i) for i in np.arange(L)], axis=1))     # Form circulant matrix
                case "B2":
                    l = 7
                    L = 63
                    a = -np.ones((l))
                    a[0:5] = [27, 0, 27, 18, 0]
                    A = expand_base(np.stack([np.roll(L-a, i) for i in np.arange(l)], axis=1), L)
                    b = np.zeros((L))
                    b[[0,1,6]] = 1
                    B = np.kron(np.eye(l), np.stack([np.roll(b, i) for i in np.arange(L)], axis=1))     # Form circulant matrix
                case "B3":
                    l = 5
                    L = 127
                    a = np.array([
                        [0, -1, 51, 52, -1],
                        [-1, 0, -1, 111, 20],
                        [0, -1, 98, -1, 122],
                        [0, 80, -1, 119, -1],
                        [-1, 0, 5, -1, 106]], dtype=int)
                    A = expand_base(L-a, L)
                    b = np.zeros((L))
                    b[[0,1,7]] = 1
                    B = np.kron(np.eye(l), np.stack([np.roll(b, i) for i in np.arange(L)], axis=1))     # Form circulant matrix
                case _:
                    raise ValueError("PK: unrecognized code family Bn.")
        case _:
            raise ValueError("PK: unrecognized code family.")


    Hx = np.concatenate((A,B), axis=1).astype(np.int8)            
    Hz = np.concatenate((B.transpose(),A.transpose()), axis=1).astype(np.int8)

    return Hx, Hz


if __name__ == "__main__":
    print("Shor code Hx, Hz shapes:", shor_code()[0].shape, shor_code()[1].shape)
    print("Steane code Hx, Hz shapes:", steane_code()[0].shape, steane_code()[1].shape)
    Hx, Hz = qc_ldpc_lifted_code()
    print("QC-LDPC lifted Hx, Hz shapes:", Hx.shape, Hz.shape)

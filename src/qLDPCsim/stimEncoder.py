"""
Copyright (c) 2025, Alberto G. Perotti
All rights reserved.

Created on Mon Nov 17 13:17:38 2025

@author: AlbertoGP71
"""

import numpy as np
import stim
from qLDPCsim import gf2math

def css_ldpc_encoder_no_tableau(Hx: np.ndarray, Hz: np.ndarray) -> stim.Circuit:
    """
    Construct a Stim encoding circuit from a CSS LDPC code using only
    elementary Clifford gates (H, S, CNOT, SWAP), with NO use of Stim's tableau.

    Encoder = inverse of the circuit that reduces stabilizers to canonical form.
    """

    n = Hx.shape[1]
    S = []  # symplectic stabilizer rows

    # ---------------------------------------------------------------
    # Build stabilizer symplectic matrix: [X | Z]
    # Hx → Z stabilizers
    # Hz → X stabilizers
    # ---------------------------------------------------------------
    for row in Hx:
        S.append(np.concatenate([np.zeros(n, dtype=int), row % 2]))
    for row in Hz:
        S.append(np.concatenate([row % 2, np.zeros(n, dtype=int)]))

    S = np.array(S) % 2  # (r × 2n) matrix
    r = S.shape[0]

    circuit = stim.Circuit()
    applied_gates = []   # store operations to invert later

    # ---------------------------------------------------------------
    # Elementary symplectic update helpers
    # ---------------------------------------------------------------

    def apply_H(q):
        # H swaps X and Z on that qubit
        S[:, [q, q+n]] = S[:, [q+n, q]]
        applied_gates.append(("H", q))

    def apply_S(q):
        # S: X -> Y = XZ;  Z stays Z
        S[:, q+n] ^= S[:, q]
        applied_gates.append(("S", q))

    def apply_CNOT(c, t):
        # CNOT(c→t)
        # Xc → Xc Xt
        S[:, t] ^= S[:, c]
        # Zt → Zc Zt
        S[:, c+n] ^= S[:, t+n]
        applied_gates.append(("CNOT", c, t))

    def apply_SWAP(a, b):
        S[:, [a, b]] = S[:, [b, a]]
        S[:, [a+n, b+n]] = S[:, [b+n, a+n]]
        applied_gates.append(("SWAP", a, b))

    # ---------------------------------------------------------------
    # Symplectic Gaussian elimination
    # ---------------------------------------------------------------
    # Goal: put stabilizer matrix into canonical CSS form
    # using only H, S, CNOT, SWAP
    # ---------------------------------------------------------------
    row = 0

    # --- First eliminate Z-part rows (Z stabilizers) ---
    for col in range(n):
        # find pivot in Z block
        piv = np.where(S[row:, col+n] == 1)[0]
        if len(piv) == 0:
            continue
        piv = piv[0] + row

        # swap rows
        if piv != row:
            S[[row, piv], :] = S[[piv, row], :]

        # eliminate others
        for rr in range(r):
            if rr != row and S[rr, col+n] == 1:
                S[rr, :] ^= S[row, :]

        row += 1
        if row == r:
            break

    # --- Next eliminate X-part rows (X stabilizers) ---
    for col in range(n):
        piv = np.where(S[row:, col] == 1)[0]
        if len(piv) == 0:
            continue
        piv = piv[0] + row

        # swap rows
        if piv != row:
            S[[row, piv], :] = S[[piv, row], :]

        # eliminate others
        for rr in range(r):
            if rr != row and S[rr, col] == 1:
                S[rr, :] ^= S[row, :]

        row += 1
        if row == r:
            break

    # ---------------------------------------------------------------
    # Build the encoder as the INVERSE of the reduction circuit
    # ---------------------------------------------------------------
    encoder = stim.Circuit()

    for gate in reversed(applied_gates):
        if gate[0] == "H":
            encoder.append_operation("H", gate[1])
        elif gate[0] == "S":
            encoder.append_operation("S_DAG", gate[1])  # inverse of S
        elif gate[0] == "CNOT":
            encoder.append_operation("CNOT", [gate[1], gate[2]])
        elif gate[0] == "SWAP":
            encoder.append_operation("SWAP", [gate[1], gate[2]])

    return encoder


def css_ldpc_encoder(Hx: np.ndarray, Hz: np.ndarray) -> stim.Circuit:
    """
    Build an encoder circuit for a CSS-type quantum LDPC code
    using Stim's tableau construction and only elementary gates.

    Input:
        Hx : Z-stabilizer parity check matrix   (rows = Z stabilizers)
        Hz : X-stabilizer parity check matrix   (rows = X stabilizers)

    Output:
        stim.Circuit implementing the encoder.
    """
    n = Hx.shape[1]

    stabilizers = []
    seen = set()
    

    
    # ----------------------------------------------------------
    # Build Z stabilizers from Hz  → Z on 1-entries
    # ----------------------------------------------------------
    for row in Hx:
        s = '+' + ''.join("X" if b else "I" for b in row)
        if s not in seen:
            stabilizers.append(stim.PauliString(s))
            seen.add(s)
            
    # ----------------------------------------------------------
    # Build X stabilizers from Hx  → Z on 1-entries
    # ----------------------------------------------------------
    for row in Hz:
        s = '+' + ''.join("Z" if b else "I" for b in row)
        if s not in seen:
            stabilizers.append(stim.PauliString(s))
            seen.add(s)

    # ---------------------------------------------------------
    # Validate stabilizers by checking if they commute
    # ---------------------------------------------------------
    # def commute(pa1, pa2):
    #     """
    #     Check commutation: pa1, pa2 are strings over I,X,Y,Z.
    #     Returns True if they commute.
    #     """
    #     anti = 0
    #     for a, b in zip(pa1, pa2):
    #         if a == 'I' or b == 'I': 
    #             continue
    #         if a != b:
    #             anti ^= 1
    #     return (anti == 0)
    # for i in range(len(stabilizers)):
    #     for j in range(i+1, len(stabilizers)):
    #         if not commute(stabilizers[i], stabilizers[j]):
    #             raise ValueError("Stabilizers set is not commuting")

    # ----------------------------------------------------------
    # Construct a stabilizer Tableau from generators
    # Stim will auto-complete with destabilizers + logicals
    # ----------------------------------------------------------
    T = stim.Tableau.from_stabilizers(stabilizers, allow_underconstrained=True)
    breakpoint()
    # ----------------------------------------------------------
    # The encoder is the *inverse* of the stabilizer tableau
    # ----------------------------------------------------------
    enc = T.to_circuit()
    

    return enc

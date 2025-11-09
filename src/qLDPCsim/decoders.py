# decoders.py

"""
Decoders for quantum LDPC codes

REFERENCES
[1] 
"""

import numpy as np
from typing import Tuple





# -----------------------------
# Simple naive syndrome->correction heuristic
# -----------------------------
def naive_greedy_decoder(H: np.ndarray, syndrome_bits: np.ndarray) -> np.ndarray:
    """
    A very simple greedy decoder for demonstration:
    - repeatedly pick a variable node connected to the currently nonzero syndrome with highest degree,
      flip that variable in the candidate error, update the syndrome, and repeat up to some limit.
    This is NOT a production-grade decoder, but serves as a placeholder.
    """
    m, n = H.shape
    residual = syndrome_bits.copy()
    est = np.zeros(n, dtype=np.int8)
    # build adjacency lists
    check_to_vars = [list(np.where(H[i, :] == 1)[0]) for i in range(m)]
    var_to_checks = [list(np.where(H[:, j] == 1)[0]) for j in range(n)]

    max_steps = n * 2
    steps = 0
    while (residual.sum() > 0) and (steps < max_steps):
        steps += 1
        # compute scores for variables: number of failing checks it touches
        scores = np.zeros(n, dtype=int)
        failing_checks = np.where(residual == 1)[0]
        for c in failing_checks:
            for v in check_to_vars[c]:
                scores[v] += 1
        if scores.max() == 0:
            break
        v = int(np.argmax(scores))
        # flip v
        est[v] ^= 1
        # update residual
        for c in var_to_checks[v]:
            residual[c] ^= 1
    return est





# ---------------------------------------------------------------------
# Min-Sum Decoder
# ---------------------------------------------------------------------
def min_sum_decode(H: np.ndarray, syndrome: np.ndarray, p_err: float, max_iter: int = 50, eps: float = 1e-9) -> np.ndarray:
    """
    Vectorized Min-Sum decoding for binary LDPC.

    Args:
        H : parity-check matrix (m x n)
        syndrome : length-m binary array
        p_err : prior error probability
    Returns:
        estimated error vector (0/1)
    """
    if H.size == 0 or syndrome.size == 0:
        return np.zeros(H.shape[1] if H.size else 0, dtype=np.int8)

    m, n = H.shape
    # Initialize messages
    L_ch = np.log((1 - p_err + eps) / (p_err + eps))
    msg_vc = np.zeros((m, n), dtype=np.float32)
    msg_vc[H == 1] = L_ch
    msg_cv = np.zeros_like(msg_vc)
    syn_sign = np.where(syndrome[:, None] == 1, -1.0, 1.0)

    for _ in range(max_iter):
        # Check node update
        abs_msg = np.abs(msg_vc)
        sign_msg = np.sign(msg_vc)
        sign_msg[sign_msg == 0] = 1.0
        prod_sign = np.prod(np.where(H == 1, sign_msg, 1.0), axis=1, keepdims=True)
        min_abs = np.min(np.where(H == 1, abs_msg, np.inf), axis=1, keepdims=True)
        min_abs[np.isinf(min_abs)] = 0.0
        msg_cv = np.where(H == 1, syn_sign * prod_sign * min_abs / (sign_msg + (1 - H)), 0.0)

        # Variable node update
        incoming_sum = np.sum(msg_cv, axis=0)
        posterior = L_ch + incoming_sum
        e_hat = (posterior < 0).astype(np.int8)
        if np.array_equal(syndrome, mod2(H.dot(e_hat))):
            return e_hat
        msg_vc = np.where(H == 1, L_ch + incoming_sum - msg_cv, 0.0)
    return e_hat


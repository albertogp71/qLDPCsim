# decoders.py

"""
Decoders for quantum LDPC codes

REFERENCES
[1] https://doi.org/10.48550/arXiv.2205.02341
[2] 
"""

import numpy as np
from typing import Tuple
import sys





# -----------------------------
# Simple naive syndrome->correction heuristic
# -----------------------------
def naive_greedy_decoder(H: np.ndarray, syndrome_bits: np.ndarray) -> np.ndarray:
    """
    A very simple greedy decoder for demonstration:
    - repeatedly pick a variable node connected to the currently nonzero syndrome with highest degree,
      flip that variable in the candidate error, update the syndrome, and repeat up to some limit.
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
# Min-Sum decoder [1]
# ---------------------------------------------------------------------
def min_sum_decoder(H: np.ndarray, syndrome: np.ndarray, p: float, max_iter: int = 50, beta: float = 0.75, eps: float = 1e-9) -> np.ndarray:
    """
    Min-Sum decoding for binary LDPC.

    Args:
        H : parity-check matrix (m x n)
        syndrome : length-m binary array
        p : prior error probability
    Returns:
        estimated error vector (0/1)
    """
    if H.size == 0 or syndrome.size == 0:
        return np.zeros(H.shape[1] if H.size else 0, dtype=np.int8)

    m, n = H.shape
    # Initialize messages
    L_ch = np.log((1 - p) / max(p, eps))
    msg_v2c = np.zeros((m, n), dtype=np.float32)
    msg_v2c[H == 1] = L_ch
    msg_c2v = np.zeros_like(msg_v2c)
    syn_sign = np.where(syndrome[:, None] == 1, -1.0, 1.0)

    for n_iter in range(max_iter):
        # Check node update
        abs_msg = np.abs(msg_v2c)
        sign_msg = np.sign(msg_v2c)
        sign_msg[sign_msg == 0] = 1.0
        prod_sign = np.prod(np.where(H == 1, sign_msg, 1.0), axis=1, keepdims=True)
        min_abs = np.min(np.where(H == 1, abs_msg, np.inf), axis=1, keepdims=True)
        minidx = np.argmin(np.where(H == 1, abs_msg, np.inf), axis=1)
        abs_msg2 = abs_msg
        abs_msg2[range(abs_msg.shape[0]), minidx] = np.inf
        min_abs2 = np.min(np.where(H == 1, abs_msg, np.inf), axis=1, keepdims=True)
        min_abs[np.isinf(min_abs)] = 0.0
        min_abs2[np.isinf(min_abs2)] = 0.0
        msg_c2v = np.where(np.logical_and(H == 1, np.abs(msg_v2c) != min_abs), beta * syn_sign * prod_sign * min_abs / (sign_msg + (1 - H)), np.inf)
        msg_c2v = np.where(np.logical_and(np.isinf(msg_c2v), np.abs(msg_v2c) == min_abs), beta * syn_sign * prod_sign * min_abs2 / (sign_msg + (1 - H)), msg_c2v)
        msg_c2v[np.isinf(msg_c2v)] = 0.0

        # Variable node update
        VNsum = np.sum(msg_c2v, axis=0)
        posterior = L_ch + VNsum
        e_hat = (posterior < 0).astype(np.int8)
        if np.array_equal(syndrome, (H.dot(e_hat)) % 2):
            return e_hat, n_iter
        msg_v2c = np.where(H == 1, posterior - msg_c2v, 0.0)

    return e_hat, n_iter



# ---------------------------------------------------------------------
# Belief Propagation decoder
# ---------------------------------------------------------------------
def BP_decoder(H: np.ndarray, syndrome: np.ndarray, p: float, max_iter: int = 50, eps: float = 1e-9) -> np.ndarray:
    """
    BP decoder for a binary LDPC code defined by parity-check matrix H.
    Uses LLR-based sum-product updates.
    Args:
        H : (m, n) binary matrix
        syndrome : (m,) binary vector
        p : error probability
        max_iter : max number of iterations
    Returns:
        Estimated error vector (n,) in {0,1}.
    """

    m, n = H.shape
    # Indices of edges (flatten adjacency)
    check_idx, var_idx = np.where(H)
    E = len(check_idx)

    # For each variable and check, list of edges
    edges_for_check = [np.where(check_idx == i)[0] for i in range(m)]
    edges_for_var = [np.where(var_idx == j)[0] for j in range(n)]

    # Priors as LLR
    L0 = np.log((1 - p) / max(p, eps))

    # Messages per edge
    msg_v2c = np.full(E, L0, dtype=np.float64)
    msg_c2v = np.zeros(E, dtype=np.float64)

    # Mapping for vectorization
    check_edge_ptrs = np.zeros(m + 1, dtype=int)
    for i in range(m):
        check_edge_ptrs[i + 1] = check_edge_ptrs[i] + len(edges_for_check[i])
    var_edge_ptrs = np.zeros(n + 1, dtype=int)
    for j in range(n):
        var_edge_ptrs[j + 1] = var_edge_ptrs[j] + len(edges_for_var[j])

    for n_iter in range(max_iter):
        # ---- Check update (vectorized via per-check groups) ----
        for i in range(m):
            edges = edges_for_check[i]
            if len(edges) == 0:
                continue
            msgs = msg_v2c[edges]
            prod = np.prod(np.tanh(msgs / 2.0))
            for e in edges:
                val = 2 * np.arctanh(prod / np.tanh(msg_v2c[e] / 2.0))
                if syndrome[i]:
                    val = -val
                msg_c2v[e] = val

        # ---- Variable update ----
        for j in range(n):
            edges = edges_for_var[j]
            if len(edges) == 0:
                continue
            total = L0 + np.sum(msg_c2v[edges]) - msg_c2v[edges]
            msg_v2c[edges] = total

        # ---- Posterior beliefs ----
        L_post = np.zeros(n)
        for j in range(n):
            edges = edges_for_var[j]
            if len(edges):
                L_post[j] = L0 + np.sum(msg_c2v[edges])
            else:
                L_post[j] = L0

        est = (L_post < 0).astype(int)

        # Check syndrome satisfaction
        syn_est = (H @ est) % 2
        if np.all(syn_est == syndrome):
            return est, n_iter

    return est, n_iter

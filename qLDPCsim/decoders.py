"""
Copyright (c) 2025, Alberto G. Perotti
All rights reserved.

Decoders for quantum LDPC codes.
All these decoders take in input a syndrome and a parity-check matrix.
The decoders produce an estimated error vector.
Iterative decoders also return the number of iterations.

REFERENCES
[1] https://doi.org/10.48550/arXiv.2205.02341
[2] (Any paper with BP)
[3] Vasic et al., Collective Bit-Flipping...
"""

import numpy as np



# -----------------------------
# Simple naive greedy (NG) decoder
# -----------------------------
def NG_decoder(H: np.ndarray, syndrome: np.ndarray) -> np.ndarray:
    """
    A very simple greedy decoder: repeatedly pick a variable node connected to 
    the currently nonzero syndrome with highest degree, flip that variable in 
    the candidate error, update the syndrome, and repeat up to some limit.

    Args:
        H : parity-check matrix (m x n)
        syndrome : length-m binary array
    Returns:
        estimated error vector
        number of steps
    """
    m, n = H.shape
    residual = syndrome.copy()
    est = np.zeros(n, dtype=np.int8)

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

    return est, steps




# ---------------------------------------------------------------------
# Bit-Flipping (BF) decoder [3]
# ---------------------------------------------------------------------
def BF_decoder(H: np.ndarray, syndrome: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """
    Bit-flipping decoder.

    Args:
        H : parity-check matrix (m x n)
        syndrome : length-m binary array
        max_iter : max number of iuterations
    Returns:
        estimated error vector
        number of iterations
    """
    if H.size == 0 or syndrome.size == 0:
        return np.zeros(H.shape[1] if H.size else 0, dtype=np.int8)

    e_hat = np.zeros((H.shape[1]))      # Estimated error pattern
    r = syndrome                        # Residual syndrome
    nChecks = np.sum(H, axis=0)
    s_hat = syndrome

    for n_iter in range(max_iter):
        nuc = r @ H     # Number of unsatisfied checks
        e_hat = e_hat.astype(bool) ^ (nuc > nChecks/2.)
        s_hat = H @ e_hat
        r = s_hat.astype(bool) ^ syndrome
        if np.sum(r) == 0:
            return e_hat, (n_iter+1)

    return e_hat, max_iter




# ---------------------------------------------------------------------
# Min-Sum (MS) decoder [1]
# ---------------------------------------------------------------------
def MS_decoder(H: np.ndarray, syndrome: np.ndarray, p: float, max_iter: int = 99, \
               layers: list = None, beta: float = 0.75, eps: float = 1e-9) -> np.ndarray:
    """
    Normalized Min-Sum decoding for the binary LDPC code specified by the parity-check matrix H.
    The decoder operates according to a layered scheduling specified by the list of arrays
    in parameter layers, where each array contains teh indices of a subset of parity
    checks within a layer. Flooding is obtained by configuring just one layer contaiing all 
    checks. Serial scheduling is ontained by having all single-check layers.

    Args:
        H : parity-check matrix (m x n).
        syndrome : length-m binary array.
        p : prior error probability.
        max_iter : maximum number of iterations.
        layers : partition of checks. Each array has indices of checks in same layer.
        beta : normalization factor.
        eps : a factor to avoid division by zero.
    Returns:
        Estimated error vector.
        Number of iterations.
    """
    if H.size == 0 or syndrome.size == 0:
        return np.zeros(H.shape[1] if H.size else 0, dtype=np.int8)

    m, n = H.shape

    if layers is None:
        layers = [np.range(m)]

    # Initialize messages
    L_ch = np.log((1 - p) / max(p, eps))
    msg_v2c = np.zeros((m, n), dtype=np.float32)
    msg_v2c[H == 1] = L_ch
    msg_c2v = np.zeros_like(msg_v2c)
    syn_sign = np.where(syndrome[:, None] == 1, -1.0, 1.0)

    for n_iter in range(max_iter):
        for l in range(len(layers)):
            # Check node update
            abs_msg = np.abs(msg_v2c[layers[l],:])
            sign_msg = np.sign(msg_v2c[layers[l],:])
            sign_msg[sign_msg == 0] = 1.0
            prod_sign = np.prod(np.where(H[layers[l],:] == 1, sign_msg, 1.0), axis=1, keepdims=True)
            min_abs = np.min(np.where(H[layers[l],:] == 1, abs_msg, np.inf), axis=1, keepdims=True)
            minidx = np.argmin(np.where(H[layers[l],:] == 1, abs_msg, np.inf), axis=1)
            abs_msg2 = abs_msg
            abs_msg2[range(abs_msg.shape[0]), minidx] = np.inf
            min_abs2 = np.min(np.where(H[layers[l],:] == 1, abs_msg, np.inf), axis=1, keepdims=True)
            min_abs[np.isinf(min_abs)] = 0.0
            min_abs2[np.isinf(min_abs2)] = 0.0
            msg_c2v[layers[l],:] = np.where(np.logical_and(H[layers[l],:] == 1, np.abs(msg_v2c[layers[l],:]) != min_abs), beta * syn_sign[layers[l]] * prod_sign * min_abs / (sign_msg + (1 - H[layers[l],:])), np.inf)
            msg_c2v[layers[l],:] = np.where(np.logical_and(np.isinf(msg_c2v[layers[l],:]), np.abs(msg_v2c[layers[l],:]) == min_abs), beta * syn_sign[layers[l]] * prod_sign * min_abs2 / (sign_msg + (1 - H[layers[l],:])), msg_c2v[layers[l],:])
            msg_c2v[np.isinf(msg_c2v)] = 0.0
    
            # Variable node update
            VNsum = np.sum(msg_c2v, axis=0)
            posterior = L_ch + VNsum
            e_hat = (posterior < 0).astype(np.int8)
            if np.array_equal(syndrome, (H.dot(e_hat)) % 2):
                return e_hat, (n_iter+1)
            msg_v2c = np.where(H == 1, posterior - msg_c2v, 0.0)

    return e_hat, max_iter



# ---------------------------------------------------------------------
# Belief Propagation (BP) decoder
# ---------------------------------------------------------------------
def BP_decoder(H: np.ndarray, syndrome: np.ndarray, p: float, max_iter: int = 99, \
               layers: list = None, eps: float = 1e-9) -> np.ndarray:
    """
    BP decoding for the binary LDPC code specified by the parity-check matrix H.
    The decoder operates according to a layered scheduling specified by the list of arrays
    in parameter layers, where each array contains teh indices of a subset of parity
    checks within a layer. Flooding is obtained by configuring just one layer contaiing all 
    checks. Serial scheduling is ontained by having all single-check layers.
    Args:
        H : (m, n) binary matrix
        syndrome : (m,) binary vector
        p : error probability
        max_iter : max number of iterations
        layers : partition of checks. Each array has indices of checks in same layer.
        eps : a factor to avoid division by zero.
    Returns:
        Estimated error vector.
        Number of iterations.
    """

    if H.size == 0 or syndrome.size == 0:
        return np.zeros(H.shape[1] if H.size else 0, dtype=np.int8)

    m, n = H.shape

    if layers is None:
        layers = [np.range(m)]

    # Indices of edges
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
        for l in range(len(layers)):
            # ---- Check update (vectorized via per-check groups) ----
            for i in layers[l]:
                edges = edges_for_check[i]
                if len(edges) == 0:
                    continue
                msgs = msg_v2c[edges]
                prod = np.prod(np.tanh(msgs / 2.0))
                for e in edges:
                    th2 = prod / np.tanh(msg_v2c[e] / 2.0)
                    if np.abs(th2) == 1:
                        th2 = th2 - eps * np.sign(th2)
                    val = 2 * np.arctanh(th2)
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
    
            e_hat = (L_post < 0).astype(int)
    
            # Check syndrome satisfaction
            syn_est = (H @ e_hat) % 2
            if np.all(syn_est == syndrome):
                return e_hat, n_iter+1

    return e_hat, max_iter




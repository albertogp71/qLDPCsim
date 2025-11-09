# simulate.py

"""
Performance evaluation utilities 

"""

import numpy as np
from typing import Tuple
import logical_ops_from_checks
import decoders


# -----------------------------
# Stim circuit construction
# -----------------------------
def build_stim_circuit(Hx: np.ndarray, Hz: np.ndarray, p: float) -> Tuple[stim.Circuit, int, int, int]:
    """
    Construct a stim.Circuit that:
      - has n data qubits (indices 0..n-1)
      - has ancilla qubits for Hz (Z-checks) and Hx (X-checks)
      - applies DEPOLARIZE1(p) to each data qubit
      - measures each stabilizer once, producing one measurement bit per stabilizer

    Returns (circuit, n_data, n_meas, ancilla_offset)
      - n_data: number of data qubits
      - n_meas: total number of measurement bits (rows(Hz) + rows(Hx))
      - ancilla_offset: first ancilla qubit index (useful for debugging / extension)
    """
    if Hx is None:
        Hx = np.zeros((0, 0), dtype=int)
    if Hz is None:
        Hz = np.zeros((0, 0), dtype=int)

    m_x, n_x = Hx.shape if Hx.size else (0, 0)
    m_z, n_z = Hz.shape if Hz.size else (0, 0)
    # require same number of data qubits
    if (n_x != 0 and n_z != 0) and (n_x != n_z):
        raise ValueError("Hx and Hz must have the same number of columns (data qubits).")
    n = n_x if n_x != 0 else n_z

    # Build a circuit as a list of lines -> feed to stim.Circuit(...)
    lines = []
    # allocate qubits implicitly by referencing indices in operations. Stim
    # will automatically size the circuit qubit register to accommodate the largest index used.
    # We'll use data qubits 0..n-1, then ancillas n..n+m_z+m_x-1
    ancilla_start = n
    ancilla_for_Z = list(range(ancilla_start, ancilla_start + m_z))
    ancilla_for_X = list(range(ancilla_start + m_z, ancilla_start + m_z + m_x))

    # 1) Apply depolarizing channel to each data qubit
    #    Use the operation text form: DEPOLARIZE1(p) q
    for q in range(n):
        lines.append(f"DEPOLARIZE1({p}) {q}")

    # 2) For each Z-check (row of Hz): measure product of Zs using an ancilla
    #    Circuit pattern:
    #      # ancilla starts in |0>
    #      CNOT data_q ancilla    for each q in support
    #      M ancilla
    #
    #    We'll append measurements in the same order as Hz rows.
    for row_idx in range(m_z):
        anc = ancilla_for_Z[row_idx]
        # apply CNOT from each data qubit in the stabilizer to ancilla
        cols = np.where(Hz[row_idx] % 2 == 1)[0]
        for q in cols:
            lines.append(f"CNOT {q} {anc}")
        # measure ancilla in Z
        lines.append(f"M {anc}")

    # 3) For each X-check (row of Hx): measure product of Xs by rotating data qubits with H
    #    Pattern:
    #      H q        for q in support
    #      CNOT q anc  for each q in support
    #      H q        for q in support    # undo
    #      M anc
    for row_idx in range(m_x):
        anc = ancilla_for_X[row_idx]
        cols = np.where(Hx[row_idx] % 2 == 1)[0]
        for q in cols:
            lines.append(f"H {q}")
        for q in cols:
            lines.append(f"CNOT {q} {anc}")
        for q in cols:
            lines.append(f"H {q}")
        lines.append(f"M {anc}")

    # Build circuit
    circ_text = "\n".join(lines)
    circ = stim.Circuit(circ_text)
    total_meas = m_z + m_x
    return circ, n, total_meas, ancilla_start



# -----------------------------
# Main simulation function
# -----------------------------
def simulate(Hx: np.ndarray,
             Hz: np.ndarray,
             p: float,
             shots: int = 1000,
             rng_seed: Optional[int] = None) -> dict:
    """
    Build the stim circuit, run shots, and (optionally) do naive decoding to estimate
    logical error rate.

    Returns a dictionary with:
      - 'shots' : number of measurement shots
      - 'avg_syndrome_weight' : average number of 1 bits in stabilizer measurement
      - 'syndrome_counts' : histogram dict mapping syndrome bitstrings to counts (only if shots small)
      - 'logical_error_rate' : estimated if logical ops provided (else None)
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    circ, n_data, n_meas, anc_start = build_stim_circuit(Hx, Hz, p)

    logical_X_ops, logical_Z_ops = logical_ops_from_checks.logical_ops_from_css(Hx, Hz)

    # compile sampler and sample many shots
    sampler = circ.compile_sampler()
    # sample returns dtype uint8 array of shape (shots, n_meas) with measurement bits in the order we appended
    samples = sampler.sample(shots=shots)

    # compute average syndrome weight
    weights = samples.sum(axis=1)
    avg_weight = float(weights.mean())

    logical_error_rate = None
    if (logical_X_ops is not None) or (logical_Z_ops is not None):
        # attempt simple decoding: for each shot, try to produce estimated errors for X and Z
        # From CSS: Hz (Z checks) detect X errors; Hx (X checks) detect Z errors.
        # We'll use naive_greedy_decoder on each set independently.
        m_z = Hz.shape[0] if Hz.size else 0
        m_x = Hx.shape[0] if Hx.size else 0

        failures = 0
        for shot_idx in range(shots):
            row = samples[shot_idx]
            # row structure: [Hz measurements...] followed by [Hx measurements...]
            sy_z = row[:m_z].astype(int) if m_z else np.array([], dtype=int)
            sy_x = row[m_z:m_z+m_x].astype(int) if m_x else np.array([], dtype=int)

            # decode X-errors from sy_z using Hz (Hz * eX = sy_z)
            eX_hat = naive_greedy_decoder(Hz if Hz.size else np.zeros((0, n_data), dtype=int), sy_z)
            # decode Z-errors from sy_x using Hx
            eZ_hat = naive_greedy_decoder(Hx if Hx.size else np.zeros((0, n_data), dtype=int), sy_x)

            # residuals are simply e_hat (since circuit applies error then we decode based on syndrome only)
            # Determine whether residuals anticommute with logical operators
            logical_failure = False
            if logical_X_ops is not None:
                # logical X ops detect Z residuals: dot(lx, eZ_hat) mod 2 != 0 indicates flip
                for lx in logical_X_ops:
                    if int(np.dot(lx % 2, eZ_hat % 2) % 2) == 1:
                        logical_failure = True
                        break
            if (not logical_failure) and (logical_Z_ops is not None):
                # logical Z ops detect X residuals
                for lz in logical_Z_ops:
                    if int(np.dot(lz % 2, eX_hat % 2) % 2) == 1:
                        logical_failure = True
                        break
            if logical_failure:
                failures += 1

        logical_error_rate = failures / shots

    return {
        "shots": shots,
        "avg_syndrome_weight": avg_weight,
        "syndrome_counts": syndrome_counts,
        "logical_error_rate": logical_error_rate
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stim-based QC-LDPC depolarizing-channel simulator.")
    parser.add_argument("--Hx", required=True, help="Path to Hx parity-check matrix (.npy or whitespace text).")
    parser.add_argument("--Hz", required=True, help="Path to Hz parity-check matrix (.npy or whitespace text).")
    parser.add_argument("--p", type=float, required=True, help="Depolarizing probability per qubit (0..1).")
    parser.add_argument("--shots", type=int, default=1000, help="Number of Monte Carlo shots.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed.")
    args = parser.parse_args(argv)

    Hx = load_matrix(args.Hx)
    Hz = load_matrix(args.Hz)

    res = simulate(Hx, Hz, p=args.p, shots=args.shots, rng_seed=args.seed)

    print(f"shots: {res['shots']}")
    print(f"avg_syndrome_weight: {res['avg_syndrome_weight']:.4f}")
    if res['syndrome_counts'] is not None:
        print(f"unique syndrome patterns: {len(res['syndrome_counts'])}")
    if res['logical_error_rate'] is not None:
        print(f"estimated logical error rate: {res['logical_error_rate']:.6e}")
    else:
        print("logical operators not provided; only syndrome stats computed.")


if __name__ == "__main__":
    main()

# simulate.py

"""
Performance evaluation utilities 

"""

import argparse
import numpy as np
from typing import Optional, List, Tuple
# from qLDPCsim import logical_ops_from_checks
from qLDPCsim import logical_ops_css
from qLDPCsim import decoders
import stim



# -----------------------------
# I/O helper functions
# -----------------------------
def load_matrix(path: str) -> np.ndarray:
    """Load a binary matrix from .npy or whitespace text."""
    if path.endswith(".npy"):
        mat = np.load(path)
    else:
        # attempt to parse whitespace-separated 0/1 text
        mat = []
        with open(path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = [int(x) for x in line.split()]
                mat.append(row)
        mat = np.array(mat, dtype=int)
    return (mat % 2).astype(np.int8)




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
        raise ValueError("Hx and Hz must have the same number of columns (physical qubits).")
    n = n_x if n_x != 0 else n_z

    stabs = []
    for r in range(m_z):
        stZ = stim.PauliString.from_numpy(xs=np.zeros(Hz.shape[1]).astype(bool), zs=Hz[r,:].astype(bool))
        stabs.append(stZ)
    for r in range(m_x):
        stX = stim.PauliString.from_numpy(xs=Hx[r,:].astype(bool), zs=np.zeros(Hx.shape[1]).astype(bool))
        stabs.append(stX)
    sTab = stim.Tableau.from_stabilizers(stabs, allow_underconstrained=True, allow_redundant=True)
    circEnc = sTab.to_circuit()
    # print(circEnc.diagram())



    # Build a circuit as a list of lines -> feed to stim.Circuit(...)
    lines = []
    # allocate qubits implicitly by referencing indices in operations. Stim
    # will automatically size the circuit qubit register to accommodate the largest index used.
    # We'll use data qubits 0..n-1, then ancillas n..n+m_z+m_x-1
    ancilla_start = n
    ancilla_Z = list(range(ancilla_start, ancilla_start + m_z))
    ancilla_X = list(range(ancilla_start + m_z, ancilla_start + m_z + m_x))

    # 0) Reset all qubit
    for q in range(n,n + m_z):
        lines.append(f"R {q}")
    for q in range(n + m_z, n + m_z + m_x):
        lines.append(f"R {q}")
    lines.append('TICK')

    # 1) Apply depolarizing channel to each data qubit
    for q in range(n):
        lines.append(f"DEPOLARIZE1({p}) {q}")
    lines.append('TICK')

    # 2) For each Z-check (row of Hz): measure product of Zs using an ancilla
    for row_idx in range(m_z):
        anc = ancilla_Z[row_idx]
        # apply CNOT to each physical qubit in the stabilizer to ancilla
        cols = np.where(Hz[row_idx] % 2 == 1)[0]
        for q in cols:
            lines.append(f"CNOT {q} {anc}")
        # measure ancilla in Z
    lines.append('TICK')
    for row_idx in range(m_z):
        anc = ancilla_Z[row_idx]
        lines.append(f"M {anc}")
    lines.append('TICK')

    # 3) For each X-check (row of Hx): measure product of Xs by rotating physical qubits with H
    for row_idx in range(m_x):
        anc = ancilla_X[row_idx]
        cols = np.where(Hx[row_idx] % 2 == 1)[0]
        for q in cols:
            lines.append(f"H {q}")
            lines.append(f"CNOT {q} {anc}")
            lines.append(f"H {q}")
    lines.append('TICK')
    for row_idx in range(m_x):
        anc = ancilla_X[row_idx]
        lines.append(f"M {anc}")
    lines.append('TICK')

    for q in range(-m_x-m_z,0):
        lines.append(f"DETECTOR rec[{q}]")


    # Build circuit
    circ_text = "\n".join(lines)
    circ = circEnc + stim.Circuit(circ_text)
    total_meas = m_z + m_x
    return circ, n, total_meas, ancilla_start



# -----------------------------
# Main simulation function
# -----------------------------
def simulate(Hx: np.ndarray,
             Hz: np.ndarray,
             p: float,
             shots: int = 1000,
             decType: str = 'MS',
             decIterations: int = 50,
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



    print('Building stim circuit...', end='', flush=True)
    circ, n_data, n_meas, anc_start = build_stim_circuit(Hx, Hz, p)
    # print(circ.diagram())
    print('done.')

    
    # dem = circ.detector_error_model()
    # print(repr(dem))

    print('Sampling stim circuit...', end='', flush=True)
    # compile sampler and sample many shots
    sampler = circ.compile_sampler()
    # sampDet = circ.compile_detector_sampler()
    # sample returns dtype uint8 array of shape (shots, n_meas) with measurement bits in the order we appended
    samples = sampler.sample(shots=shots)
    # samDets = sampDet.sample(shots=shots)
    print('done.')

    # compute average syndrome weight
    weights = samples.sum(axis=1)
    avg_weight = float(weights.mean())

    # syndromes histogram if small
    syndrome_counts = None
    if False: # shots <= 2000:
        # convert rows to bitstring keys
        keys, counts = np.unique(samples.astype(np.uint8).view([('b', 'u1') * samples.shape[1]]), return_counts=True)
        # simpler: use string keys
        sc = {}
        for row in samples:
            s = "".join(str(int(b)) for b in row.tolist())
            sc[s] = sc.get(s, 0) + 1
        syndrome_counts = sc

    logical_error_rate = None
    # logical_X_ops, logical_Z_ops = logical_ops_from_checks.logical_ops_from_css(Hx, Hz)
    logical_X_ops, logical_Z_ops, nlq = logical_ops_css.logical_ops_css(Hx, Hz)
    breakpoint()
    if (logical_X_ops is not None) or (logical_Z_ops is not None):
        m_z = Hz.shape[0] if Hz.size else 0
        m_x = Hx.shape[0] if Hx.size else 0

        failures = 0
        decFailures = 0
        nIterAcc = 0
        for shot_idx in range(shots):
            print(F'\rDecoding block n. {shot_idx+1:3}/{shots:4}...', end='', flush=True)
            row = samples[shot_idx]
            # row structure: [Hz measurements...] followed by [Hx measurements...]
            sy_z = row[:m_z].astype(int) if m_z else np.array([], dtype=int)
            sy_x = row[m_z:m_z+m_x].astype(int) if m_x else np.array([], dtype=int)

            # print(f"\nsy_z is ", end='')
            # print("".join("_1"[e] for e in sy_z)) #, end = ' ')
            # # print("".join("_1"[e] for e in samDets[shot_idx,:m_z].astype(int)))
            # print(f"sy_x is ", end='')
            # print("".join("_1"[e] for e in sy_x)) #, end = ' ')
            # # print("".join("_1"[e] for e in samDets[shot_idx,m_z:].astype(int)))

            match decType:
                case "NG":
                    eX_hat, nIterX = decoders.naive_greedy_decoder(Hz if Hz.size else np.zeros((0, n_data), dtype=int), sy_z)
                    eZ_hat, nIterZ = decoders.naive_greedy_decoder(Hx if Hx.size else np.zeros((0, n_data), dtype=int), sy_x)
                    nIterAcc += nIterX + nIterZ
                case "BF":
                    eX_hat, nIterX = decoders.BF_decoder(Hz if Hz.size else np.zeros((0, n_data), dtype=int), sy_z)
                    eZ_hat, nIterZ = decoders.BF_decoder(Hx if Hx.size else np.zeros((0, n_data), dtype=int), sy_x)
                    nIterAcc += nIterX + nIterZ
                case "MS":
                    eX_hat, nIterX = decoders.min_sum_decoder(Hz, sy_z, p=p/3, max_iter=decIterations)
                    eZ_hat, nIterZ = decoders.min_sum_decoder(Hx, sy_x, p=p/3, max_iter=decIterations)
                    nIterAcc += nIterX + nIterZ
                case "BP":
                    eX_hat, nIterX = decoders.BP_decoder(Hz, sy_z, p=p/3, max_iter=decIterations)
                    eZ_hat, nIterZ = decoders.BP_decoder(Hx, sy_x, p=p/3, max_iter=decIterations)
                    nIterAcc += nIterX + nIterZ
                case _:
                    raise ValueError("Unrecognized decoder type.")


            # Residuals are simply e_hat (since circuit applies error then we decode based on syndrome only)
            # Determine whether residuals anticommute with logical operators
            logical_failure = False
            for lx in logical_X_ops:
                if int(np.dot(lx % 2, eZ_hat % 2) % 2) == 1:
                    logical_failure = True
                    break
            for lz in logical_Z_ops:
                if int(np.dot(lz % 2, eX_hat % 2) % 2) == 1:
                    logical_failure = True
                    break
            if logical_failure:
                failures += 1
            decoder_failure = False
            if not np.array_equal(sy_z, (Hz.dot(eX_hat)) % 2):
                decoder_failure = True
            if not np.array_equal(sy_x, (Hx.dot(eZ_hat)) % 2):
                decoder_failure = True
            if decoder_failure:
                decFailures +=1
            print(f'N. iterations: {nIterX:2},{nIterZ:2}, Dec. failure rate: {decFailures/(shot_idx+1):.3f}')
            
        print()
        logical_error_rate = failures / float(shots)

    return {
        "shots": shots,
        "avg_syndrome_weight": avg_weight,
        "syndrome_counts": syndrome_counts,
        "logical_error_rate": logical_error_rate,
        "Decoder_failure_rate": decFailures/float(shots),
        "Avg_number_of_iterations": nIterAcc/float(shots)
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stim-based QC-LDPC depolarizing-channel simulator.")
    parser.add_argument("--Hx", required=True, help="Path to Hx parity-check matrix (.npy or whitespace text).")
    parser.add_argument("--Hz", required=True, help="Path to Hz parity-check matrix (.npy or whitespace text).")
    parser.add_argument("--p", type=float, required=True, help="Depolarizing probability per qubit (0..1).")
    parser.add_argument("--shots", type=int, default=1000, help="Number of Monte Carlo shots.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed.")
    parser.add_argument("--dectype", choices=['NG', 'BF', 'MS', 'BP'], default='MS', help="Decoder type: [NG] Naive Greedy; [MS] Min-Sum; [BP] Belief Propagation.")
    parser.add_argument("--deciterations", type=int, default=50, help="Number of decoding iterations.")
    args = parser.parse_args(argv)

    Hx = load_matrix(args.Hx)
    Hz = load_matrix(args.Hz)

    res = simulate(Hx, Hz, p=args.p, shots=args.shots, rng_seed=args.seed, decType=args.dectype, decIterations=args.deciterations)

    # print(f"shots: {res['shots']}")
    # print(f"avg_syndrome_weight: {res['avg_syndrome_weight']:.4f}")
    if res['syndrome_counts'] is not None:
        print(f"unique syndrome patterns: {len(res['syndrome_counts'])}")
    if res['logical_error_rate'] is not None:
        print(f"Logical error rate: {res['logical_error_rate']:.6e}")
        print(f"Decoder failure rate: {res['Decoder_failure_rate']:.2e}")
    else:
        print("logical operators not provided; only syndrome stats computed.")
    if res['Avg_number_of_iterations'] > 0:
        print(f"Avg number of iterations: {res['Avg_number_of_iterations']:.2f}")




if __name__ == "__main__":
    main()

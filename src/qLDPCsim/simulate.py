# simulate.py

"""
Performance evaluation utilities 

"""

import argparse
import numpy as np
from typing import Optional, List, Tuple
from qLDPCsim import logical_ops_from_checks
# from qLDPCsim import logical_ops_css
from qLDPCsim import decoders, gf2math
import stim
from qLDPCsim import stimEncoder


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

    stabilizers = []
    
    
    # HxRB = gf2math.rowBasis(Hx)
    # HzRB = gf2math.rowBasis(Hz)
    # Gx = gf2math.nullSpaceBasis(HzRB)
    # Gz = gf2math.nullSpaceBasis(HxRB)
    # Gx1 = gf2math.nullSpace(HzRB)
    # Gz1 = gf2math.nullSpace(HxRB)
    # Gx2 = gf2math.nullSpace(HzRB)
    # Gz2 = gf2math.nullSpace(HxRB)
    # Gx = gf2math.nullSpace(Hx)
    # Gz = gf2math.nullSpace(Hz)
    

    # Kx = gf2math.rowBasisMinChange(Hx) #nullspace(gf2math.nullspace(Hx))
    # Kz = gf2math.rowBasisMinChange(Hz) #nullspace(gf2math.nullspace(Hz))
    # for r in range(Kz.shape[0]):
    #     stZ = stim.PauliString.from_numpy(xs=np.zeros(n).astype(bool), zs=Kz[r,:].astype(bool))
    #     stabilizers.append(stZ)
    # for r in range(Kx.shape[0]):
    #     stX = stim.PauliString.from_numpy(xs=Kx[r,:].astype(bool), zs=np.zeros(n).astype(bool))
    #     stabilizers.append(stX)
    # sTab = stim.Tableau.from_stabilizers(stabilizers, allow_underconstrained=True, allow_redundant=False)
    # circEnc = stim.Circuit("R" + "".join(f" {q}" for q in range(n))) \
    #     + stim.Circuit("X_ERROR(0.5)" + "".join(f" {q}" for q in range(n - Kx.shape[0] - Kz.shape[0]))) \
    #     + sTab.to_circuit()
    # # print('\n')
    # # print(circEnc.diagram())
    # # breakpoint()

    # for row in Hx:
    #     z_str = "".join("Z" if b else "I" for b in row)
    #     stabilizers.append(stim.PauliString(z_str))
    # for row in Hz:
    #     x_str = "".join("X" if b else "I" for b in row)
    #     stabilizers.append(stim.PauliString(x_str))
    # T = stim.Tableau.from_stabilizers(stabilizers, allow_underconstrained=True, allow_redundant=True)
    # circEnc = stim.Circuit("R" + "".join(f" {q}" for q in range(n))) \
    #     + stim.Circuit("TICK\nX_ERROR(0.5)" + "".join(f" {q}" for q in range(n - HxRB.shape[0] - HzRB.shape[0]))) \
    #     + stim.Circuit("TICK") \
    #     + stimEncoder.css_ldpc_encoder(HxRB, HzRB)


    Hx = Hx.astype(np.bool_)
    Hz = Hz.astype(np.bool_)
    assert Hx.shape[1] == Hz.shape[1]
    r, n = Hx.shape
    stabX = [stim.PauliString.from_numpy(xs = Hx[row,:], zs = np.zeros_like(Hx[row,:], \
                                        dtype=np.bool_)) for row in range(Hx.shape[0])]
    stabZ = [stim.PauliString.from_numpy(xs = np.zeros_like(Hz[row,:], dtype=np.bool_), \
                                         zs = Hz[row,:]) for row in range(Hz.shape[0])]
    encT = stim.Tableau.from_stabilizers(stabX + stabZ, allow_underconstrained=True, allow_redundant=True)
    encC = encT.to_circuit(method='elimination')

    # breakpoint()

     # Logical generators
    # Gz = gf2math.nullSpace(Hx)   # X-type logicals (H + CNOT encoding)
    # Gx = gf2math.nullSpace(Hz)   # Z-type logicals (CNOT-only encoding)
    k = n - Hx.shape[0] - Hz.shape[0]
    # Gz = logical_ops_from_checks.remove_dependents(Gz, Hz)
    circEnc  = stim.Circuit('R' + ''.join(f' {q}' for q in range(n)))
    circEnc += stim.Circuit('X_ERROR(0.5)' + ''.join(f' {q}' for q in range(n-1,n-k-1,-1)))
    # for row in Gz:
    #     support = np.where(row == 1)[0]
    #     if len(support) == 0:
    #         continue
    #     root = support[0]
    #     circEnc += stim.Circuit('CNOT' + ''.join(f' {q} {root}' for q in support[1:]))
    # for row in Gx:
    #     support = np.where(row == 1)[0]
    #     if len(support) == 0:
    #         continue
    #     root = support[0]
    #     circEnc += stim.Circuit(f'H {root}')
    #     circEnc += stim.Circuit('CNOT' + ''.join(f' {root} {q}' for q in support[1:]))
    circEnc += encC


    # breakpoint()

    # Build a circuit as a list of lines -> feed to stim.Circuit(...)
    # allocate qubits implicitly by referencing indices in operations. Stim
    # will automatically size the circuit qubit register to accommodate the largest index used.
    # We'll use data qubits 0..n-1, then ancillas n..n+m_z+m_x-1
    ancilla_start = n
    ancilla_Z = list(range(ancilla_start, ancilla_start + m_z))
    ancilla_X = list(range(ancilla_start + m_z, ancilla_start + m_z + m_x))


    # 1. Apply depolarizing channel to each data qubit, and reset syndrome generator ancilla qubits
    lines = []
    lines.append(f"DEPOLARIZE1({p})" + "".join(f" {q}" for q in range(n)))
    lines.append("R" + "".join(f" {q}" for q in range(n, n + m_z + m_x)))
    circChannel = stim.Circuit("\n".join(lines))

    # 2. For each Z-check: CNOT of physical qbit (controlling) and ancilla (controlled)
    lines = []
    for row_idx in range(m_z):
        anc = ancilla_Z[row_idx]
        cols = np.where(Hz[row_idx] % 2 == 1)[0]
        lines.append("CNOT" + "".join(f" {q} {anc}" for q in cols))
        
    # 3. For each X-check: H on physical qbit + CNOT of physical qbit (controlling) and ancilla (controlled) + H
    lines.append('TICK')
    lines.append('H' + ''.join(f' {ancilla_X[row_idx]}' for row_idx in range(m_x)))
    lines.append('TICK')
    for row_idx in range(m_x):
        anc = ancilla_X[row_idx]
        cols = np.where(Hx[row_idx] % 2 == 1)[0]
        for q in cols:
            lines.append(f"CNOT {anc} {q}")
    lines.append('TICK')
    lines.append('H' + ''.join(f' {ancilla_X[row_idx]}' for row_idx in range(m_x)))
    lines.append('TICK')
    
    # 4. Measure ancillas
    lines.append("M" + "".join(f" {ancilla_Z[row_idx]}" for row_idx in range(m_z)))
    lines.append("M" + "".join(f" {ancilla_X[row_idx]}" for row_idx in range(m_x)))
    
    # lines.append('TICK')

    # for q in range(-m_x-m_z,0):
    #     lines.append(f"DETECTOR rec[{q}]")


    # Build circuit
    circ = circEnc + stim.Circuit('TICK') + circChannel + stim.Circuit('TICK') + stim.Circuit("\n".join(lines))
    total_meas = m_z + m_x
    # breakpoint()
    return circ, n, total_meas, ancilla_start



# -----------------------------
# Main simulation function
# -----------------------------
def simulate(Hx: np.ndarray,
             Hz: np.ndarray,
             p: float,
             shots: int = 1000,
             decType: str = 'MS',
             decIterations: int = 99,
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

    print('Preparation: building Stim circuit...', end='', flush=True)
    circ, n_data, n_meas, anc_start = build_stim_circuit(Hx, Hz, p)
    # print(circ.diagram())

    print('sampling circuit...', end='', flush=True)
    sampler = circ.compile_sampler()
    samples = sampler.sample(shots=shots)

    print('generating logical operators...', end='', flush=True)
    logical_error_rate = None
    logical_X_ops, logical_Z_ops = logical_ops_from_checks.logical_ops_from_css(Hx, Hz)
    print('done.')



    m_z = Hz.shape[0] if Hz.size else 0
    m_x = Hx.shape[0] if Hx.size else 0

    corrFailures = 0
    decFailures = 0
    nIterAccX = 0
    nIterAccZ = 0
    for shot_idx in range(shots):
        print(F'\rDecoding block n. {shot_idx+1:3}/{shots:4}...', end='', flush=True)
        row = samples[shot_idx]
        # row structure: [Hz measurements...] followed by [Hx measurements...]
        sy_z = row[:m_z].astype(int) if m_z else np.array([], dtype=int)
        sy_x = row[m_z:m_z+m_x].astype(int) if m_x else np.array([], dtype=int)
        # breakpoint()
        print(f"\nsyn_z is ", end='')
        print("".join("_1"[e] for e in sy_z)) #, end = ' ')
        # print("".join("_1"[e] for e in samDets[shot_idx,:m_z].astype(int)))
        print(f"syn_x is ", end='')
        print("".join("_1"[e] for e in sy_x)) #, end = ' ')
        # print("".join("_1"[e] for e in samDets[shot_idx,m_z:].astype(int)))

        match decType:
            case "NG":
                eX_hat, nIterX = decoders.naive_greedy_decoder(Hz if Hz.size else np.zeros((0, n_data), dtype=int), sy_z)
                eZ_hat, nIterZ = decoders.naive_greedy_decoder(Hx if Hx.size else np.zeros((0, n_data), dtype=int), sy_x)
            case "BF":
                eX_hat, nIterX = decoders.BF_decoder(Hz if Hz.size else np.zeros((0, n_data), dtype=int), sy_z)
                eZ_hat, nIterZ = decoders.BF_decoder(Hx if Hx.size else np.zeros((0, n_data), dtype=int), sy_x)
            case "MS":
                eX_hat, nIterX = decoders.min_sum_decoder(Hz, sy_z, p=p/3, max_iter=decIterations)
                eZ_hat, nIterZ = decoders.min_sum_decoder(Hx, sy_x, p=p/3, max_iter=decIterations)
            case "BP":
                eX_hat, nIterX = decoders.BP_decoder(Hz, sy_z, p=p/3, max_iter=decIterations)
                eZ_hat, nIterZ = decoders.BP_decoder(Hx, sy_x, p=p/3, max_iter=decIterations)
            case _:
                raise ValueError("Unrecognized decoder type.")
                
        nIterAccX += nIterX
        nIterAccZ += nIterZ


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
            corrFailures += 1
            
        decoder_failure = False
        if not np.array_equal(sy_z, (Hz.dot(eX_hat)) % 2):
            decoder_failure = True
        if not np.array_equal(sy_x, (Hx.dot(eZ_hat)) % 2):
            decoder_failure = True
        if decoder_failure:
            decFailures +=1
        print(f'N. iterations (X, Z): {nIterX:2},{nIterZ:2}, Dec. failure rate: {decFailures/(shot_idx+1):.2e}', end='', flush=True)
        
    print()

    return {
        "Logical_error_rate": corrFailures / float(shots),
        "Decoder_failure_rate": decFailures/float(shots),
        "Avg_number_of_iterations_X": nIterAccX/float(shots),
        "Avg_number_of_iterations_Z": nIterAccZ/float(shots)
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stim-based QC-LDPC depolarizing-channel simulator.")
    parser.add_argument("--Hx", required=True, help="Path to Hx parity-check matrix (.npy or whitespace text).")
    parser.add_argument("--Hz", required=True, help="Path to Hz parity-check matrix (.npy or whitespace text).")
    parser.add_argument("--p", type=float, required=True, help="Depolarizing probability per qubit (0..1).")
    parser.add_argument("--shots", type=int, default=1000, help="Number of Monte Carlo shots.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed.")
    parser.add_argument("--dectype", choices=['NG', 'BF', 'MS', 'BP'], default='MS', help="Decoder type: [NG] Naive Greedy; [MS] Min-Sum; [BP] Belief Propagation.")
    parser.add_argument("--deciterations", type=int, default=99, help="Number of decoding iterations.")
    args = parser.parse_args(argv)

    Hx = load_matrix(args.Hx)
    Hz = load_matrix(args.Hz)
    
    res = simulate(Hx, Hz, p=args.p, shots=args.shots, rng_seed=args.seed, decType=args.dectype, decIterations=args.deciterations)

    print(f"Logical error rate: {res['Logical_error_rate']:.2e}")
    print(f"Decoder failure rate: {res['Decoder_failure_rate']:.2e}")
    print(f"Avg number of iterations (X,Z): ({res['Avg_number_of_iterations_X']:.2f},{res['Avg_number_of_iterations_Z']:.2f})")




if __name__ == "__main__":
    main()

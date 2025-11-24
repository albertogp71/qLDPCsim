"""
Copyright (c) 2025, Alberto G. Perotti
All rights reserved.

User callable function .

"""

import argparse
import numpy as np
from qLDPCsim import decoders, gf2math
import stim
from typing import Tuple, Optional



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
    # Check that same number of physical qubits
    if (n_x != 0 and n_z != 0) and (n_x != n_z):
        raise ValueError("Hx and Hz must have the same number of columns (physical qubits).")
    n = n_x if n_x != 0 else n_z

    
    Hx = Hx.astype(np.bool_)
    Hz = Hz.astype(np.bool_)
    assert Hx.shape[1] == Hz.shape[1]

    r, n = Hx.shape
    k = n - gf2math.rank(Hx) - gf2math.rank(Hz)     # Number of logical qbits

    stabX = [stim.PauliString.from_numpy(xs = Hx[row,:], zs = np.zeros_like(Hx[row,:], \
                                        dtype=np.bool_)) for row in range(Hx.shape[0])]
    stabZ = [stim.PauliString.from_numpy(xs = np.zeros_like(Hz[row,:], dtype=np.bool_), \
                                         zs = Hz[row,:]) for row in range(Hz.shape[0])]
    encT = stim.Tableau.from_stabilizers(stabX + stabZ, allow_underconstrained=True, allow_redundant=True)
    encC = encT.to_circuit(method='elimination')

    circEnc  = stim.Circuit('R' + ''.join(f' {q}' for q in range(n)))
    circEnc += stim.Circuit('DEPOLARIZE1(0.75)' + ''.join(f' {q}' for q in range(n-1,n-k-1,-1)))
    circEnc += encC


    # Physical qbits 0..(n-1), then ancillas n..(n+m_z+m_x-1)
    ancilla_start = n
    ancilla_Z = list(range(ancilla_start, ancilla_start + m_z))
    ancilla_X = list(range(ancilla_start + m_z, ancilla_start + m_z + m_x))


    # Apply depolarizing channel to each physical qbit, and reset syndrome generator ancilla qubits
    lines = []
    lines.append(f'DEPOLARIZE1({p})' + ''.join(f' {q}' for q in range(n)))
    lines.append('R' + ''.join(f' {q}' for q in range(n, n + m_z + m_x)))
    circChannel = stim.Circuit("\n".join(lines))

    # For each Z-check: CNOT of physical qbit (controlling) and ancilla (controlled)
    lines = []
    for row_idx in range(m_z):
        anc = ancilla_Z[row_idx]
        cols = np.where(Hz[row_idx] % 2 == 1)[0]
        lines.append('CNOT' + ''.join(f' {q} {anc}' for q in cols))
        
    # For each X-check: H on physical qbit + CNOT of physical qbit (controlling) and ancilla (controlled) + H
    lines.append('TICK')
    lines.append('H' + ''.join(f' {ancilla_X[row_idx]}' for row_idx in range(m_x)))
    lines.append('TICK')
    for row_idx in range(m_x):
        anc = ancilla_X[row_idx]
        cols = np.where(Hx[row_idx] % 2 == 1)[0]
        for q in cols:
            lines.append(f'CNOT {anc} {q}')
    lines.append('TICK')
    lines.append('H' + ''.join(f' {ancilla_X[row_idx]}' for row_idx in range(m_x)))
    lines.append('TICK')
    
    # Measure ancillas
    lines.append('M' + ''.join(f' {ancilla_Z[row_idx]}' for row_idx in range(m_z)))
    lines.append('M' + ''.join(f' {ancilla_X[row_idx]}' for row_idx in range(m_x)))
    

    # Build circuit
    circ = circEnc + stim.Circuit('TICK') + circChannel + stim.Circuit('TICK') + stim.Circuit("\n".join(lines))
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
             decIterations: int = 99,
             decSchedule: str = 'F',
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
    # breakpoint()
    print('generating logical operators...', end='', flush=True)
    # logical_X_ops, logical_Z_ops = logical_ops_from_checks.logical_ops_from_css(Hx, Hz)
    # Gz = gf2math.nullSpace(Hx)
    # Gx = gf2math.nullSpace(Hz)

    # Gz = logical_ops_from_checks.remove_dependents(Gz, Hz)
    # Gx = logical_ops_from_checks.remove_dependents(Gx, Hx)
    print('done.', end='', flush=True)



    m_z = Hz.shape[0] if Hz.size else 0
    m_x = Hx.shape[0] if Hx.size else 0

    def layerize(H:np.ndarray, serial:bool = False):
        layers = []
        m = H.shape[0]
        mUp = 1
        mDn = 0
        while mUp <= m:
            if np.max(np.sum(H[mDn:mUp,:], axis=0)) > 1 or (serial and mUp > mDn+1):
                layers.append(np.arange(mDn,mUp-1))
                mDn = mUp-1
            else:
                mUp += 1
        layers.append(np.arange(mDn,mUp-1))
        return layers

    match decSchedule:
        case "F":
            layersX = [np.arange(m_x)]
            layersZ = [np.arange(m_z)]
        case "L" | "S":
            layersX = layerize(Hx, serial=True if decSchedule== "S" else False)
            layersZ = layerize(Hz, serial=True if decSchedule== "S" else False)
        case _:
            raise ValueError("Unrecognized decoder scheduling option.")
        
    corrFailures = 0
    decFailures = 0
    nIterAccX = 0
    nIterAccZ = 0
    for shot_idx in range(shots):
        print(f'\rDecoding block n. {shot_idx+1:3}/{shots:4}...', end='', flush=True)
        row = samples[shot_idx]

        # Syndromes
        sy_z = row[:m_z].astype(int) if m_z else np.array([], dtype=int)
        sy_x = row[m_z:m_z+m_x].astype(int) if m_x else np.array([], dtype=int)

        # print(f"\nsyn_z is ", end='')
        # print("".join("_1"[e] for e in sy_z)) #, end = ' ')
        # # print("".join("_1"[e] for e in samDets[shot_idx,:m_z].astype(int)))
        # print(f"syn_x is ", end='')
        # print("".join("_1"[e] for e in sy_x)) #, end = ' ')
        # # print("".join("_1"[e] for e in samDets[shot_idx,m_z:].astype(int)))

        match decType:
            case "NG":
                eX_hat, nIterX = decoders.NG_decoder(Hz if Hz.size else np.zeros((0, n_data), dtype=int), sy_z)
                eZ_hat, nIterZ = decoders.NG_decoder(Hx if Hx.size else np.zeros((0, n_data), dtype=int), sy_x)
            case "BF":
                eX_hat, nIterX = decoders.BF_decoder(Hz if Hz.size else np.zeros((0, n_data), dtype=int), sy_z)
                eZ_hat, nIterZ = decoders.BF_decoder(Hx if Hx.size else np.zeros((0, n_data), dtype=int), sy_x)
            case "MS":
                eX_hat, nIterX = decoders.MS_decoder(Hz, sy_z, p=p/3, max_iter=decIterations, layers=layersX)
                eZ_hat, nIterZ = decoders.MS_decoder(Hx, sy_x, p=p/3, max_iter=decIterations, layers=layersZ)
            case "BP":
                eX_hat, nIterX = decoders.BP_decoder(Hz, sy_z, p=p/3, max_iter=decIterations, layers=layersX)
                eZ_hat, nIterZ = decoders.BP_decoder(Hx, sy_x, p=p/3, max_iter=decIterations, layers=layersZ)
            case _:
                raise ValueError("Unrecognized decoder type.")
                
        nIterAccX += nIterX
        nIterAccZ += nIterZ


        # Residuals are simply e_hat (since circuit applies error then we decode based on syndrome only)
        # Determine whether residuals anticommute with logical operators
        logical_failure = False
        # if any(Gx @ eX_hat) or any(Gz @ eZ_hat):
        #         logical_failure = True
        # if logical_failure:
        #     corrFailures += 1
            
        decoder_failure = False
        if not np.array_equal(sy_z, (Hz.dot(eX_hat)) % 2):
            decoder_failure = True
        if not np.array_equal(sy_x, (Hx.dot(eZ_hat)) % 2):
            decoder_failure = True
        if decoder_failure:
            decFailures +=1
        print(f'N. iterations (X, Z): {nIterX:2},{nIterZ:2}, Dec. failure rate: {decFailures/(shot_idx+1):.2e}             ', end='', flush=True)
        
    print()

    return {
        "LogicalErrors": corrFailures,
        "DecFailures": decFailures,
        "Avg_number_of_iterations_X": nIterAccX/float(shots),
        "Avg_number_of_iterations_Z": nIterAccZ/float(shots)
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stim-based QC-LDPC depolarizing-channel simulator.")
    parser.add_argument("--Hx", required=True, help="Path to Hx parity-check matrix (.npy).")
    parser.add_argument("--Hz", required=True, help="Path to Hz parity-check matrix (.npy).")
    parser.add_argument("--p", type=float, nargs='+', required=True, help="Depolarizing probability.")
    parser.add_argument("--shots", type=int, default=1000, help="Number of Monte Carlo shots.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed.")
    parser.add_argument("--decType", choices=['NG', 'BF', 'MS', 'BP'], default='MS', \
                        help="Decoder type: [NG] Naive Greedy; [MS] Min-Sum; [BP] Belief Propagation.")
    parser.add_argument("--decIterations", type=int, default=99, help="Number of decoding iterations.")
    parser.add_argument("--decSchedule", choices=['F','L','S'], default='F', \
                        help="Decoder scheduling method: [F] flooding; [L] layered; [S] serial.")
    args = parser.parse_args(argv)

    print(args)

    Hx = load_matrix(args.Hx)
    Hz = load_matrix(args.Hz)
    
    assert max(args.p) <= 1. and min(args.p) >= 0.

    results = []
    for pT in args.p:
        res = simulate(Hx, Hz, p=pT, shots=args.shots, rng_seed=args.seed, \
                       decType=args.decType, decIterations=args.decIterations, \
                    decSchedule=args.decSchedule)
        results.append(res)

    
    print('\n                    ===          SIMULATION RESULTS          ===\n')
    print('   Depolarizing probability | Decoding failures | Average iterations (X,Z)')
    print('----------------------------+-------------------+---------------------------')
    for i in range(len(args.p)):
        pT = args.p[i]
        print(f'         {pT:10.2e}         |       {results[i]['DecFailures']:5}       |      {results[i]['Avg_number_of_iterations_X']:5.2f}, {results[i]['Avg_number_of_iterations_Z']:5.2f}')




if __name__ == "__main__":
    main()

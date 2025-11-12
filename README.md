<div style="text-align: center;">
<img src="qLDPClogo.jpg" alt="qLDPCsim logo" style="width:25%; height:auto;"> <br>
A quantum LDPC simulator
</div>
---

## Brief description
qLDPCsim is a simulation toolkit for quantum LDPC (CSS-type) error correction codes focused on decoding algorithms.
qLDPCsim creates a [Stim](https://github.com/quantumlib/Stim) circuit based on a pair of parity check matrices **H_x**, **H_z**.
The Stim circuit
- generates a sequence of random logical qubits
- encodes the logical qubits to physical qubits
- simulates a depolarizing channel
- generates a pair of syndromes.

The syndromes are processed by the quantum LDPC decoder so as to obtain corresponding error sequences to be used for correction.
Currently available decoding algorithms are:
- conventional iterative belief-propagation
- the iterative sum-product algorithm
- the bit-flipping algorithm

More decoders are in my plans - stay tuned!!


## Source code structure
The source code is organized in modules:
- **simulate**: simulation execution and error statistics evaluation.
- **PCMlibrary**: a library of parity check matrix pairs (Hx, Hz).
- **decoders**: qLDPC decoders.
- **logical_ops_from_checks**: generation of logical ops for detecting failures.

Please be aware that qLDPCsim is at an initial development stage -- code stability is not guaranteed!


## Installation
Clone the git repository to a local drive, cd to the src subfolder.

Generate a pair of PCMs and save them to .npy files.
You can generate your own matrices, or you can take a pair from the library as follows: 
- In [?]: import numpy as np
- In [?]: from qLDPCsim import PCMlibrary as PCMlib
- In [?]: Hx, Hz = PCMlib.five_qubit_code()
- In [?]: np.save('Hx', Hx); np.save('Hz', Hz)

Once the PCMs are on files, you can run the simulation as follows:
- In [?]: import qLDPCsim
- In [?]: cd qLDPCsim
- In [?]: run simulate --Hx Hx.npy --Hz Hz.npy --shots 10000 --p 0.1 


Thanks for your interest in !
Alberto
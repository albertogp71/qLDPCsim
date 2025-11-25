<div style="text-align: center;">
<img src="qLDPClogo.jpg" alt="qLDPCsim logo" style="width:25%; height:auto;"> <br>
A quantum LDPC simulator
</div>

***

## Description
qLDPCsim is a simulation toolkit for quantum LDPC (CSS-type) error correction 
codes aimed at performance evaluation by Monte Carlo simulations.

Given a pair of parity-check matrices ${\bf H}_X$ and ${\bf H}_Z$, qLDPCsim 
creates a [Stim](https://github.com/quantumlib/Stim) circuit that consists of
an ecoder, a depolarizing channel and a syndrome generator.
The Stim circuit
- generates a sequence of $k$ random logical qubits;
- encodes the logical qubits into $n$ physical qubits;
- simulates depolarization on the $n$ physical qubits;
- determines the bit-flip and phase-flip syndromes.

The syndromes are fed to a quantum LDPC decoder so as to obtain corresponding
error sequences.
Currently available decoding algorithms are:
- conventional __Belief-Propagation__ (BP), a.k.a. sum-product;
- normalized __Min-Sum__ (MS);
- __Bit-Flipping__ (BF);
- a naive greedy algorithm.

BP and MS decoders operate according to one of the following check-node update
schedules:
1. flooding;
2. layered (automatic layer partitioning);
3. serial.

Evaluated performance>
1. **decoding failure rate** (separately for X and
Z);
2. **average number of iterations** (separately for X and Z).

A decoding failure event occurs when a decoder returns an error sequence that
does not match the corresponding syndrome.



## Source code structure
The source code is organized in modules:
- **simulate**: simulation execution and error statistics evaluation.
- **PCMlibrary**: a library of parity check matrix pairs (Hx, Hz).
- **decoders**: qLDPC decoders.
- **gf2math**: some useful GF(2) functions.

Note that qLDPCsim is work in progress -- code stability is not guaranteed!



## Installation

``pip install git+https://github.com/AlbertoGP71/qLDPCsim.git``.



## Usage

Generate a pair of PCMs and save them to .npy files.
You can generate your own matrices, or you can take a pair from the library as 
follows:  
``import numpy as np``  
``from qLDPCsim import PCMlibrary as PCMlib``  
``Hx, Hz = PCMlib.shor_code()``  
``np.save('Hx', Hx); np.save('Hz', Hz)``

Available PCM generators (see PCMlibrary.py):  
``shor_code()``  
``steane_code()``  
``bicycle_code()``  
``qc_ldpc_tanner_code()``  
``qc_ldpc_tanner_code()``  
``qc_ldpc_lifted_code(family: "LP04" or "LP118", index: 0 to 4)``  

Once the PCMs are on files ``Hx.npy`` and ``Hx.npy``, you can start a 
simulation as follows:  
``import qLDPCsim``  
``cd qLDPCsim``  
``run simulate --Hx path/to/Hx.npy --Hz path/to/Hz.npy --shots 10000 --p 0.01 0.02. 0.05 0.1 --shots 1000 --decType "MS" --decIterations 50 --decSchedule "L"``


Thanks for your interest in qLDPCsim!
Alberto

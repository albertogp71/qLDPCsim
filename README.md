<img src="qLDPClogo.jpg" alt="qLDPCsim logo" style="width:25%; height:auto;">

***

## Description
qLDPCsim is a simulation toolkit for quantum LDPC (CSS-type) error correction 
codes aimed at performance evaluation by Monte Carlo simulations.

Given a pair of parity-check matrices (PCM) ${\bf H}_X$ and ${\bf H}_Z$,
qLDPCsim evaluates the true quantum block (qBlock) error rate (qBLER) by
performing repeated encodings of $k$ logical qbits into $n$ physical qbits,
simulating physical qbit depolarization, and decoding the depolarized qbits.

Performance is evaluated by counting the following decoding events:  
. __Successful decoding with perfect match__: the estimated error is equal to 
the true channel error;
. __Successful decoding with degenerate error__: the difference between the
true channel error and the estimated error belongs to the stabilizer group;
. __Decoder failure__: the estimated error yields a different syndrome;
. __Logical error__: the difference between the true channel error and the
estimated error belongs to the normalizer group but is not a stabilizer.


qLDPCsim generates a [Stim](https://github.com/quantumlib/Stim) circuit that 
consists of an ecoder, a depolarizing channel and a syndrome generator.
The Stim circuit
- generates a sequence of $k$ random logical qubits;
- encodes the logical qubits into $n$ physical qubits;
- simulates depolarization on the $n$ physical qubits;
- determines the bit-flip and phase-flip syndromes;
- records the true X and Z errors introduced by the channel.

The syndromes are fed to a quantum LDPC decoder so as to obtain estimates of 
corresponding error sequences.
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

__[New in v0.2]___ BP and MS decoders can be configured to perform an optional
Ordered Statistics Decoding (OSD) post-decoding step.
Currently, only order 0 OSD is implemented.

Evaluated performance indicators are the following:
1. **quantum block (qBlock) error rate**. Ratio of quantum  block errors
(decoder failures + logical errors). 
2. **decoding failures** (separately for X and Z). Number of times the decoders 
failed to produce an error sequence that yields the syndrome.
3. **average number of iterations** (separately for X and Z).




## Source code structure
The source code is organized in modules:
- **simulator**: simulation execution and performance evaluation.
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

Available PCM generators (see PCMlibrary.py source code):  
``shor_code()``  
``steane_code()``  
``bicycle_code()``  
``qc_ldpc_tanner_code()``  
``qc_ldpc_lifted_code(family: "LP04" or "LP118", index: 0 to 4)``  

Once you have the PCM files ``Hx.npy`` and ``Hz.npy``, you can run a simulation 
as follows (example, assuming iPython CLI):  
``from qLDPCsim import simulator``  
``simulator.simulate(HxFile='path/to/Hx.npy', HzFile='path/to/Hz.npy', p=[0.01,0.02,0.05,0.1], shots=1000, decType='MS', decIterations=50, decSchedule='L')``


The function simulator.simulate() takes the following input arguments:
- HxFile : path to Hx.npy file
- HzFile : path to Hx.npy file
- p      : array of depolarization probabilities
- shots  : number of shots per probability point
- rngSeed: seed for RNG initialization
- decType: one of 'NG' (Naive Greedy), 'BF' (Bit Flipping), 'MS' (min-sum), 'BP' (belief propagation)
- decIterations: maximum number of decoding iterations
- decSchedule: one of 'F' (flooding), 'L' (layered), 'S' (serial)



Thanks for your interest in qLDPCsim!  
Alberto
albertogp71.github@gmail.com
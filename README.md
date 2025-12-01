<img src="qLDPClogo.jpg" alt="qLDPCsim logo" style="width:25%; height:auto;">

***

## Description
qLDPCsim is a simulation toolkit for quantum LDPC (CSS-type) error correction 
codes aimed at performance evaluation by Monte Carlo simulations.

Given a pair of parity-check matrices (PCM) ${\bf H}_X$ and ${\bf H}_Z$,
qLDPCsim estimates the quantum block (qBlock) error rate (qBLER) by
performing repeated encodings of $k$ randomly generated logical qbits into $n$
physical qbits, simulating physical qbit depolarization, and decoding the
depolarized qbits.

The performance is evaluated by counting the following decoding events:  
- __Successful decoding, perfect match__: the estimated error produced by
the decoder is equal to the true channel error;
- __Successful decoding, degenerate error__: the difference between the
true channel error and the estimated error belongs to the stabilizer group;
- __Logical error__: the difference between the true channel error and the
estimated error is not a stabilizer but belongs to the normalizer group;
- __Decoder failure__: the estimated error yields a different syndrome.


qLDPCsim generates a [Stim](https://github.com/quantumlib/Stim) circuit that 
consists of an ecoder, a depolarizing channel and a syndrome generator.
The Stim circuit
- generates a sequence of $k$ random logical qubits;
- encodes the logical qubits into $n$ physical qubits;
- simulates depolarization on the $n$ physical qubits;
- determines the bit-flip and phase-flip syndromes;
- records the true X and Z errors that the channel introduced.

The syndromes are fed to a quantum LDPC decoder so as to obtain estimates of 
corresponding error sequences.
Currently available decoding algorithms are:
- the conventional __Belief-Propagation__ (BP) algorithm, a.k.a. sum-product;
- the __Min-Sum__ (MS) algorithm with check node message normalization;
- the __Bit-Flipping__ (BF) algorithm;
- a naive greedy (NG) algorithm that flips valiables having largest number of 
unsatisfied checks.

BP and MS decoders may operate according to one of the following message update
schedules:
1. __flooding__: first update all check-to-variable messages, then update all
varible-to-check messages;
2. __layered__: check nodes are partitioned into layers; two check nodes that 
belong to the same layer do not have any adjacent variable nodes in common.
Layered BP updates all check-to-variable messages in a layer, then updates 
all the variable-to-check messages before processing another layer.
3. __serial__: a variant of the layered schedule where each layer contains one 
check node.

*[New in v0.2]* BP and MS decoders may perform an optional __Ordered
Statistics Decoding__ (OSD) decoding step after BP or MS iterations.

Evaluated performance indicators are the following:
1. **quantum block (qBlock) error rate**. Ratio of quantum  block errors
(decoder failures + logical errors). 
2. **decoding failures count** (separately for X and Z). Number of times the decoders 
failed to produce an error sequence that yields the given syndrome.
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
- OSDorder: order of OSD post-decoding. Set as -1 to disable OSD.


Thanks for your interest in qLDPCsim!  
Alberto  
albertogp71.github@gmail.com



# qLDPCsim
---
qLDPCsim is a toolkit for performance evaluation of quantum LDPC error correction codes focused on decoding algorithms.
Simulation of noise channels and generation of symdrome measurements is done using [stim](https://github.com/quantumlib/Stim).

qLDPCsim has the foollowing modules:
- **PCMlibrary**: a library of parity check matrix pairs (Hx, Hz).
- **decoders**: qLDPC decoders.
- **simulate**: simulation execution and error statistics evaluation.

qLDPC is at an initial development stage -- more features will be added.



## A quick user guide

Generate a pair of PCMs and save them to .npy files.
You can generate your own matrices or take a pair from the library as follows: 

In [??]: from qLDPCsim import PCMlibrary as PCMlib

In [??]: Hx, Hz = PCMlib.five_qubit_code)

In [??]: np.save('Hx', Hx); np.save('Hz', Hz)

Once the PCMs are on files, you can run the simulation as follows:

In [??]: run simulate --Hx Hx.npy --Hz Hz.npy --shots 10000 --p 0.1 


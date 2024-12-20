import numpy as np


def bitstring2bitarray(bitstring):
    # 'bn...b1b0' -> [b0,b1,...,bn] - compatible with Qiskit OptimizationResult.x
    return np.array([float(e) for i, e in enumerate(reversed(bitstring))])


def bitarray2bitstring(bitarray):
    return "".join(['0' if int(x) == 0 else '1' for x in reversed(bitarray)])


def unmap_qubits(bitarray, qc_isa):
    return [bitarray[i] for i in qc_isa.layout.final_index_layout()]


def unmap_bitstring(bitstring, qc_isa):
    return bitarray2bitstring(unmap_qubits(bitstring2bitarray(bitstring), qc_isa))

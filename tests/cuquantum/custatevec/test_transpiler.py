import random

import pytest
from qulacsvis import circuit_drawer
from quri_parts.circuit import QuantumCircuit, QuantumGate
from quri_parts.qulacs.circuit import convert_circuit
from quri_parts.stim.sampler import create_stim_clifford_sampler

from quri_parts.cuquantum.custatevec.transpiler import MultiGPUReorderingTranspiler


def _draw(circuit: QuantumCircuit):
    qc = convert_circuit(circuit)
    circuit_drawer(qc, "text", verbose=True)


def _random_clifford(n_qubits: int, n_gates: int, seed: int) -> QuantumCircuit:
    qubits = list(range(n_qubits))
    random.seed(seed)
    circuit = QuantumCircuit(n_qubits)
    circuit_reverse = QuantumCircuit(n_qubits)
    for _ in range(n_gates):
        name = random.choice(["X", "Y", "Z", "S", "Sdag", "H", "CNOT"])
        if name == "CNOT":
            qs = random.sample(qubits, k=2)
            circuit.add_CNOT_gate(*qs)
            circuit_reverse.add_CNOT_gate(*qs)
        else:
            q = random.choice(qubits)
            g = QuantumGate(name=name, target_indices=(q,))
            circuit.add_gate(g)
            if name == "S":
                circuit_reverse.add_gate(QuantumGate(name="Sdag", target_indices=(q,)))
            elif name == "Sdag":
                circuit_reverse.add_gate(QuantumGate(name="S", target_indices=(q,)))
            else:
                circuit_reverse.add_gate(g)
    reversed = list(circuit_reverse.gates)
    reversed.reverse()
    circuit.extend(reversed)
    return circuit


def _check_circuit(n_local_qubits: int, circuit: QuantumCircuit) -> bool:
    local_qubits = set(range(n_local_qubits))
    for gate in circuit.gates:
        if gate.name != "SWAP":
            targ = set(gate.control_indices) | set(gate.target_indices)
            if not targ.issubset(local_qubits):
                return False
    return True


@pytest.mark.parametrize(
    ["seed", "n_qubits", "n_local_qubits", "n_gates"],
    [
        (0, 5, 2, 20),
        (0, 5, 3, 20),
        (0, 5, 4, 20),
        (0, 10, 5, 40),
        (0, 10, 9, 40),
        (0, 10, 4, 39),
        (0, 10, 3, 39),
        (0, 10, 2, 39),
        (0, 9, 3, 29),
        (0, 9, 2, 24),
        (0, 8, 4, 36),
        (0, 8, 3, 29),
        (0, 8, 2, 29),
        (0, 7, 5, 34),
        (0, 7, 4, 28),
        (0, 7, 3, 28),
        (0, 7, 2, 16),
        (0, 32, 32, 50),
        (0, 33, 32, 50),
        (0, 34, 32, 50),
        (0, 35, 32, 50),
    ],
)
def test_transpiler_random(
    seed: int, n_qubits: int, n_local_qubits: int, n_gates: int
) -> None:
    transpiler = MultiGPUReorderingTranspiler(n_local_qubits)
    orig_circuit = _random_clifford(n_qubits, n_gates, seed)
    print("original")
    _draw(orig_circuit)
    transpiled_circuit = transpiler(orig_circuit)
    print("transpile")
    _draw(transpiled_circuit)
    assert _check_circuit(n_local_qubits, transpiled_circuit)
    sampler = create_stim_clifford_sampler()
    assert set(sampler(orig_circuit, 3)) == {0}
    assert set(sampler(transpiled_circuit, 3)) == {0}

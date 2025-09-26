# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from quri_parts.circuit import QuantumCircuit, QuantumGate
from quri_parts.cuquantum.custatevec.sampler import (
    create_cuquantum_vector_sampler,
)
from quri_parts.cuquantum.custatevec.estimator import (
    create_cuquantum_vector_estimator,
)
from quri_parts.core.operator import Operator, pauli_label
from quri_parts.core.state import GeneralCircuitQuantumState
import math
import cmath
import random


def random_clifford(n_qubits: int, n_gates: int, seed: int):
    qubits = list(range(n_qubits))
    random.seed(seed)
    circuit = QuantumCircuit(n_qubits)
    for _ in range(n_gates):
        name = random.choice(["X", "Y", "Z", "S", "Sdag", "H", "CNOT"])
        if name == "CNOT":
            qs = random.sample(qubits, k=2)
            circuit.add_CNOT_gate(*qs)
        else:
            q = random.choice(qubits)
            g = QuantumGate(name=name, target_indices=(q,))
            circuit.add_gate(g)

    reverse = QuantumCircuit(n_qubits)
    for g in reversed(circuit.gates):
        if g.name == "S":
            reverse.add_Sdag_gate(g.target_indices[0])
        elif g.name == "Sdag":
            reverse.add_S_gate(g.target_indices[0])
        else:
            reverse.add_gate(g)

    return circuit, reverse


def cat_state(qubits: int) -> QuantumCircuit:
    circuit = QuantumCircuit(qubits)
    circuit.add_H_gate(0)
    for i in range(1, qubits):
        j = int(math.floor((i + 1) / 2)) - 1
        circuit.add_CNOT_gate(j, i)
    return circuit


def make_circuit(qubits: int, bits: int) -> QuantumCircuit:
    circuit = cat_state(qubits)
    for q in range(qubits):
        if bits % 2 == 1:
            circuit.add_X_gate(q)
        bits = bits // 2
    assert bits == 0
    circuit_a, circuit_b = random_clifford(qubits, 6 * qubits, 0)
    circuit.extend(circuit_a)
    circuit.extend(circuit_b)
    return circuit


@pytest.mark.parametrize("local_qubits", [16, 32])
@pytest.mark.parametrize("global_qubits", [0, 1, 2, 3])
def test_sampler_mixed(local_qubits: int, global_qubits: int) -> None:
    qubits = local_qubits + global_qubits
    circuit = QuantumCircuit(qubits)
    for q in range(qubits):
        circuit.add_H_gate(q)

    sampler = create_cuquantum_vector_sampler(n_global_qubits=global_qubits)
    counts = sampler(circuit, 10000)
    limit = 2**(qubits - 1)
    samples = set(counts.keys())
    count = len([1 for s in samples if s > limit])
    assert count > 10000 / 2**(qubits - 1) - 5

@pytest.mark.parametrize("local_qubits", [32])
@pytest.mark.parametrize("global_qubits", [0, 1, 2, 3])
def test_sampler_cat(local_qubits: int, global_qubits: int) -> None:
    qubits = local_qubits + global_qubits
    circuit = cat_state(qubits)

    sampler = create_cuquantum_vector_sampler(n_global_qubits=global_qubits)
    counts = sampler(circuit, 10)
    samples = set(counts.keys())

    assert samples == {0, 2**qubits - 1}
    assert sum(counts.values()) == 10

@pytest.mark.parametrize("local_qubits", [32])
@pytest.mark.parametrize("global_qubits", [0, 1, 2])
def test_sampler_large(local_qubits: int, global_qubits: int) -> None:
    qubits = local_qubits + global_qubits
    bits = sum([2**m for m in range(0, qubits, 2)])
    circuit = make_circuit(qubits, bits)

    sampler = create_cuquantum_vector_sampler(n_global_qubits=global_qubits)
    counts = sampler(circuit, 10)
    samples = set(counts.keys())

    assert samples == {bits, 2**qubits - 1 - bits}
    assert sum(counts.values()) == 10

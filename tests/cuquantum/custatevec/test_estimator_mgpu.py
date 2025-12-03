# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cmath
import math
import random

import numpy as np
import pytest
from quri_parts.circuit import QuantumCircuit, QuantumGate
from quri_parts.core.operator import Operator, pauli_label
from quri_parts.core.state import GeneralCircuitQuantumState
from quri_parts.stim.estimator import create_stim_clifford_estimator

from quri_parts.cuquantum.custatevec.estimator import create_cuquantum_vector_estimator


def make_circuit(qubits: int) -> QuantumCircuit:
    circuit = QuantumCircuit(qubits)
    circuit.add_H_gate(0)
    for i in range(1, qubits):
        j = int(math.floor((i + 1) / 2)) - 1
        circuit.add_CNOT_gate(j, i)
    return circuit


def _random_clifford(n_qubits: int, n_gates: int, seed: int) -> QuantumCircuit:
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
    return circuit


@pytest.mark.parametrize("qubits", [12])
@pytest.mark.parametrize("global_qubits", [0])
@pytest.mark.parametrize("axis", ["Z"])
def test_general_estimator(qubits: int, global_qubits: int, axis: str) -> None:
    circuit = make_circuit(qubits)
    estimator = create_cuquantum_vector_estimator(n_global_qubits=global_qubits)
    label = " ".join([f"{axis}{i}" for i in range(6)])
    state = GeneralCircuitQuantumState(qubits, circuit)
    operator = Operator({pauli_label(label): 1.0})
    estimate = estimator(operator, state)
    assert math.isclose(1.0, estimate.value.real)
    assert math.isclose(0.0, estimate.value.imag)


@pytest.mark.parametrize("qubits", [12])
@pytest.mark.parametrize("gates", [2])
@pytest.mark.parametrize("global_qubits", [0, 1, 2, 3])
def test_estimator_with_random_circuit(
    qubits: int, gates: int, global_qubits: int
) -> None:
    q = random.sample(range(qubits), 4)
    operator = operator = Operator(
        {
            pauli_label(f"Z{q[0]} Y{q[1]} Z{q[3]}"): 0.25,
            pauli_label(f"Z{q[0]} X{q[2]} Y{q[3]}"): 0.33,
            pauli_label(f"X{q[1]} Z{q[2]} Z{q[3]}"): 0.77,
        }
    )
    circuit = _random_clifford(qubits, gates, 0)
    state = GeneralCircuitQuantumState(qubits, circuit)
    estimator_stim = create_stim_clifford_estimator()
    estimator_cusv = create_cuquantum_vector_estimator(n_global_qubits=global_qubits)
    estimate_stim = estimator_stim(operator, state)
    estimate_cusv = estimator_cusv(operator, state)
    assert np.allclose(
        [estimate_cusv.value.real, estimate_cusv.value.imag],
        [estimate_stim.value.real, estimate_stim.value.imag],
    )

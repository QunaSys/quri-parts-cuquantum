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
from quri_parts.circuit import QuantumCircuit
from quri_parts.core.state import quantum_state
from quri_parts.qulacs.simulator import (
    evaluate_state_to_vector as qulacs_evaluate_state_to_vector,
)

from quri_parts.cuquantum.custatevec.simulator import evaluate_state_to_vector


@pytest.mark.parametrize("qubits", [4, 12])
def test_evaluate_state_to_vector(qubits: int) -> None:
    circuit = QuantumCircuit(qubits)
    for i in range(qubits):
        circuit.add_H_gate(i)
    circuit.add_X_gate(0)
    circuit.add_H_gate(1)
    circuit.add_Z_gate(2)
    circuit.add_Y_gate(3)
    circuit.add_PauliRotation_gate((0, 2, 3), (1, 2, 3), 0.1)

    state = quantum_state(n_qubits=qubits, circuit=circuit)
    vector = evaluate_state_to_vector(state).vector
    target_state = quantum_state(n_qubits=qubits, circuit=circuit)
    target_vector = qulacs_evaluate_state_to_vector(target_state).vector

    assert all(abs(v - tv) < 1e-6 for v, tv in zip(vector, target_vector))

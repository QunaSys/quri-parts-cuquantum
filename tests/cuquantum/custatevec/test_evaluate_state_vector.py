# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.circuit import QuantumCircuit
from quri_parts.core.state import quantum_state

from quri_parts.cuquantum.custatevec.simulator import evaluate_state_to_vector


def test_evaluate_state_vector():
    circuit = QuantumCircuit(4)
    circuit.add_H_gate(3)
    state = quantum_state(4)

    state = state.with_gates_applied(circuit)

    state = evaluate_state_to_vector(state)
    assert abs(1.0 - sum([(c * c.conjugate()).real for c in state.vector])) < 1e-5

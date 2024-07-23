# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import numpy as np
import pytest
from quri_parts.circuit import H, S, UnboundParametricQuantumCircuit
from quri_parts.core.estimator import (
    create_concurrent_estimator_from_estimator,
    create_concurrent_parametric_estimator_from_concurrent_estimator,
    create_parametric_estimator_from_concurrent_estimator,
)
from quri_parts.core.operator import Operator, pauli_label
from quri_parts.core.state import (
    ComputationalBasisState,
    ParametricCircuitQuantumState,
    ParametricQuantumStateVector,
    QuantumStateVector,
    StateVectorType,
)

from quri_parts.cuquantum.custatevec.estimator import (
    _estimate,
    create_cuquantum_general_vector_estimator,
    create_cuquantum_vector_estimator,
)


def vector(qubit_count: int, bits: int) -> StateVectorType:
    vector: StateVectorType = np.zeros(2**qubit_count, dtype=np.cfloat)
    vector[bits] = 1.0
    return vector


def parametric_circuit() -> UnboundParametricQuantumCircuit:
    circuit = UnboundParametricQuantumCircuit(6)

    circuit.add_RX_gate(0, -math.pi / 4)
    circuit.add_ParametricRX_gate(0)

    circuit.add_RY_gate(2, -math.pi / 4)
    circuit.add_ParametricRY_gate(2)

    circuit.add_H_gate(5)
    circuit.add_RZ_gate(5, -math.pi / 4)
    circuit.add_ParametricRZ_gate(5)

    circuit.add_ParametricPauliRotation_gate((0, 2, 5), (1, 2, 3))
    return circuit


def test_estimate_pauli_label() -> None:
    qubit_count = 6

    estimator = create_cuquantum_vector_estimator()

    p_label = pauli_label("Z2 Z0 Z5")
    state = ComputationalBasisState(qubit_count, bits=0b100101)
    assert math.isclose(estimator(p_label, state).value, -1.0)

    p_label = pauli_label("Z2 X0 Z5")
    state_h_appl = state.with_gates_applied([H(0)])
    assert math.isclose(estimator(p_label, state_h_appl).value, -1.0)


def test_estimate_operator() -> None:
    qubit_count = 6
    op = Operator(
        {
            pauli_label("Z0"): 1.0,
            pauli_label("Z0 Z2"): 2.0,
            pauli_label("X1 X3 X5"): 3.0,
            pauli_label("Y0 Y2 Y4"): 4.0,
            pauli_label("Z0 Y1 X2"): 5.0,
        }
    )
    estimator = create_cuquantum_vector_estimator()

    state = ComputationalBasisState(qubit_count, bits=0b0)
    assert math.isclose(estimator(op, state).value, 3.0)

    state = ComputationalBasisState(qubit_count, bits=0b101)
    assert math.isclose(estimator(op, state).value, 1.0)

    state = ComputationalBasisState(qubit_count, bits=0b101000)
    state_h_appl = state.with_gates_applied([H(1), H(3), H(5)])
    assert math.isclose(estimator(op, state_h_appl).value, 6.0)

    state = ComputationalBasisState(qubit_count, bits=0b10101)
    state_g_appl = state.with_gates_applied([H(0), S(0), H(2), S(2), H(4), S(4)])
    assert math.isclose(estimator(op, state_g_appl).value, -4.0)

    state = ComputationalBasisState(qubit_count, bits=0b111)
    state_g_appl = state.with_gates_applied([H(1), S(1), H(2)])
    assert math.isclose(estimator(op, state_g_appl).value, -6.0)


def test_estimate_invalid_precision() -> None:
    qubit_count = 6
    op = Operator(
        {
            pauli_label("Z0"): 1.0,
            pauli_label("Z0 Z2"): 2.0,
            pauli_label("X1 X3 X5"): 3.0,
            pauli_label("Y0 Y2 Y4"): 4.0,
            pauli_label("Z0 Y1 X2"): 5.0,
        }
    )
    state = ComputationalBasisState(qubit_count, bits=0b0)

    with pytest.raises(ValueError):
        _estimate(op, state, precision="complex32")


def test_general_estimator() -> None:
    general_estimator = create_cuquantum_general_vector_estimator()

    pauli = pauli_label("Z0 Z2 Z5")
    operator = Operator(
        {
            pauli_label("Z0 Z2 Z5"): 0.25,
            pauli_label("Z1 Z2 Z4"): 0.5j,
        }
    )

    state_1 = ComputationalBasisState(6, bits=0b110000)
    state_2 = ComputationalBasisState(6, bits=0b110010)
    state_3 = QuantumStateVector(6, vector(6, 0b110000))

    # test estimator
    estimator = create_cuquantum_vector_estimator()
    assert general_estimator(pauli, state_1) == estimator(pauli, state_1)
    assert general_estimator(operator, state_2) == estimator(operator, state_2)
    assert general_estimator(pauli, state_3) == estimator(pauli, state_3)

    # test concurrent estimator
    concurrent_estimator = create_concurrent_estimator_from_estimator(estimator)

    estimates = concurrent_estimator([pauli], [state_1, state_2, state_3])
    assert general_estimator(pauli, [state_1, state_2, state_3]) == estimates

    estimates = concurrent_estimator([operator], [state_1, state_2, state_3])
    assert general_estimator([operator], [state_1, state_2, state_3]) == estimates

    estimates = concurrent_estimator([pauli, operator], [state_1])
    assert general_estimator([pauli, operator], state_1) == estimates

    estimates = concurrent_estimator([pauli, operator], [state_2])
    assert general_estimator([pauli, operator], [state_2]) == estimates

    estimates = concurrent_estimator([pauli, operator], [state_1, state_2])
    assert general_estimator([pauli, operator], [state_1, state_2]) == estimates

    # parametric estimator
    param_circuit = parametric_circuit()
    p_circuit_state = ParametricCircuitQuantumState(
        param_circuit.qubit_count, param_circuit
    )
    p_vector_state = ParametricQuantumStateVector(
        param_circuit.qubit_count,
        param_circuit,
        vector=vector(param_circuit.qubit_count, bits=1),
    )
    params = [np.random.random(param_circuit.parameter_count) for _ in range(4)]

    # test parametric estimator
    parametric_estimator = create_parametric_estimator_from_concurrent_estimator(
        concurrent_estimator
    )
    assert general_estimator(pauli, p_circuit_state, params[0]) == parametric_estimator(
        pauli, p_circuit_state, params[0].tolist()
    )
    assert general_estimator(pauli, p_vector_state, params[0]) == parametric_estimator(
        pauli, p_vector_state, params[0].tolist()
    )

    # test concurrent parametric estimator
    cp_estimator = create_concurrent_parametric_estimator_from_concurrent_estimator(
        concurrent_estimator
    )
    assert general_estimator(pauli, p_circuit_state, params) == cp_estimator(
        pauli, p_circuit_state, [ps.tolist() for ps in params]
    )
    assert general_estimator(pauli, p_vector_state, params) == cp_estimator(
        pauli, p_vector_state, [ps.tolist() for ps in params]
    )

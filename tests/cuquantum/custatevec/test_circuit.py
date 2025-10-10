# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
from quri_parts.circuit import (
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    SWAP,
    U1,
    U2,
    U3,
    H,
    Pauli,
    PauliRotation,
    S,
    Sdag,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
    TOFFOLI,
    X,
    Y,
    Z,
)
from quri_parts.cuquantum.custatevec.circuit import gate_array


def test_gate_array() -> None:
    assert np.allclose(gate_array(X(0)), [0.0, 1.0, 1.0, 0.0])
    assert np.allclose(gate_array(Y(1)), [0.0, -1.0j, 1.0j, 0.0])
    assert np.allclose(gate_array(Z(2)), [1.0, 0.0, 0.0, -1.0])
    assert np.allclose(
        gate_array(H(3)),
        [1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 1.0 / np.sqrt(2), -1.0 / np.sqrt(2)],
    )
    assert np.allclose(gate_array(S(4)), [1.0, 0.0, 0.0, 1.0j])
    assert np.allclose(gate_array(Sdag(5)), [1.0, 0.0, 0.0, -1.0j])
    assert np.allclose(
        gate_array(SqrtX(6)),
        [(1.0 + 1.0j) / 2, (1.0 - 1.0j) / 2, (1.0 - 1.0j) / 2, (1.0 + 1.0j) / 2],
    )
    assert np.allclose(
        gate_array(SqrtXdag(7)),
        [(1.0 - 1.0j) / 2, (1.0 + 1.0j) / 2, (1.0 + 1.0j) / 2, (1.0 - 1.0j) / 2],
    )
    assert np.allclose(
        gate_array(SqrtY(8)),
        [(1.0 + 1.0j) / 2, (-1.0 - 1.0j) / 2, (1.0 + 1.0j) / 2, (1.0 + 1.0j) / 2],
    )
    assert np.allclose(
        gate_array(SqrtYdag(9)),
        [(1.0 - 1.0j) / 2, (1.0 - 1.0j) / 2, (-1.0 + 1.0j) / 2, (1.0 - 1.0j) / 2],
    )
    assert np.allclose(gate_array(CNOT(0, 1)), [0.0, 1.0, 1.0, 0.0])
    assert np.allclose(gate_array(CZ(0, 1)), [1.0, 0.0, 0.0, -1.0])
    assert np.allclose(
        gate_array(SWAP(0, 1)),
        [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    )
    assert np.allclose(gate_array(TOFFOLI(0, 1, 2)), [0.0, 1.0, 1.0, 0.0])
    assert np.allclose(
        gate_array(RX(0, np.pi / 2)),
        [1.0 / np.sqrt(2), -1.0j / np.sqrt(2), -1.0j / np.sqrt(2), 1.0 / np.sqrt(2)],
    )
    assert np.allclose(gate_array(RY(0, np.pi)), [0.0, -1.0, 1.0, 0.0])
    assert np.allclose(
        gate_array(RZ(0, 3 * np.pi / 2)),
        [(-1.0 - 1.0j) / np.sqrt(2), 0.0, 0.0, (-1.0 + 1.0j) / np.sqrt(2)],
    )
    assert np.allclose(
        gate_array(U1(0, np.pi / 2)),
        [1.0, 0.0, 0.0, 1.0j],
    )
    assert np.allclose(
        gate_array(U2(0, np.pi, np.pi / 2)),
        [1.0 / np.sqrt(2), -1.0j / np.sqrt(2), -1.0 / np.sqrt(2), -1.0j / np.sqrt(2)],
    )
    assert np.allclose(
        gate_array(U3(0, np.pi, -np.pi / 2, 3 * np.pi / 2)), [0.0, 1.0j, -1.0j, 0.0]
    )

    with pytest.raises(ValueError):
        assert gate_array(Pauli([0, 1, 2], [1, 2, 3]))
    with pytest.raises(ValueError):
        gate_array(PauliRotation([0, 1], [1, 2], 1.0))


def test_gate_array_dtype() -> None:
    assert gate_array(X(0), dtype="complex64").dtype == np.complex64
    assert gate_array(X(0), dtype="complex128").dtype == np.complex128

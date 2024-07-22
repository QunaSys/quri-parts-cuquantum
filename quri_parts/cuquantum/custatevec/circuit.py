# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import numpy as np
import numpy.typing as npt
from quri_parts.circuit import NonParametricQuantumCircuit, QuantumGate, gate_names
from quri_parts.circuit.gate_names import GateNameType, is_gate_name

gate_map: dict[GateNameType, npt.NDArray] = {
    gate_names.Identity: np.array([1.0, 0.0, 0.0, 1.0]),
    gate_names.X: np.array([0.0, 1.0, 1.0, 0.0]),
    gate_names.Y: np.array([0.0, -1.0j, 1.0j, 0.0]),
    gate_names.Z: np.array([1.0, 0.0, 0.0, -1.0]),
    gate_names.H: np.array([1.0, 1.0, 1.0, -1.0]) / np.sqrt(2),
    gate_names.S: np.array([1.0, 0.0, 0.0, 1.0j]),
    gate_names.Sdag: np.array([1.0, 0.0, 0.0, -1.0j]),
    gate_names.SqrtX: np.array([0.5 + 0.5j, 0.5 - 0.5j, 0.5 - 0.5j, 0.5 + 0.5j]),
    gate_names.SqrtXdag: np.array([0.5 - 0.5j, 0.5 + 0.5j, 0.5 + 0.5j, 0.5 - 0.5j]),
    gate_names.SqrtY: np.array([0.5, -0.5, 0.5, 0.5]) * (1 + 1.0j),
    gate_names.SqrtYdag: np.array([0.5, 0.5, -0.5, 0.5]) * (1 - 1.0j),
    gate_names.T: np.array([1.0, 0.0, 0.0, np.exp(1.0j * np.pi / 4)]),
    gate_names.Tdag: np.array([1.0, 0.0, 0.0, np.exp(-1.0j * np.pi / 4)]),
    gate_names.CNOT: np.array([0.0, 1.0, 1.0, 0.0]),
    gate_names.CZ: np.array([1.0, 0.0, 0.0, -1.0]),
    gate_names.SWAP: np.array(
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
        ]
    ),
    gate_names.TOFFOLI: np.array([0.0, 1.0, 1.0, 0.0]),
}


def gate_array(gate: QuantumGate, dtype: str = "complex128") -> Optional[npt.NDArray]:
    """Returns the array representation of given gate.

    Currently Pauli, PauliRotation, and measurement gates are not
    supported.
    """
    if gate.name in gate_map and is_gate_name(gate.name):
        return gate_map[gate.name].astype(dtype)
    p = gate.params
    if gate.name == "RX":
        return np.array(
            [
                np.cos(p[0] / 2),
                -1.0j * np.sin(p[0] / 2),
                -1.0j * np.sin(p[0] / 2),
                np.cos(p[0] / 2),
            ],
            dtype=dtype,
        )
    elif gate.name == "RY":
        return np.array(
            [
                np.cos(p[0] / 2),
                -np.sin(p[0] / 2),
                np.sin(p[0] / 2),
                np.cos(p[0] / 2),
            ],
            dtype=dtype,
        )
    elif gate.name == "RZ":
        return np.array(
            [
                np.exp(-1.0j * p[0] / 2),
                0.0,
                0.0,
                np.exp(1.0j * p[0] / 2),
            ],
            dtype=dtype,
        )
    elif gate.name == "U1":
        return np.array(
            [
                1.0,
                0.0,
                0.0,
                np.exp(1.0j * p[0]),
            ],
            dtype=dtype,
        )
    elif gate.name == "U2":
        return np.array(
            [
                1.0 / np.sqrt(2),
                -np.exp(1.0j * p[1]) / np.sqrt(2),
                np.exp(1.0j * p[0]) / np.sqrt(2),
                np.exp(1.0j * (p[0] + p[1])) / np.sqrt(2),
            ],
            dtype=dtype,
        )
    elif gate.name == "U3":
        return np.array(
            [
                np.cos(p[0] / 2),
                -np.exp(1.0j * p[2]) * np.sin(p[0] / 2),
                np.exp(1.0j * p[1]) * np.sin(p[0] / 2),
                np.exp(1.0j * (p[1] + p[2])) * np.cos(p[0] / 2),
            ],
            dtype=dtype,
        )
    elif gate.name == "UnitaryMatrix":
        return np.array(gate.unitary_matrix, dtype=dtype).flatten()
    elif gate.name == "Measurement":
        return None
    raise ValueError(f"{gate.name} gate is not supported.")

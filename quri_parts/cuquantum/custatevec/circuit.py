# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence

try:
    import cupy as cp
except ImportError:
    raise RuntimeError("CuPy is not installed.")
import numpy as np
import numpy.typing as npt
from quri_parts.circuit import QuantumGate, gate_names
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

cuda_code = r"""
__constant__ double RX_LUT[4] = {1, 0, 0, 1};
__constant__ double RY_COS_LUT[4] = {1, 0, 0, 1};
__constant__ double RY_SIN_LUT[4] = {0, -1, 1, 0};
__constant__ double RZ_LUT[4] = {1, 0, 0, 1};
__constant__ double M_SQRT1_2 = 0.7071067811865476;

extern "C" __global__
void initialize_gate_matrix(double2 *mat, int gate_type, double param1, double param2, double param3) {
    int i = threadIdx.x;
    double cost = cos(param1 / 2.0);
    double sint = sin(param1 / 2.0);

    if (gate_type == 0) { // RX
        mat[i].x = cost * RX_LUT[i];
        mat[i].y = -sint * (1 - RX_LUT[i]);
    } else if (gate_type == 1) { // RY
        mat[i].x = cost * RY_COS_LUT[i] + sint * RY_SIN_LUT[i];
        mat[i].y = 0;
    } else if (gate_type == 2) { // RZ
        mat[i].x = cost * RZ_LUT[i];
        mat[i].y = -sint * (i == 0 ? 1 : (i == 3 ? -1 : 0));
    } else if (gate_type == 3) { // U1 (Phase gate)
        if (i == 0) {
            mat[i].x = 1;
            mat[i].y = 0;
        } else if (i == 3) {
            mat[i].x = cos(param1);
            mat[i].y = sin(param1);
        } else {
            mat[i].x = 0;
            mat[i].y = 0;
        }
    } else if (gate_type == 4) { // U2
        if (i == 0) {
            mat[i].x = M_SQRT1_2;  // 1/sqrt(2)
            mat[i].y = 0;
        } else if (i == 1) {
            mat[i].x = -M_SQRT1_2 * cos(param2);
            mat[i].y = -M_SQRT1_2 * sin(param2);
        } else if (i == 2) {
            mat[i].x = M_SQRT1_2 * cos(param1);
            mat[i].y = M_SQRT1_2 * sin(param1);
        } else {
            mat[i].x = M_SQRT1_2 * cos(param1 + param2);
            mat[i].y = M_SQRT1_2 * sin(param1 + param2);
        }
    } else if (gate_type == 5) { // U3
        if (i == 0) {
            mat[i].x = cos(param1 / 2.0);
            mat[i].y = 0;
        } else if (i == 1) {
            mat[i].x = -sin(param1 / 2.0) * cos(param3);
            mat[i].y = -sin(param1 / 2.0) * sin(param3);
        } else if (i == 2) {
            mat[i].x = sin(param1 / 2.0) * cos(param2);
            mat[i].y = sin(param1 / 2.0) * sin(param2);
        } else {
            mat[i].x = cos(param1 / 2.0) * cos(param2 + param3);
            mat[i].y = cos(param1 / 2.0) * sin(param2 + param3);
        }
    }
}
"""


initialize_kernel = cp.RawKernel(cuda_code, "initialize_gate_matrix")

# dry run to compile the kernel
mat = cp.empty(4, dtype=cp.complex128)
initialize_kernel((1,), (4,), (mat, 0, 0.0, 0.0, 0.0))


def fast_gate_array(gate_name: str, params: Sequence[float]) -> cp.ndarray:
    mat = cp.empty(4, dtype=cp.complex128)

    gate_types = {"RX": 0, "RY": 1, "RZ": 2, "U1": 3, "U2": 4, "U3": 5}

    gate_type = gate_types.get(gate_name, -1)
    if gate_type == -1:
        raise ValueError(f"Unsupported gate: {gate_name}")

    if gate_type <= 3:
        initialize_kernel((1,), (4,), (mat, gate_type, params[0], 0.0, 0.0))
    elif gate_type <= 4:
        initialize_kernel((1,), (4,), (mat, gate_type, params[0], params[1], 0.0))
    else:
        initialize_kernel((1,), (4,), (mat, gate_type, params[0], params[1], params[2]))
    return mat


rot_gates = set(["RX", "RY", "RZ", "U1", "U2", "U3"])


def gate_array(gate: QuantumGate, dtype: str = "complex128") -> Optional[cp.ndarray]:
    """Returns the array representation of given gate.

    Currently Pauli, PauliRotation, and measurement gates are not
    supported.
    """
    if gate.name in gate_map and is_gate_name(gate.name):
        return cp.array(gate_map[gate.name], dtype=dtype)
    p = gate.params
    if gate.name in rot_gates:
        return fast_gate_array(gate.name, p)
    elif gate.name == "UnitaryMatrix":
        return cp.array(
            np.array(gate.unitary_matrix, dtype=dtype).flatten(), dtype=dtype
        )
    elif gate.name == "Measurement":
        return None
    raise ValueError(f"{gate.name} gate is not supported.")

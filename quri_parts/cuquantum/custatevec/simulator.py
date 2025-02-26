# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

try:
    import cupy as cp
except ImportError:
    cp = None
try:
    import cuquantum
except ImportError:
    cuquantum = None
import numpy as np
from quri_parts.circuit.transpile import (
    ParallelDecomposer,
    PauliDecomposeTranspiler,
    PauliRotationDecomposeTranspiler,
)
from quri_parts.core.state import CircuitQuantumState, QuantumStateVector
from quri_parts.circuit import NonParametricQuantumCircuit

from . import PRECISIONS, Precision
from .circuit import gate_array, gate_map

gates_to_cache = set()
for gate_name in gate_map.keys():
    gates_to_cache.add(gate_name)

rot_gates_to_cache = set(["RX", "RY", "RZ", "U1", "U2", "U3"])

def _update_statevector(
    circuit: NonParametricQuantumCircuit,
    sv: cp.ndarray,
    qubit_count: int,
    precision: Precision,
    handle: int,
    cuda_d_type: cuquantum.cudaDataType,
    cuda_c_type: cuquantum.ComputeType,
) -> None:
    mat_dict = {}
    rot_mat_dict = {}
    rot_mat_dict_cnt = 0
    max_rot_mat_dict_cnt = 1e7
    for rot_gate in rot_gates_to_cache:
        rot_mat_dict[rot_gate] = {}
    qubits_cache = {}
    qubits_ptr_cache = {}

    for g in circuit.gates:
        len_targets = len(g.target_indices)
        len_controls = len(g.control_indices)
        if len_targets <= 2:
            if g.target_indices not in qubits_cache:
                qubits_cache[g.target_indices] = np.array(
                    g.target_indices, dtype=np.int32
                )
                qubits_ptr_cache[g.target_indices] = qubits_cache[
                    g.target_indices
                ].ctypes.data
            targets = qubits_cache[g.target_indices]
            targets_ptr = qubits_ptr_cache[g.target_indices]
        else:
            targets = np.array(g.target_indices, dtype=np.int32)
            targets_ptr = targets.ctypes.data
        if len_controls <= 2:
            if g.control_indices not in qubits_cache:
                qubits_cache[g.control_indices] = np.array(
                    g.control_indices, dtype=np.int32
                )
                qubits_ptr_cache[g.control_indices] = qubits_cache[
                    g.control_indices
                ].ctypes.data
            controls = qubits_cache[g.control_indices]
            controls_ptr = qubits_ptr_cache[g.control_indices]
        else:
            controls = np.array(g.control_indices, dtype=np.int32)
            controls_ptr = controls.ctypes.data

        if g.name in gates_to_cache:
            if g.name not in mat_dict:
                mat = gate_array(g, dtype=precision)
                mat_dict[g.name] = mat
            else:
                mat = mat_dict[g.name]
        elif g.name in rot_gates_to_cache:
            if (
                rot_mat_dict_cnt <= max_rot_mat_dict_cnt
                and g.params not in rot_mat_dict[g.name]
            ):
                mat = gate_array(g, dtype=precision)
                rot_mat_dict[g.name][g.params] = mat
                rot_mat_dict_cnt += 1
            else:
                mat = rot_mat_dict[g.name][g.params]
        else:
            mat = gate_array(g, dtype=precision)
        mat_ptr = mat.data.ptr  # type: ignore

        workspace_size = cuquantum.custatevec.apply_matrix_get_workspace_size(
            handle,
            cuda_d_type,
            qubit_count,
            mat_ptr,
            cuda_d_type,
            cuquantum.custatevec.MatrixLayout.ROW,
            0,
            len_targets,
            len_controls,
            cuda_c_type,
        )

        # check the size of external workspace
        if workspace_size > 0:
            workspace = cp.cuda.memory.alloc(workspace_size)
            workspace_ptr = workspace.ptr
        else:
            workspace_ptr = 0

        # apply gate
        cuquantum.custatevec.apply_matrix(
            handle,
            sv.data.ptr,  # type: ignore
            cuda_d_type,
            qubit_count,
            mat_ptr,
            cuda_d_type,
            cuquantum.custatevec.MatrixLayout.ROW,
            0,
            targets_ptr,
            len_targets,
            controls_ptr,
            0,
            len_controls,
            cuda_c_type,
            workspace_ptr,
            workspace_size,
        )


def evaluate_state_to_vector(
    state: Union[CircuitQuantumState, QuantumStateVector],
    precision: Precision = "complex128",
) -> QuantumStateVector:
    if cp is None:
        raise RuntimeError("CuPy is not installed.")
    if cuquantum is None:
        raise RuntimeError("cuQuantum is not installed.")

    if precision not in PRECISIONS:
        raise ValueError(f"Invalid precision: {precision}")

    if precision == "complex64":
        cuda_d_type = cuquantum.cudaDataType.CUDA_C_32F
        cuda_c_type = cuquantum.ComputeType.COMPUTE_32F
    else:
        cuda_d_type = cuquantum.cudaDataType.CUDA_C_64F
        cuda_c_type = cuquantum.ComputeType.COMPUTE_64F

    qubit_count = state.qubit_count

    if isinstance(state, QuantumStateVector):
        sv = cp.array(state.vector, dtype=precision)
    else:
        sv = cp.zeros(2**qubit_count, dtype=precision)
        sv[0] = 1.0

    circuit = state.circuit

    transpiler = ParallelDecomposer(
        [PauliDecomposeTranspiler(), PauliRotationDecomposeTranspiler()]
    )
    circuit = transpiler(circuit)

    handle = cuquantum.custatevec.create()

    _update_statevector(
        circuit=circuit,
        sv=sv,
        precision=precision,
        qubit_count=qubit_count,
        handle=handle,
        cuda_d_type=cuda_d_type,
        cuda_c_type=cuda_c_type,
    )

    cuquantum.custatevec.destroy(handle)

    return QuantumStateVector(qubit_count, vector=cp.asnumpy(sv))

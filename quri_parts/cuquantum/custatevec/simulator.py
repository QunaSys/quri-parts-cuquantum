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

from quri_parts.core.state import CircuitQuantumState, QuantumStateVector

from . import PRECISIONS, Precision
from .circuit import gate_array, gate_map

gates_to_cache = set()
for gate_name in gate_map.keys():
    gates_to_cache.add(gate_name)


def evaluate_state_to_vector(
    state: Union[CircuitQuantumState, QuantumStateVector]
) -> QuantumStateVector:
    if cp is None:
        raise RuntimeError("CuPy is not installed.")
    if cuquantum is None:
        raise RuntimeError("cuQuantum is not installed.")

    qubit_count = state.qubit_count

    if isinstance(state, QuantumStateVector):
        sv = np.array(state.vector, dtype=np.complex64)
    else:
        sv = np.zeros(2**qubit_count, dtype=np.complex64)
        sv[0] = 1.0

    sv = cp.array(sv)

    mat_dict = {}

    handle = cuquantum.custatevec.create()
    for g in state.circuit.gates:
        targets = np.array(g.target_indices, dtype=np.int32)
        controls = np.array(g.control_indices, dtype=np.int32)
        if g.name in gates_to_cache:
            if g.name not in mat_dict:
                mat = cp.array(gate_array(g), dtype=precision)
                mat_dict[g.name] = mat
            else:
                mat = mat_dict[g.name]
        else:
            mat = cp.array(gate_array(g), dtype=precision)
        mat_ptr = mat.data.ptr

        workspaceSize = cuquantum.custatevec.apply_matrix_get_workspace_size(
            handle,
            cuquantum.cudaDataType.CUDA_C_32F,
            qubit_count,
            mat_ptr,
            cuquantum.cudaDataType.CUDA_C_32F,
            cuquantum.custatevec.MatrixLayout.ROW,
            0,
            len(targets),
            len(controls),
            cuquantum.ComputeType.COMPUTE_32F,
        )

        # check the size of external workspace
        if workspaceSize > 0:
            workspace = cp.cuda.memory.alloc(workspaceSize)
            workspace_ptr = workspace.ptr
        else:
            workspace_ptr = 0

        # apply gate
        cuquantum.custatevec.apply_matrix(
            handle,
            sv.data.ptr,  # type: ignore
            cuquantum.cudaDataType.CUDA_C_32F,
            qubit_count,
            mat_ptr,
            cuquantum.cudaDataType.CUDA_C_32F,
            cuquantum.custatevec.MatrixLayout.ROW,
            0,
            targets.ctypes.data,
            len(targets),
            controls.ctypes.data,
            0,
            len(controls),
            cuquantum.ComputeType.COMPUTE_32F,
            workspace_ptr,
            workspaceSize,
        )

    cuquantum.custatevec.destroy(handle)

    return QuantumStateVector(qubit_count, vector=cp.asnumpy(sv))

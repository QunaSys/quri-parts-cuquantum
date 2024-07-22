# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import NamedTuple, Union

from typing_extensions import TypeAlias

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
from quri_parts.core.estimator import (
    ConcurrentParametricQuantumEstimator,
    Estimatable,
    Estimate,
    GeneralQuantumEstimator,
    ParametricQuantumEstimator,
    QuantumEstimator,
    create_concurrent_estimator_from_estimator,
    create_concurrent_parametric_estimator_from_concurrent_estimator,
    create_parametric_estimator_from_concurrent_estimator,
)
from quri_parts.core.state import (
    CircuitQuantumState,
    ParametricCircuitQuantumState,
    ParametricQuantumStateVector,
    QuantumStateVector,
)

from . import PRECISIONS, Precision
from .circuit import gate_array
from .operator import convert_operator


class _Estimate(NamedTuple):
    value: complex
    error: float = 0.0


#: A type alias for state classes supported by cuQuantum estimators.
#: cuQuantum estimators support both of circuit states and state vectors.
CuQuantumStateT: TypeAlias = Union[CircuitQuantumState, QuantumStateVector]
#: A type alias for parametric state classes supported by cuQuantum estimators.
#: cuQUantum estimators support both of circuit states and state vectors.
CuQuantumParametricStateT: TypeAlias = Union[
    ParametricCircuitQuantumState, ParametricQuantumStateVector
]


def _estimate(
    op: Estimatable,
    state: CuQuantumStateT,
    precision: Precision = "complex128",
) -> Estimate[complex]:
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
    paulis, pauli_ids, coefs = convert_operator(op)
    exp_values = np.empty(len(paulis), dtype=np.float64)

    if isinstance(state, QuantumStateVector):
        sv = np.array(state.vector, dtype=precision)
    else:
        sv = np.zeros(2**qubit_count, dtype=precision)
        sv[0] = 1.0

    sv = cp.array(sv)

    transpiler = ParallelDecomposer(
        [PauliDecomposeTranspiler(), PauliRotationDecomposeTranspiler()]
    )
    circuit = transpiler(state.circuit)

    handle = cuquantum.custatevec.create()
    for g in circuit.gates:
        targets = np.array(g.target_indices, dtype=np.int32)
        controls = np.array(g.control_indices, dtype=np.int32)
        mat = cp.array(gate_array(g, dtype=precision))
        mat_ptr = mat.data.ptr

        workspace_size = cuquantum.custatevec.apply_matrix_get_workspace_size(
            handle,
            cuda_d_type,
            qubit_count,
            mat_ptr,
            cuda_d_type,
            cuquantum.custatevec.MatrixLayout.ROW,
            0,
            len(targets),
            len(controls),
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
            targets.ctypes.data,
            len(targets),
            controls.ctypes.data,
            0,
            len(controls),
            cuda_c_type,
            workspace_ptr,
            workspace_size,
        )

    # apply Pauli operator
    cuquantum.custatevec.compute_expectations_on_pauli_basis(
        handle,
        sv.data.ptr,  # type: ignore
        cuda_d_type,
        qubit_count,
        exp_values.ctypes.data,
        paulis,
        len(paulis),
        pauli_ids,
        [len(arr) for arr in pauli_ids],
    )

    # destroy handle
    cuquantum.custatevec.destroy(handle)

    return _Estimate(value=np.sum(exp_values * coefs))


def create_cuquantum_vector_estimator(
    precision: Precision = "complex128",
) -> QuantumEstimator[Union[CircuitQuantumState, QuantumStateVector]]:
    return partial(_estimate, precision=precision)


def create_cuquantum_general_vector_estimator(
    precision: Precision = "complex128",
) -> GeneralQuantumEstimator[CuQuantumStateT, CuQuantumParametricStateT,]:
    estimator = create_cuquantum_vector_estimator(precision)
    concurrent_estimator = create_concurrent_estimator_from_estimator(estimator)
    parametric_estimator: ParametricQuantumEstimator[
        CuQuantumParametricStateT
    ] = create_parametric_estimator_from_concurrent_estimator(concurrent_estimator)
    concurrent_parametric_estimator: ConcurrentParametricQuantumEstimator[
        CuQuantumParametricStateT
    ] = create_concurrent_parametric_estimator_from_concurrent_estimator(
        concurrent_estimator
    )
    return GeneralQuantumEstimator(
        estimator,
        concurrent_estimator,
        parametric_estimator,
        concurrent_parametric_estimator,
    )

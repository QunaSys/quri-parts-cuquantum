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
from typing import NamedTuple, Union, Optional

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
    CircuitTranspiler,
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
from quri_parts.circuit import QuantumCircuit

from . import PRECISIONS, Precision
from .circuit import gate_array
from .operator import convert_operator
from .sampler import _update_state


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
    n_global_qubits: int = 0,
    device_network_type = None,
    transpiler: Optional[CircuitTranspiler] = None,
) -> Estimate[complex]:
    if device_network_type is None:
        device_network_type = cuquantum.bindings.custatevec.DeviceNetworkType.SWITCH
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
    gpu_count = 2 ** n_global_qubits
    n_local_qubits = qubit_count - n_global_qubits
    local_qubits = set(range(n_local_qubits))

    svs = []
    if isinstance(state, QuantumStateVector):
        assert len(state.vector) == 2 ** qubit_count
        local_vec_len = 2 ** n_local_qubits
        for i in range(gpu_count):
            with cp.cuda.Device(i):
                svs.append(cp.array(state.vector[i * local_vec_len:(i + 1) * local_vec_len], dtype=precision))
    else:
        for i in range(gpu_count):
            with cp.cuda.Device(i):
                svs.append(cp.zeros(2**n_local_qubits, dtype=precision))
        cp.put(svs[0], [0], [1.0])

    handles = []
    for i in range(gpu_count):
        with cp.cuda.Device(i):
            handles.append(cuquantum.bindings.custatevec.create())

    phys_to_virt_map = list(range(qubit_count))

    # Remove trailing swap gates for efficiency
    circuit = state.circuit

    swap_buf = _update_state(svs, handles, circuit, n_global_qubits, transpiler, precision, device_network_type)

    for i, j in reversed(swap_buf):
        phys_to_virt_map[i], phys_to_virt_map[j] = phys_to_virt_map[j], phys_to_virt_map[i]

    paulis, pauli_ids, coefs = convert_operator(op)
    exp_values = np.empty(len(paulis) * gpu_count, dtype=np.float64)
    phys_map = [0 for _ in range(qubit_count)]
    for i, j in enumerate(phys_to_virt_map):
        phys_map[j] = i

    for pauli_idx, (pauli_list, idx_list) in enumerate(zip(paulis, pauli_ids)):
        phys_idx_list = [phys_map[l] for l in idx_list]
        phys_idx_set = set(phys_idx_list)
        swap_buf = []
        i = 0
        j = n_local_qubits
        while True:
            while i in phys_idx_set:
                i += 1
            while j < qubit_count and j not in phys_idx_list:
                j += 1
            if j < qubit_count:
                if i < n_local_qubits:
                    swap_buf.append((i, j))
                    i += 1
                    j += 1
                else:
                    raise RuntimeError("Cannot process op term")
            else:
                break
        if swap_buf:
            for i in range(gpu_count):
                cp.cuda.Device(i).synchronize()
            with cp.cuda.Device(0):
                cuquantum.bindings.custatevec.multi_device_swap_index_bits(
                    handles,
                    gpu_count,
                    [sv.data.ptr for sv in svs],
                    cuda_d_type,
                    n_global_qubits, n_local_qubits,
                    swap_buf, len(swap_buf), [], [], 0,
                    device_network_type,
                )
        phys_to_virt_map = [0 for _ in range(qubit_count)]
        for i, j in enumerate(phys_map):
            phys_to_virt_map[j] = i
        for i, j in swap_buf:
            phys_to_virt_map[i], phys_to_virt_map[j] = phys_to_virt_map[j], phys_to_virt_map[i]
        phys_map = [0 for _ in range(qubit_count)]
        for i, j in enumerate(phys_to_virt_map):
            phys_map[j] = i

        # apply Pauli operator
        for i in range(gpu_count):
            with cp.cuda.Device(i):
                cuquantum.bindings.custatevec.compute_expectations_on_pauli_basis(
                    handles[i],
                    svs[i].data.ptr,  # type: ignore
                    cuda_d_type,
                    n_local_qubits,
                    exp_values.ctypes.data + exp_values.dtype.itemsize * (pauli_idx * gpu_count + i),
                    [pauli_list],
                    1,
                    [[phys_map[j] for j in idx_list]],
                    [len(idx_list)],
                )

    # destroy handle
    for i in range(gpu_count):
        with cp.cuda.Device(i) as dev:
            dev.synchronize()
            cuquantum.bindings.custatevec.destroy(handles[i])

    exp_value = 0.0
    for i in range(len(paulis)):
        for j in range(i * gpu_count, (i + 1) * gpu_count):
            # exp_value += exp_values[j] * sv_norms[j] * coefs[i]
            exp_value += exp_values[j] * coefs[i]

    return _Estimate(value=exp_value)


def create_cuquantum_vector_estimator(
    precision: Precision = "complex128",
    n_global_qubits: int = 0,
    device_network_type = None,
    transpiler: Optional[CircuitTranspiler] = None,
) -> QuantumEstimator[Union[CircuitQuantumState, QuantumStateVector]]:
    return partial(
        _estimate,
        precision=precision,
        n_global_qubits=n_global_qubits,
        device_network_type=device_network_type,
        transpiler=transpiler
    )


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

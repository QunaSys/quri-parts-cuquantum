# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import Counter
from functools import partial
from typing import Iterable, Optional, Any, Tuple
from .transpiler import MultiGPUReorderingTranspiler
import warnings
import math
import time
import random

try:
    import cupy as cp
    from cupy import ndarray
except ImportError:
    cp = None
    ndarray = None
try:
    import cuquantum
except ImportError:
    cuquantum = None

import numpy as np
from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.circuit.transpile import (
    CircuitTranspiler,
    ParallelDecomposer,
    PauliDecomposeTranspiler,
    PauliRotationDecomposeTranspiler,
    SequentialTranspiler,
)
from quri_parts.core.sampling import ConcurrentSampler, MeasurementCounts, Sampler

from .circuit import gate_array
from . import Precision, PRECISIONS


def _swap_bits(b: int, swap_buf: list[Tuple[int, int]]) -> int:
    for i, j in swap_buf:
        assert i != j
        pi = 2**i
        pj = 2**j
        ii = (b & pi) > 0
        jj = (b & pj) > 0
        if ii and not jj:
            b -= pi - pj
        elif jj and not ii:
            b -= pj - pi
    return b


def _update_state(
    svs: list[Any],
    handles: list[Any],
    circuit: NonParametricQuantumCircuit,
    n_global_qubits: int = 0,
    transpiler: Optional[CircuitTranspiler] = None,
    precision: Precision = "complex128",
    device_network_type: Any = None
) -> list[Tuple[int, int]]:
    if device_network_type is None:
        device_network_type = cuquantum.bindings.custatevec.DeviceNetworkType.SWITCH
    qubit_count = circuit.qubit_count
    gpu_count = 2 ** n_global_qubits
    n_local_qubits = qubit_count - n_global_qubits
    local_qubits = set(range(n_local_qubits))

    if cp is None:
        raise RuntimeError("CuPy is not installed.")
    if cuquantum is None:
        raise RuntimeError("cuQuantum is not installed.")

    if precision == "complex64":
        cuda_d_type = cuquantum.cudaDataType.CUDA_C_32F
        cuda_c_type = cuquantum.ComputeType.COMPUTE_32F
    else:
        cuda_d_type = cuquantum.cudaDataType.CUDA_C_64F
        cuda_c_type = cuquantum.ComputeType.COMPUTE_64F

    for i in range(gpu_count):
        with cp.cuda.Device(i):
            for j in range(gpu_count):
                if i == j: continue
                if cp.cuda.runtime.deviceCanAccessPeer(i, j) != 1:
                    raise RuntimeError(f"P2P access between device id {i} and {j} is unsupported")
                try:
                    cp.cuda.runtime.deviceEnablePeerAccess(j)
                except:
                    # Already enabled?
                    pass

    if transpiler is None:
        transpiler = SequentialTranspiler([
            ParallelDecomposer([PauliDecomposeTranspiler(), PauliRotationDecomposeTranspiler()]),
            MultiGPUReorderingTranspiler(n_local_qubits)
        ])

    circuit = transpiler(circuit)
    cached_mat_list = []
    for i in range(gpu_count):
        with cp.cuda.Device(i):
            cached_mat_list.append(cp.zeros(4 * 4, dtype=precision))

    swap_buf = []
    for g in circuit.gates:
        all_qubits = set(list(g.target_indices) + list(g.control_indices))

        if all_qubits.issubset(local_qubits):
            # apply global swap gates
            if swap_buf:
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
                swap_buf = []

            targets = np.array(g.target_indices, dtype=np.int32)
            controls = np.array(g.control_indices, dtype=np.int32)
            mat = gate_array(g)

            if mat is None:
                raise RuntimeError("Measurement is not supported")

            if mat.shape[0] > cached_mat_list[0].shape[0]:
                for i in range(gpu_count):
                    with cp.cuda.Device(i):
                        cached_mat_list[i] = cp.array(mat, dtype=precision)
            else:
                for i in range(gpu_count):
                    with cp.cuda.Device(i):
                        cp.put(cached_mat_list[i], range(mat.shape[0]), mat)

            # check the size of external workspace
            workspace_sizes = []
            for i in range(gpu_count):
                with cp.cuda.Device(i):
                    workspace_sizes.append(cuquantum.bindings.custatevec.apply_matrix_get_workspace_size(
                        handles[i],
                        cuda_d_type,
                        n_local_qubits,
                        cached_mat_list[i].data.ptr,
                        cuda_d_type,
                        cuquantum.bindings.custatevec.MatrixLayout.ROW,
                        0,
                        len(targets),
                        len(controls),
                        cuda_c_type,
                    ))

            # allocate external workspace
            workspaces = []
            for i, workspace_size in enumerate(workspace_sizes):
                if workspace_size > 0:
                    with cp.cuda.Device(i):
                        workspaces.append(cp.cuda.memory.alloc(workspace_size))
                else:
                    workspaces.append(None)

            # apply gate
            for i in range(gpu_count):
                with cp.cuda.Device(i):
                    cuquantum.bindings.custatevec.apply_matrix(
                        handles[i],
                        svs[i].data.ptr,  # type: ignore
                        cuda_d_type,
                        n_local_qubits,
                        cached_mat_list[i].data.ptr,
                        cuda_d_type,
                        cuquantum.bindings.custatevec.MatrixLayout.ROW,
                        0,
                        targets.ctypes.data,
                        len(targets),
                        controls.ctypes.data,
                        0,
                        len(controls),
                        cuda_c_type,
                        0 if workspaces[i] is None else workspaces[i].ptr,
                        workspace_sizes[i],
                    )
        else:
            if g.name != "SWAP":
                raise RuntimeError(f"Cannot apply gate {g.name} between qubits {all_qubits}")
            swap_buf.append(tuple(g.target_indices))

    # if swap_buf:
    #     with cp.cuda.Device(0):
    #         cuquantum.bindings.custatevec.multi_device_swap_index_bits(
    #             handles,
    #             gpu_count,
    #             [sv.data.ptr for sv in svs],
    #             cuda_d_type,
    #             n_global_qubits, n_local_qubits,
    #             swap_buf, len(swap_buf), [], [], 0,
    #             device_network_type,
    #         )

    return swap_buf


def _sample(
    circuit: NonParametricQuantumCircuit,
    shots: int,
    precision: Precision = "complex128",
    seed: int = 0,
    n_global_qubits: int = 0,
    device_network_type = None,
    transpiler: Optional[CircuitTranspiler] = None,
) -> MeasurementCounts:
    if cp is None:
        raise RuntimeError("CuPy is not installed.")
    if cuquantum is None:
        raise RuntimeError("cuQuantum is not installed.")
    if device_network_type is None:
        device_network_type = cuquantum.bindings.custatevec.DeviceNetworkType.SWITCH

    if precision not in PRECISIONS:
        raise ValueError(f"Invalid precision: {precision}")

    if precision == "complex64":
        cuda_d_type = cuquantum.cudaDataType.CUDA_C_32F
        cuda_c_type = cuquantum.ComputeType.COMPUTE_32F
    else:
        cuda_d_type = cuquantum.cudaDataType.CUDA_C_64F
        cuda_c_type = cuquantum.ComputeType.COMPUTE_64F

    qubit_count = circuit.qubit_count
    assert qubit_count >= n_global_qubits
    gpu_count = 2 ** n_global_qubits
    existing_gpu_count = cp.cuda.runtime.getDeviceCount()

    if gpu_count > existing_gpu_count:
        raise RuntimeError(
            f"Cannot proceed quantum simulation with {n_global_qubits=}.\n" +
            "Required GPUs: {gpu_count}, Existing GPUs: {existing_gpu_count}"
        )

    n_local_qubits = qubit_count - n_global_qubits

    handles = []
    for i in range(gpu_count):
        with cp.cuda.Device(i):
            handles.append(cuquantum.bindings.custatevec.create())

    svs = []
    for i in range(gpu_count):
        with cp.cuda.Device(i):
            svs.append(cp.zeros(2**n_local_qubits, dtype=precision))
    cp.put(svs[0], [0], [1.0])

    swap_buf = _update_state(svs, handles, circuit, n_global_qubits, transpiler, precision, device_network_type)

    for i in range(gpu_count):
        cp.cuda.Device(i).synchronize()

    res_bits = np.empty(shots, dtype=np.int64)
    bit_ordering = np.arange(n_local_qubits, dtype=np.int32)

    rng = np.random.default_rng(seed)
    rand_nums = rng.random(shots)

    # create sampler and check the size of external workspace
    samplers = []
    sampler_workspace_sizes = []
    for i in range(gpu_count):
        with cp.cuda.Device(i):
            sampler, sampler_workspace_size = cuquantum.bindings.custatevec.sampler_create(
                handles[i], svs[i].data.ptr, cuda_d_type, n_local_qubits, shots
            )
            samplers.append(sampler)
            sampler_workspace_sizes.append(sampler_workspace_size)

    # allocate external workspace
    sampler_workspaces = []
    for i in range(gpu_count):
        with cp.cuda.Device(i):
            sampler_workspaces.append(cp.cuda.alloc(sampler_workspace_sizes[i]))

    # sample preprocess
    for i in range(gpu_count):
        with cp.cuda.Device(i):
            cuquantum.bindings.custatevec.sampler_preprocess(
                handles[i], samplers[i], sampler_workspaces[i].ptr, sampler_workspace_sizes[i]
            )

    # calculate norms
    if gpu_count > 1:
        sv_norms = []
        for i in range(gpu_count):
            with cp.cuda.Device(i) as dev:
                sv_norms.append(cuquantum.bindings.custatevec.sampler_get_squared_norm(
                    handles[i], samplers[i]
                ))
                dev.synchronize()
        norm = sum(sv_norms)
    else:
        sv_norms = [1.0]
        norm = 1.0

    # # apply offsets and norm
    # for i in range(gpu_count):
    #     if sv_norms[i] > 0.0:
    #         with cp.cuda.Device(i):
    #             cuquantum.bindings.custatevec.sampler_apply_sub_sv_offset(
    #                 handles[i],
    #                 samplers[i],
    #                 i,
    #                 gpu_count,
    #                 sum(sv_norms[:i]),
    #                 norm
    #             )

    # distribute shots into sub vectors
    shot_svs = random.choices(list(range(gpu_count)), k=shots, weights=sv_norms)
    shot_counts = [sum(1 for sv in shot_svs if sv == i) for i in range(gpu_count)]

    # sample bit strings
    for i in range(gpu_count):
        if shot_counts[i] == 0:
            continue
        shot_offset = sum(shot_counts[:i])
        with cp.cuda.Device(i):
            cuquantum.bindings.custatevec.sampler_sample(
                handles[i],
                samplers[i],
                res_bits.ctypes.data + shot_offset * res_bits.dtype.itemsize,
                bit_ordering.ctypes.data,
                n_local_qubits,
                rand_nums.ctypes.data + shot_offset * rand_nums.dtype.itemsize,
                shot_counts[i],
                cuquantum.bindings.custatevec.SamplerOutput.RANDNUM_ORDER,
            )

    for i in range(gpu_count):
        cp.cuda.Device(i).synchronize()

    result = Counter()
    global_qubit_shift = 2 ** n_local_qubits
    shot_offset = 0
    for i in range(gpu_count):
        global_qubits = i * global_qubit_shift
        for _ in range(shot_counts[i]):
            b = res_bits[shot_offset] | global_qubits
            result[_swap_bits(b, swap_buf)] += 1
            shot_offset += 1

    # destroy sampler and handle
    for i in range(gpu_count):
        with cp.cuda.Device(i):
            cuquantum.bindings.custatevec.sampler_destroy(samplers[i])
            cuquantum.bindings.custatevec.destroy(handles[i])

    return result


def create_cuquantum_vector_sampler(
        precision: Precision = "complex128",
        seed: int = 0,
        n_global_qubits: int = 0,
        device_network_type = None,
        transpiler: Optional[CircuitTranspiler] = None,
) -> Sampler:
    return partial(
        _sample,
        precision=precision,
        seed=seed,
        n_global_qubits=n_global_qubits,
        device_network_type=device_network_type,
        transpiler=transpiler,
    )


def create_cuquantum_vector_concurrent_sampler(
    precision: Precision = "complex128", seed: int = 0, n_global_qubits: int = 0,
) -> ConcurrentSampler:
    sampler = create_cuquantum_vector_sampler(precision, seed, n_global_qubits)

    def _concurrent_sampling(
        circuit_shots_tuples: Iterable[tuple[NonParametricQuantumCircuit, int]]
    ) -> Iterable[MeasurementCounts]:
        return [sampler(circuit, shots) for circuit, shots in circuit_shots_tuples]

    return _concurrent_sampling

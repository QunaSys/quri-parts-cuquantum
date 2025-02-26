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
from typing import Iterable

try:
    import cupy as cp
except ImportError:
    cp = None
try:
    import cuquantum
except ImportError:
    cuquantum = None
import numpy as np
from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.circuit.transpile import (
    ParallelDecomposer,
    PauliDecomposeTranspiler,
    PauliRotationDecomposeTranspiler,
)
from quri_parts.core.sampling import ConcurrentSampler, MeasurementCounts, Sampler

from . import PRECISIONS, Precision

from .simulator import _update_statevector


def _sample(
    circuit: NonParametricQuantumCircuit,
    shots: int,
    precision: Precision = "complex128",
    seed: int = 0,
) -> MeasurementCounts:
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

    qubit_count = circuit.qubit_count
    sv = cp.zeros(2**qubit_count, dtype=precision)
    sv[0] = 1.0
    res_bits = np.empty((shots,), dtype=np.int64)
    bit_ordering = np.arange(qubit_count, dtype=np.int32)

    rng = np.random.default_rng(seed)
    rand_nums = rng.random(shots)

    transpiler = ParallelDecomposer(
        [PauliDecomposeTranspiler(), PauliRotationDecomposeTranspiler()]
    )
    circuit = transpiler(circuit)

    handle = cuquantum.custatevec.create()

    _update_statevector(
        circuit=circuit,
        sv=sv,
        qubit_count=qubit_count,
        precision=precision,
        handle=handle,
        cuda_d_type=cuda_d_type,
        cuda_c_type=cuda_c_type,
    )

    # create sampler and check the size of external workspace
    sampler, extraworkspace_sizeInBytes = cuquantum.custatevec.sampler_create(
        handle, sv.data.ptr, cuda_d_type, qubit_count, shots
    )

    # allocate external workspace
    extraWorkspace = cp.cuda.alloc(extraworkspace_sizeInBytes)

    # sample preprocess
    cuquantum.custatevec.sampler_preprocess(
        handle, sampler, extraWorkspace.ptr, extraworkspace_sizeInBytes
    )

    # sample bit strings
    cuquantum.custatevec.sampler_sample(
        handle,
        sampler,
        res_bits.ctypes.data,
        bit_ordering.ctypes.data,
        qubit_count,
        rand_nums.ctypes.data,
        shots,
        cuquantum.custatevec.SamplerOutput.RANDNUM_ORDER,
    )

    # destroy sampler
    cuquantum.custatevec.sampler_destroy(sampler)

    # destroy handle
    cuquantum.custatevec.destroy(handle)

    return Counter(res_bits)


def create_cuquantum_vector_sampler(
    precision: Precision = "complex128", seed: int = 0
) -> Sampler:
    return partial(_sample, precision=precision, seed=seed)


def create_cuquantum_vector_concurrent_sampler(
    precision: Precision = "complex128", seed: int = 0
) -> ConcurrentSampler:
    sampler = create_cuquantum_vector_sampler(precision, seed)

    def _concurrent_sampling(
        circuit_shots_tuples: Iterable[tuple[NonParametricQuantumCircuit, int]]
    ) -> Iterable[MeasurementCounts]:
        return [sampler(circuit, shots) for circuit, shots in circuit_shots_tuples]

    return _concurrent_sampling

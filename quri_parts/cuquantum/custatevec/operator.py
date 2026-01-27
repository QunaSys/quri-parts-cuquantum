# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Iterable, Union
from typing_extensions import TypeAlias

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    from cuquantum import bindings.custatevec as cusv
except ImportError:
    cusv = None

from quri_parts.core.operator import PAULI_IDENTITY, Operator, PauliLabel

_OperatorKey: TypeAlias = frozenset[tuple[PauliLabel, complex]]
_operator_cache: dict[
    _OperatorKey, tuple[list[list[int]], list[list[int]], list[complex]]
] = {}


def convert_operator(
    operator: Union[Operator, PauliLabel]
) -> tuple[list[list[int]], list[list[int]], list[complex]]:
    """Converts an :class:`~Operator` (or :class:`~PauliLabel`) to the Sequence
    of the list of :class:`custatevec.Pauli`s along with their indices and it's
    coefficient."""

    if cp is None:
        raise RuntimeError("CuPy is not installed.")
    if cusv is None:
        raise RuntimeError("cuQuantum is not installed.")

    pauli_coef_tuples: Iterable[tuple[PauliLabel, complex]]
    op_key: _OperatorKey
    if isinstance(operator, PauliLabel):
        pauli_coef_tuples = [(operator, 1)]
    else:
        pauli_coef_tuples = operator.items()

    op_key = frozenset(pauli_coef_tuples)
    if op_key in _operator_cache:
        return _operator_cache[op_key]

    paulis: list[list[int]] = []
    pauli_idxs: list[list[int]] = []
    coefs: list[complex] = []
    for pauli, coef in pauli_coef_tuples:
        if pauli == PAULI_IDENTITY:
            paulis.append([0])
            pauli_idxs.append([0])
            coefs.append(coef)
            continue
        pauli_seq = []
        pauli_idx_seq = []
        for idx, p in pauli:
            pauli_seq.append(p)
            pauli_idx_seq.append(idx)
        paulis.append(pauli_seq)
        pauli_idxs.append(pauli_idx_seq)
        coefs.append(coef)
    _operator_cache[op_key] = (paulis, pauli_idxs, coefs)
    return (paulis, pauli_idxs, coefs)

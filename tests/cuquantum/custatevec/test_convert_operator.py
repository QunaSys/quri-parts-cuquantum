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
from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label

from quri_parts.cuquantum.custatevec.operator import convert_operator

P_LABELS = [
    pauli_label("X0"),
    pauli_label("Y1"),
    pauli_label("Z1 X4"),
    pauli_label("X4 Y2"),
]
COEFS = [(i + 1) * 0.1 for i in range(len(P_LABELS))]


def test_convert_operator() -> None:
    op = Operator()
    for p_label, coef in zip(P_LABELS, COEFS):
        op += Operator({p_label: coef})
    op_converted = convert_operator(op)

    for i in range(len(op_converted[0])):
        ps, p_ids = P_LABELS[i].index_and_pauli_id_list
        assert {p: idx for idx, p in zip(ps, p_ids)} == {
            p: idx for p, idx in zip(op_converted[0][i], op_converted[1][i])
        }

    assert np.allclose(op_converted[2], COEFS)

    # constant
    op = Operator()
    op.constant = 3.0
    converted = convert_operator(op)
    assert converted[0] == [[0]]
    assert converted[1] == [[0]]
    assert converted[2] == [3.0]


def test_convert_pauli_identity() -> None:
    converted = convert_operator(PAULI_IDENTITY)
    assert converted[0] == [[0]]
    assert converted[1] == [[0]]
    assert converted[2] == [1.0]

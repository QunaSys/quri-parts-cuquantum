from quri_parts.circuit.transpile import CircuitTranspilerProtocol
from quri_parts.circuit import ImmutableQuantumCircuit, QuantumGate, SWAP, QuantumCircuit
from typing import Tuple, Iterable
from itertools import chain


def _greedy_tiling(
    gates: Iterable[QuantumGate], n_qubits: int, n_local_qubits: int
) -> Tuple[set[int], list[QuantumGate], Iterable[QuantumGate]]:
    avail = set(range(n_qubits))
    local_qubits = set()
    tile = list()
    reminder = list()
    for gate in gates:
        targ = set(gate.control_indices) | set(gate.target_indices)
        if targ.issubset(avail) and len(local_qubits | targ) <= n_local_qubits:
            tile.append(gate)
            local_qubits.update(targ)
        else:
            reminder.append(gate)
            avail -= targ
    return local_qubits, tile, reminder


def _mapping(
    phys_map: list[int],
    next_virt_locals: set[int],
    n_local_qubits: int,
) -> Tuple[list[int], list[QuantumGate]]:
    n_qubits = len(phys_map)
    virt_locals = set(phys_map[:n_local_qubits])
    swap_gates = list()
    i, j = 0, n_local_qubits
    while True:
        while i < n_local_qubits and phys_map[i] in next_virt_locals:
            i += 1
        while j < n_qubits and phys_map[j] not in next_virt_locals:
            j += 1
        if j >= n_qubits:
            break
        if i >= n_local_qubits:
            raise RuntimeError("Cannot place virtual qubits in local qubits")
        swap_gates.append(SWAP(i, j))
        phys_map[i], phys_map[j] = phys_map[j], phys_map[i]
    return phys_map, swap_gates


def _map_qubits(virt_to_phys: list[int], gate: QuantumGate) -> QuantumGate:
    return QuantumGate(
        name=gate.name,
        target_indices=[virt_to_phys[q] for q in gate.target_indices],
        control_indices=[virt_to_phys[q] for q in gate.control_indices],
        classical_indices=gate.classical_indices,
        params=gate.params,
        pauli_ids=gate.pauli_ids,
        unitary_matrix=gate.unitary_matrix,
    )


def _make_swaps_from_map(phys_map: list[int]) -> list[Tuple[int, int]]:
    family: list[list[int]] = list()
    for i, j in enumerate(phys_map):
        grp_i, grp_j = None, None
        for grp in family:
            if i in grp:
                grp_i = grp
            if j in grp:
                grp_j = grp
        if grp_i is None and grp_j is None:
            family += [i < j and [i, j] or (i == j and [i] or [j, i])]
        elif grp_i is not None and grp_j is None:
            grp_i.append(j)
            grp_i.sort()
        elif grp_i is None and grp_j is not None:
            grp_j.append(i)
            grp_j.sort()
        elif grp_i is not None and grp_j is not None:
            if grp_i != grp_j:
                family.remove(grp_j)
                grp_i.extend(grp_j)
                grp_i.sort()
    return sum([
        [(grp[i - 1], grp[i]) for i in range(1, len(grp))]
        for grp in family
    ], [])


class MultiGPUReorderingTranspiler(CircuitTranspilerProtocol):
    """
    Reorder and insert SWAP gates, to be executed efficiently in multi-GPU environments.

    [^1]: Y. Teranishi et al., Lazy Qubit Reordering for Accelerating Parallel
         State-Vector-based Quantum Circuit Simulation, arXiv: 2410.04252

    Args:
        n_local_qubits: Number of qubits, in which multiple qubit gates can be applied
                        by native (or fast) operations.
                        For multi-GPU environments, it means that the first $n_{\\mathrm{local}}$
                        qubits are placed in one GPU memory. (consumes $2^{n_{\\mathrm{local}}}$
                        complex numbers in statevec form)
        n_internal_qubits: The number of qubits stored across multiple memories that therefore
                        require internal (i.e., high-speed) communication when applying multi-qubit
                        gates. For example, in a multi-GPU environment, this refers to the number
                        of qubits that are distributed across multiple GPUs within a single node.
    """

    _n_local_qubits: int
    _n_internal_qubits: int
    
    def __init__(self, n_local_qubits: int, n_internal_qubits: int = 0):
        self._n_local_qubits = n_local_qubits
        self._n_internal_qubits = n_internal_qubits

    def __call__(self, circuit: ImmutableQuantumCircuit) -> ImmutableQuantumCircuit:
        n_qubits = circuit.qubit_count
        phys_map = list(range(n_qubits))
        gates = list(circuit.gates)
        circuit = QuantumCircuit(n_qubits)
        i = 0
        while len(gates) > 0:
            virt_locals, tile, gates = _greedy_tiling(gates, n_qubits, self._n_local_qubits)
            if len(tile) == 0:
                raise RuntimeError("Cannot place gates in local qubits")
            gates = list(gates)
            phys_map, swap_gates = _mapping(phys_map, virt_locals, self._n_local_qubits)
            circuit.extend(swap_gates)
            virt_to_phys = [[j for j, e in enumerate(phys_map) if e == i][0] for i in range(n_qubits)]
            circuit.extend([_map_qubits(virt_to_phys, g) for g in tile])
        circuit.extend([SWAP(i, j) for i, j in _make_swaps_from_map(phys_map)])
        return circuit.freeze()

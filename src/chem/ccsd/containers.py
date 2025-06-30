from __future__ import annotations
from dataclasses import dataclass, field
from enum import StrEnum, auto
import numpy as np
from numpy.typing import NDArray

from chem.hf.intermediates_builders import Intermediates


@dataclass
class UHF_CCSD_Lambda_Data:
    l1_aa: NDArray
    l1_bb: NDArray
    l2_aaaa: NDArray
    l2_abab: NDArray
    l2_bbbb: NDArray

@dataclass
class UHF_CCSD_Data:
    t1_aa: NDArray
    t1_bb: NDArray
    t2_aaaa: NDArray
    t2_abab: NDArray
    t2_bbbb: NDArray
    t2_abba: NDArray
    t2_baab: NDArray
    t2_baba: NDArray

    lmbda: UHF_CCSD_Lambda_Data | None = None


class E1_spin(StrEnum):
    aa = auto()
    bb = auto()


class E2_spin(StrEnum):
    aaaa = auto()
    abab = auto()
    abba = auto()
    baab = auto()
    baba = auto()
    bbbb = auto()


@dataclass
class Spin_MBE():
    """ MBE stands for many body expansion. 
    TODO: find a better place for it. Uncouple it from the Intermediates.
    """
    singles: dict[E1_spin, NDArray] = field(default_factory=dict)
    doubles: dict[E2_spin, NDArray] = field(default_factory=dict)
    EQUAL_THRESHOLD: float = 1e-6

    def pretty_print_mbe(self) -> None:
        with np.printoptions(precision=3, suppress=True):

            print("Singles:")
            for key, value in self.singles.items():
                print(f' {key}'
                      f' shape = {value.shape}'
                      f' norm = {float(np.linalg.norm(value)):.3f}')
                print(value)

            print("Doubles:")
            for key, value in self.doubles.items():
                print(f' {key}'
                      f' shape = {value.shape}'
                      f' norm = {float(np.linalg.norm(value)):.3f}')
            print("")

    def pretty_print_doubles_block(self, block: E2_spin) -> None:
        if not block in self.doubles:
            raise ValueError(f"Block {block} is not part of this vector.")

        matrix = self.doubles[block]
        print(f' {block}'
              f' shape = {matrix.shape}'
              f' norm = {float(np.linalg.norm(matrix)):.3f}')

        with np.printoptions(precision=3, suppress=True):
            print(matrix)

    @staticmethod
    def find_dims_slices_shapes(uhf_scf_data: Intermediates) -> tuple:
        """ TODO: make it work alright """
        scf = uhf_scf_data
        nmo = scf.nmo
        noa = scf.noa
        nva = nmo - noa
        nob = scf.nob
        nvb = nmo - nob
        dims = {
            E1_spin.aa: nva * noa,
            E1_spin.bb: nvb * nob,
            E2_spin.aaaa: nva * nva * noa * noa,
            E2_spin.abab: nva * nvb * noa * nob,
            E2_spin.abba: nva * nvb * nob * noa,
            E2_spin.baab: nvb * nva * noa * nob,
            E2_spin.baba: nvb * nva * nob * noa,
            E2_spin.bbbb: nvb * nvb * nob * nob,
        }

        slices = dict()
        current_size = 0
        for block in E1_spin:
            block_dim = dims[block]
            slices[block] = slice(current_size, current_size + block_dim)
            current_size += block_dim

        for block in E2_spin:
            block_dim = dims[block]
            slices[block] = slice(current_size, current_size + block_dim)
            current_size += block_dim

        shapes = {
            E1_spin.aa: (nva, noa),
            E1_spin.bb: (nvb, nob),
            E2_spin.aaaa: (nva, nva, noa, noa),
            E2_spin.abab: (nva, nvb, noa, nob),
            E2_spin.abba: (nva, nvb, nob, noa),
            E2_spin.baab: (nvb, nva, noa, nob),
            E2_spin.baba: (nvb, nva, nob, noa),
            E2_spin.bbbb: (nvb, nvb, nob, nob),
        }
        return dims, slices, shapes


    @classmethod
    def from_flattened_NDArray(
        cls,
        vector: NDArray,
        uhf_scf_data: Intermediates,
    ) -> Spin_MBE:
        dims, slices, shapes = Spin_MBE.find_dims_slices_shapes(uhf_scf_data)
        assert len(vector.shape) == 1
        assert vector.shape[0] == Spin_MBE.get_vector_dim(dims)

        mbe = Spin_MBE()
        for spin_block in E1_spin:
            sub_vec = vector[slices[spin_block]]
            mbe.singles[spin_block] = sub_vec.reshape(shapes[spin_block])

        for spin_block in E2_spin:
            sub_vec = vector[slices[spin_block]]
            mbe.doubles[spin_block] = sub_vec.reshape(shapes[spin_block])

        return mbe

    def flatten(self) -> NDArray:
        return np.vstack(
            list(vec.reshape(-1, 1) for _, vec in self.singles.items())
            +
            list(vec.reshape(-1, 1) for _, vec in self.doubles.items())
        ).flatten()

    @staticmethod
    def get_singles_dim(dims: dict[str, int]) -> int:
        return sum(dims[block] for block in E1_spin)

    @staticmethod
    def get_doubles_dim(dims: dict[str, int]) -> int:
        return sum(dims[block] for block in E2_spin)

    @staticmethod
    def get_vector_dim(dims: dict[str, int]) -> int:
        singles_dim = Spin_MBE.get_singles_dim(dims)
        doubles_dim = Spin_MBE.get_doubles_dim(dims)
        return singles_dim + doubles_dim

    def __eq__(self, other) -> bool:
        if not isinstance(other, Spin_MBE):
            return False

        if self.singles.keys() != other.singles.keys():
            return False
        for key in self.singles.keys():
            if not np.allclose(
                self.singles[key],
                other.singles[key],
                atol=self.EQUAL_THRESHOLD,
            ):
                return False

        if self.doubles.keys() != other.doubles.keys():
            return False
        for key in self.doubles.keys():
            if not np.allclose(
                self.doubles[key],
                other.doubles[key],
                atol=self.EQUAL_THRESHOLD,
            ):
                return False

        return True

    def __neg__(self):
        negated = Spin_MBE()
        for key, value in self.singles.items():
            negated.singles[key] = -value
        for key, value in self.doubles.items():
            negated.doubles[key] = -value
        return negated

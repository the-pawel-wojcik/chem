from __future__ import annotations
from dataclasses import dataclass, field
from enum import StrEnum, auto

from numpy.typing import NDArray
import numpy as np
from typing import TypeVar, Protocol


_T_contra = TypeVar("_T_contra", contravariant=True)


class SupportsWrite(Protocol[_T_contra]):
    def write(self, s: _T_contra, /) -> object: ...


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
class UHF_ov_data:
    """ Unrestricted Hartee-Fock data about the occupied and virtual molecular
    orbitals. The total number of molecular orbitals `nmo`; the number of
    occupied up (alpha) `noa` and occupied down (beta) `nob` orbitals; the
    number of virtual up (alpha) `nva` and the number of virtual down (beta)
    `nvb` orbitals. """
    nmo: int
    noa: int
    nva: int
    nob: int
    nvb: int

    def get_dims(self) -> dict[E1_spin | E2_spin, int]:
        nva = self.nva
        noa = self.noa
        nvb = self.nvb
        nob = self.nob
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
        return dims

    def get_shapes(self) -> dict[E1_spin | E2_spin, tuple[int, ...]]:
        nva = self.nva
        noa = self.noa
        nvb = self.nvb
        nob = self.nob
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

        return shapes

    def get_slices(self) -> dict[E1_spin | E2_spin, slice]:
        dims = self.get_dims()
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

        return slices

    def get_singles_dim(self) -> int:
        dims = self.get_dims()
        return sum(dims[block] for block in E1_spin)

    def get_doubles_dim(self) -> int:
        dims = self.get_dims()
        return sum(dims[block] for block in E2_spin)

    def get_vector_dim(self) -> int:
        """ Dimension of a flattened Spin_MBE vector, i.e. sum of the full
        dimensions of the aa block, bb blocl, aaaa block, ... """
        singles_dim = self.get_singles_dim()
        doubles_dim = self.get_doubles_dim()
        return singles_dim + doubles_dim


@dataclass
class Spin_MBE():
    """ MBE stands for many body expansion. """
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

    def pretty_print_doubles_block(
        self,
        block: E2_spin,
        file: SupportsWrite | None = None,
    ) -> None:
        if not block in self.doubles:
            raise ValueError(f"Block {block} is not part of this vector.")

        matrix = self.doubles[block]
        print(f' {block}'
              f' shape = {matrix.shape}'
              f' norm = {float(np.linalg.norm(matrix)):.3f}', file=file)

        with np.printoptions(precision=3, suppress=True):
            print(matrix, file=file)

    def flatten(self) -> NDArray:
        return np.vstack(
            list(vec.reshape(-1, 1) for _, vec in self.singles.items())
            +
            list(vec.reshape(-1, 1) for _, vec in self.doubles.items())
        ).flatten()


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
    
    def __matmul__(self, other):
        if not isinstance(other, Spin_MBE):
            raise ValueError(f"Unsupported operation {self.__class__.__name__}"
                             f" @ {other.__class__.__name__}.")

        return sum(
            [
                np.einsum(
                    'ai,ai->',
                    self.singles[spin_block],
                    other.singles[spin_block],
                ) for spin_block in E1_spin
            ]
            + 
            [
                np.einsum(
                    'abji,abji->',
                    self.doubles[spin_block],
                    other.doubles[spin_block],
                ) for spin_block in E2_spin
            ]
       )

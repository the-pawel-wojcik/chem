from __future__ import annotations
from dataclasses import dataclass

from numpy.typing import NDArray
import numpy as np
from typing import TypeVar, Protocol


@dataclass
class GHF_ov_data:
    """ Generalized Hartee-Fock data about the occupied and virtual molecular
    orbitals. The total number of molecular orbitals `nmo`; the number of
    occupied `no` and virtual `nv` orbitals. """
    nmo: int
    no: int
    nv: int

    def get_dims(self) -> dict[str, int]:
        nv = self.nv
        no = self.no
        dims = {
            'ref': 1,
            'singles': nv * no,
            'doubles': nv * nv * no * no,
        }
        return dims

    def get_shapes(self) -> dict[str, tuple[int, ...]]:
        no = self.no
        nv = self.nv
        shapes = {
            'ref': (1),
            'singles': (nv, no),
            'doubles': (nv, nv, no, no),
        }

        return shapes

    def get_slices(self) -> dict[str, slice]:
        dims = self.get_dims()
        slices = dict()
        current_size = 0
        for block in ['ref', 'singles', 'doubles']:
            block_dim = dims[block]
            slices[block] = slice(current_size, current_size + block_dim)
            current_size += block_dim

        return slices

    def get_vector_dim(self) -> int:
        """ Dimension of a flattened GHF_CCSD_MBE vector, i.e. sum of the full
        dimensions of the singles and doubles blocks. """
        dims = self.get_dims()
        return dims['ref'] + dims['singles'] + dims['doubles']


_T_contra = TypeVar("_T_contra", contravariant=True)


class SupportsWrite(Protocol[_T_contra]):
    def write(self, s: _T_contra, /) -> object: ...


@dataclass
class GHF_CCSD_MBE:
    """ MBE stands for many body expansion. """
    ref: NDArray
    singles: NDArray
    doubles: NDArray
    EQUAL_THRESHOLD: float = 1e-6

    @classmethod
    def from_NDArray(
        cls,
        vector: NDArray,
        ghf_ov_data: GHF_ov_data,
    ) -> GHF_CCSD_MBE:
        assert len(vector.shape) == 1
        assert vector.shape[0] == ghf_ov_data.get_vector_dim()

        slices = ghf_ov_data.get_slices()
        shapes = ghf_ov_data.get_shapes()

        ref = vector[slices['ref']]
        ref = ref.reshape(shapes['ref'])

        singles = vector[slices['singles']]
        singles = singles.reshape(shapes['singles'])

        doubles = vector[slices['doubles']]
        doubles = doubles.reshape(shapes['doubles'])

        return cls(ref=ref, singles=singles, doubles=doubles)

    def pretty_print_mbe(
        self,
        verbose_ref: bool = False,
        verbose_singles: bool = False,
        verbose_doubles: bool = False,
        file: SupportsWrite | None = None,
    ) -> None:

        ref = self.ref
        print(f'Reference:'
              f' shape = {ref.shape}'
              f' norm = {float(np.linalg.norm(ref)):.3f}',
              file=file,)

        if verbose_ref:
            with np.printoptions(precision=3, suppress=True):
                print(ref, file=file)

        singles = self.singles
        print(f'Singles:'
              f' shape = {singles.shape}'
              f' norm = {float(np.linalg.norm(singles)):.3f}',
              file=file,)

        if verbose_singles:
            with np.printoptions(precision=3, suppress=True):
                print(singles, file=file)

        doubles = self.doubles
        print(f'Doubles:'
              f' shape = {doubles.shape}'
              f' norm = {float(np.linalg.norm(doubles)):.3f}',
              file=file,)

        if verbose_doubles:
            with np.printoptions(precision=3, suppress=True):
                print(doubles, file=file)

    def flatten(self) -> NDArray:
        return np.vstack(
            (
                self.ref.reshape(-1, 1),
                self.singles.reshape(-1, 1),
                self.doubles.reshape(-1, 1),
            )
        ).flatten()

    def __eq__(self, other) -> bool:
        if not isinstance(other, GHF_CCSD_MBE):
            return False

        if not np.allclose(self.ref, other.ref, atol=self.EQUAL_THRESHOLD):
            return False

        if not np.allclose(
            self.singles,
            other.singles,
            atol=self.EQUAL_THRESHOLD,
        ):
            return False

        if not np.allclose(
            self.doubles,
            other.doubles,
            atol=self.EQUAL_THRESHOLD,
        ):
            return False

        return True

    def __neg__(self):
        negated = GHF_CCSD_MBE(-self.ref, -self.singles, -self.doubles)
        return negated
    
    def __matmul__(self, other: GHF_CCSD_MBE) -> float:
        if not isinstance(other, GHF_CCSD_MBE):
            raise ValueError(f"Unsupported operation {self.__class__.__name__}"
                             f" @ {other.__class__.__name__}.")

        return float(
            np.sum(self.ref * self.ref)
            +
            np.einsum('ai,ai->', self.singles, other.singles)
            + 
            np.einsum('abji,abji->', self.doubles, other.doubles)
       )

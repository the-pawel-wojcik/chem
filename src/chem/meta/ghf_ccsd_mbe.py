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
            'singles': nv * no,
            'doubles': nv * nv * no * no,
        }
        return dims

    def get_shapes(self) -> dict[str, tuple[int, ...]]:
        no = self.no
        nv = self.nv
        shapes = {
            'singles': (nv, no),
            'doubles': (nv, nv, no, no),
        }

        return shapes

    def get_slices(self) -> dict[str, slice]:
        dims = self.get_dims()
        slices = dict()
        current_size = 0
        for block in ['singles', 'doubles']:
            block_dim = dims[block]
            slices[block] = slice(current_size, current_size + block_dim)
            current_size += block_dim

        return slices

    def get_vector_dim(self) -> int:
        """ Dimension of a flattened GHF_CCSD_MBE vector, i.e. sum of the full
        dimensions of the singles and doubles blocks. """
        dims = self.get_dims()
        return dims['singles'] + dims['doubles']


_T_contra = TypeVar("_T_contra", contravariant=True)


class SupportsWrite(Protocol[_T_contra]):
    def write(self, s: _T_contra, /) -> object: ...


@dataclass
class GHF_CCSD_MBE:
    """ MBE stands for many body expansion. """
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

        singles = vector[slices['singles']]
        singles = singles.reshape(shapes['singles'])

        doubles = vector[slices['doubles']]
        doubles = doubles.reshape(shapes['doubles'])

        return cls(singles=singles, doubles=doubles)

    def pretty_print_mbe(
        self,
        verbose_singles: bool = False,
        verbose_doubles: bool = False,
        file: SupportsWrite | None = None,
    ) -> None:

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
            (self.singles.reshape(-1, 1), self.doubles.reshape(-1, 1))
        ).flatten()

    def flatten_single_count(self, ghf_ov_data: GHF_ov_data) -> NDArray:
        nv = ghf_ov_data.nv
        no = ghf_ov_data.no
        dim_s = nv * no
        dim_d = (nv * (nv - 1) // 2) * (no * (no - 1) // 2)
        out = np.zeros(shape=(dim_s + dim_d))
        out[0:dim_s] = self.singles.reshape(-1)
        abij = 0
        for a in range(0, nv):
            for b in range(a + 1, nv):
                for i in range(0, no):
                    for j in range(i + 1, no):
                        out[dim_s + abij] = self.doubles[a, b, i, j]
                        abij += 1
        return out

    def __eq__(self, other) -> bool:
        if not isinstance(other, GHF_CCSD_MBE):
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
        negated = GHF_CCSD_MBE(-self.singles, -self.doubles)
        return negated
    
    def __matmul__(self, other: GHF_CCSD_MBE) -> float:
        if not isinstance(other, GHF_CCSD_MBE):
            raise ValueError(f"Unsupported operation {self.__class__.__name__}"
                             f" @ {other.__class__.__name__}.")

        return float(
            np.einsum('ai,ai->', self.singles, other.singles)
            + 
            np.einsum('abji,abji->', self.doubles, other.doubles)
       )

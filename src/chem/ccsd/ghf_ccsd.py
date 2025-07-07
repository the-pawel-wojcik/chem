from dataclasses import dataclass

from chem.ccsd.containers import GHF_CCSD_Data, GHF_CCSD_Lambda_Data
from chem.hf.ghf_data import GHF_Data
import numpy as np


@dataclass
class GHF_CCSD_Config:
    verbose: int = 0
    use_diis: bool = True
    max_iterations: int = 100
    energy_convergence: float = 1e-10
    residuals_convergence: float = 1e-10
    shift_1e: float = 0.0
    shift_2e: float = 0.0


class GHF_CCSD:

    def __init__(
        self,
        ghf_data: GHF_Data,
        config: GHF_CCSD_Config | None = None,
    ) -> None:
        self.ghf_data = ghf_data
        nv = ghf_data.nv
        no = ghf_data.no

        self.data = GHF_CCSD_Data(
            t1=np.zeros(shape=(nv, no)),
            t2=np.zeros(shape=(nv, nv, no, no)),
        )

        if config is None:
            self.CONFIG = GHF_CCSD_Config()
        else:
            self.CONFIG = config
        self.dampers = self.build_dampers(
            shift_1e=self.CONFIG.shift_1e,
            shift_2e=self.CONFIG.shift_2e,
        )

        self.cc_solved = False
        self.lambda_cc_solved = False

        if self.CONFIG.use_diis is True:
            self.diis = None
            # TODO:
            # self.diis = DIIS(noa, nva, nob, nvb)
            # self.diis = Alt_DIIS(noa, nva, nob, nvb)
        else:
            self.diis = None

    def build_dampers(self, shift_1e: float = 0.0, shift_2e: float = 0.0):
        """ Helper objects that allow you to take a `matrix` and do
        `matrix / (f_ii - f_aa)`
        by doing
        `(f_ii - f_aa)^-1 * matrix`

        a set of matrices where for each matrix the index [a][i] or
        [a][b][i][j] (you get the point) gives you the inverse of the sum of
        the fock eigenvalues for these indices e.g dampers['aa'][a][i] = 1 /
        (-fock_aa[a][a] + fock_aa[i][i]) See that the values are attempted to
        be negative bc, the virtual eigenvalues come with a minus sign.
        """
        o = self.ghf_data.o
        v = self.ghf_data.v
        new_axis = np.newaxis

        fock_diagonal = self.ghf_data.f.diagonal()
        dampers = {
            'singles': 1.0 / (
                - fock_diagonal[v, new_axis]
                + fock_diagonal[new_axis, o]
                - shift_1e
            ),
            'doubles': 1.0 / (
                - fock_diagonal[v, new_axis, new_axis, new_axis]
                - fock_diagonal[new_axis, v, new_axis, new_axis]
                + fock_diagonal[new_axis, new_axis, o, new_axis]
                + fock_diagonal[new_axis, new_axis, new_axis, o]
                - shift_2e
            ),
        }

        return dampers

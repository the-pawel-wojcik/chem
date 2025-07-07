from dataclasses import dataclass

from numpy.typing import NDArray

from chem.ccsd.containers import GHF_CCSD_Data, GHF_CCSD_Lambda_Data
from chem.ccsd.equations.ghf.cc_residuals.doubles import get_doubles_residual
from chem.ccsd.equations.ghf.cc_residuals.singles import get_singles_residual
from chem.ccsd.equations.ghf.energy.energy import get_ghf_ccsd_energy
from chem.ccsd.equations.ghf.util import GHF_Generators_Input
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

    def solve_cc_equations(self):
        MAX_CCSD_ITER = self.CONFIG.max_iterations
        ENERGY_CONVERGENCE = self.CONFIG.energy_convergence
        RESIDUALS_CONVERGENCE = self.CONFIG.residuals_convergence

        for iter_idx in range(MAX_CCSD_ITER):
            old_energy = self.get_energy()

            residuals = self.calculate_residuals()
            new_t_amps = self.calculate_new_amplitudes(residuals)
            if self.diis is not None:
                new_t_amps = self.diis.find_next_guess(new_t_amps, residuals)
            self.update_t_amps(new_t_amps)

            new_energy = self.get_energy()
            energy_change = new_energy - old_energy
            residuals_norm = self.get_residuals_norm(residuals)
            self.print_iteration_report(
                iter_idx, new_energy, energy_change, residuals_norm,
            )

            energy_converged = np.abs(energy_change) < ENERGY_CONVERGENCE
            residuals_converged = residuals_norm < RESIDUALS_CONVERGENCE

            if energy_converged and residuals_converged:
                break
        else:
            raise RuntimeError("CCSD didn't converge")
        self.cc_solved = True

    def get_energy(self) -> float:
        ghf_ccsd_energy = get_ghf_ccsd_energy(
            ghf_data=self.ghf_data,
            ghf_ccsd_data=self.data,
        )
        return float(ghf_ccsd_energy)

    def calculate_residuals(self):
        residuals = dict()

        kwargs = GHF_Generators_Input(
            ghf_data=self.ghf_data,
            ghf_ccsd_data=self.data,
        )

        residuals['singles'] = get_singles_residual(**kwargs)
        residuals['doubles'] = get_doubles_residual(**kwargs)

        return residuals

    def calculate_new_amplitudes(
        self,
        residuals: dict[str, NDArray]
    ) -> dict[str, NDArray]:
        new_t_amps = dict()
        new_t_amps['singles'] = (
            self.data.t1
            +
            residuals['singles'] * self.dampers['singles']
        )
        new_t_amps['doubles'] = (
            self.data.t2
            +
            residuals['doubles'] * self.dampers['doubles']
        )

        return new_t_amps

    def update_t_amps(self, new_t_amps: dict[str, NDArray]) -> None:
        self.data.t1 = new_t_amps['singles']
        self.data.t2 = new_t_amps['doubles']

    def get_residuals_norm(self, residuals: dict[str, NDArray]) -> float:
        norm = sum(np.linalg.norm(residual) for residual in residuals.values())
        return float(norm)

    def print_iteration_report(
        self,
        iter_idx: int,
        current_energy: float,
        energy_change: float,
        residuals_norm: float,
    ):
        if self.CONFIG.verbose == 0:
            return

        e_fmt = '12.6f'
        print(f"Iteration {iter_idx + 1:>2d}:", end='')
        print(f' {current_energy:{e_fmt}}', end='')
        print(f' {energy_change:{e_fmt}}', end='')
        print(f' {residuals_norm:{e_fmt}}', end='')
        if self.diis is not None:
            if iter_idx + 1 >= self.diis.START_DIIS_AT_ITER:
                print(' DIIS', end='')
        print('')

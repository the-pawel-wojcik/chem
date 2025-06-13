from chem.hf.intermediates_builders import Intermediates
from chem.ccs.containers import UHF_CCS_Data
import numpy as np
from numpy.typing import NDArray
from chem.ccs.equations.energy import get_ccs_energy
from chem.ccs.equations.util import UHF_CCS_InputPair
from chem.ccs.equations.uhf_ccs_singles import (
    get_uhf_ccs_singles_residuals_aa,
    get_uhf_ccs_singles_residuals_bb,
)


class UHF_CCS:

    def __init__(
        self,
        scf_data: Intermediates,
        shift_1e: float = 0.0,
        use_diis: bool = True,
    ) -> None:
        self.scf_data = scf_data
        noa = scf_data.noa
        nva = scf_data.nmo - noa
        nob = scf_data.nob
        nvb = scf_data.nmo - nob

        self.data = UHF_CCS_Data(
            t1_aa = np.zeros(shape=(nva, noa)),
            t1_bb = np.zeros(shape=(nvb, nob)),
        )

        self._dampers = self._build_dampers(shift_1e=shift_1e)

        self.cc_solved = False
        self.lambda_cc_solved = False
        
        if use_diis is True:
            self.diis = None
            # DIIS(noa, nva, nob, nvb)
            # self.diis = Alt_DIIS(noa, nva, nob, nvb)
        else:
            self.diis = None

        # UI
        self.verbose = 0


    def solve_cc_equations(self):
        MAX_CCSD_ITER = 50
        ENERGY_CONVERGENCE = 1e-10
        RESIDUALS_CONVERGENCE = 1e-10

        for iter_idx in range(MAX_CCSD_ITER):
            old_energy = self.get_energy()

            residuals = self._calculate_residuals()
            new_t_amps = self._calculate_new_amplitudes(residuals)
            if self.diis is not None:
                new_t_amps = self.diis.find_next_guess(new_t_amps, residuals)
            self._update_t_amps(new_t_amps)

            new_energy = self.get_energy()
            energy_change = new_energy - old_energy
            residuals_norm = self._get_residuals_norm(residuals)
            self._print_iteration_report(
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
        uhf_ccs_energy = get_ccs_energy(self.scf_data, self.data)
        return float(uhf_ccs_energy)

    def _get_residuals_norm(self, residuals: dict[str, NDArray]) -> float:
        return float(sum(
            np.linalg.norm(residual) for residual in residuals.values()
        ))

    def _calculate_residuals(self) -> dict[str, NDArray]:
        residuals = dict()

        kwargs = UHF_CCS_InputPair(
            uhf_scf_data=self.scf_data,
            uhf_ccs_data=self.data,
        )

        residuals['aa'] = get_uhf_ccs_singles_residuals_aa(**kwargs)
        residuals['bb'] = get_uhf_ccs_singles_residuals_bb(**kwargs)

        return residuals

    def _calculate_new_amplitudes(
        self,
        residuals: dict[str, NDArray],
    ) -> dict[str, NDArray]:
        new_t_amps = dict()
        new_t_amps['aa'] = (
            self.data.t1_aa + residuals['aa'] * self._dampers['aa']
        )
        new_t_amps['bb'] = (
            self.data.t1_bb + residuals['bb'] * self._dampers['bb']
        )

        return new_t_amps

    def _update_t_amps(self, new_t_amps: dict[str, NDArray]) -> None:
        self.data.t1_aa = new_t_amps['aa']
        self.data.t1_bb = new_t_amps['bb']

    def _build_dampers(self, shift_1e: float = 0.0) -> dict[str, NDArray]:
        """ Helper objects that allow you to take a `matrix` and do
        `matrix / (f_ii - f_aa)`
        by doing
        `(f_ii - f_aa)^-1 * matrix`

        a set of matrices where for each matrix the index [a][i] gives you the
        inverse of the sum of the fock eigenvalues for these indices e.g
        dampers['aa'][a][i] = 1 / (-fock_aa[a][a] + fock_aa[i][i]) See that the
        values are attempted to be negative bc, the virtual eigenvalues come
        with a minus sign.
        """
        oa = self.scf_data.oa
        va = self.scf_data.va
        ob = self.scf_data.ob
        vb = self.scf_data.vb
        new_axis = np.newaxis

        fock_energy_a = self.scf_data.f_aa.diagonal()
        fock_energy_b = self.scf_data.f_bb.diagonal()
        dampers = {
            'aa': 1.0 / (
                - fock_energy_a[va, new_axis]
                + fock_energy_a[new_axis, oa]
                - shift_1e
            ),
            'bb': 1.0 / (
                - fock_energy_b[vb, new_axis]
                + fock_energy_b[new_axis, ob]
                - shift_1e
            ),
        }

        return dampers

    def _print_iteration_report(
        self,
        iter_idx: int,
        current_energy: float,
        energy_change: float,
        residuals_norm: float,
    ) -> None:
        if self.verbose == 0:
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

from chem.hf.intermediates_builders import Intermediates
from chem.ccs.containers import UHF_CCS_Data, UHF_CCS_Lambda_Data
import numpy as np
from numpy.typing import NDArray
from chem.ccs.equations.energy import get_ccs_energy
from chem.ccs.equations.util import UHF_CCS_InputPair, UHF_CCS_InputTriple
from chem.ccs.equations.uhf_ccs_singles import (
    get_uhf_ccs_singles_residuals_aa,
    get_uhf_ccs_singles_residuals_bb,
)
from chem.ccs.equations.lmbd.uhf_ccs_lmbd_singles_res import (
    get_uhf_ccs_lambda_singles_res_aa,
    get_uhf_ccs_lambda_singles_res_bb,
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
        self.cc_lambda_data = UHF_CCS_Lambda_Data(
            l1_aa=np.zeros(shape=(noa, nva)),
            l1_bb=np.zeros(shape=(noa, nva)),
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

    def get_energy(self) -> float:
        uhf_ccs_energy = get_ccs_energy(self.scf_data, self.data)
        return float(uhf_ccs_energy)

    def solve_cc_equations(self) -> None:
        """The CCS equations are solved by T = 0. So all of this is useless."""
        MAX_CCS_ITER = 50
        ENERGY_CONVERGENCE = 1e-10
        RESIDUALS_CONVERGENCE = 1e-10

        for iter_idx in range(MAX_CCS_ITER):
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
            raise RuntimeError("UHF-CCS didn't converge.")
        self.cc_solved = True

    def solve_lambda_equations(self) -> None:
        """Unnecessary: for T=0, CCS-Lambda are solved by L=0."""
        MAX_CCS_LAMBDA_ITER = 50
        RESIDUALS_CONVERGENCE = 1e-10

        if self.cc_solved is False:
            self.solve_cc_equations()

        for iter_idx in range(MAX_CCS_LAMBDA_ITER):

            residuals = self._calculate_lambda_residuals()
            new_lambdas = self._calculate_new_lambdas(residuals)
            if self.diis is not None:
                new_lambdas = self.diis.find_next_guess(new_lambdas, residuals)
            self._update_lambdas(new_lambdas)

            residuals_norm = self._get_residuals_norm(residuals)
            self._print_lambda_iteration_report(iter_idx, residuals_norm)
            residuals_converged = residuals_norm < RESIDUALS_CONVERGENCE
            if residuals_converged:
                break
        else:
            raise RuntimeError("UHF-CCS-Lambda equations didn't converge.")
        self.lambda_cc_solved = True

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

    def _calculate_lambda_residuals(self) -> dict[str, NDArray]:
        residuals = dict()

        kwargs = UHF_CCS_InputTriple(
            uhf_data=self.scf_data,
            uhf_ccs_data=self.data,
            uhf_ccs_lambda_data=self.cc_lambda_data,
        )

        residuals['aa'] = get_uhf_ccs_lambda_singles_res_aa(**kwargs)
        residuals['bb'] = get_uhf_ccs_lambda_singles_res_bb(**kwargs)

        return residuals

    def _calculate_new_lambdas(
        self,
        residuals: dict[str, NDArray],
    ) -> dict[str, NDArray]:
        new_lambdas = dict()
        if self.cc_lambda_data is None:
            raise RuntimeError("UHF-CCS-Lambda uninitialized.")

        lmbda = self.cc_lambda_data
        dampers = self._dampers

        new_lambdas['aa'] = (
            lmbda.l1_aa + residuals['aa'] * dampers['aa'].transpose((1, 0))
        )
        new_lambdas['bb'] = (
            lmbda.l1_bb + residuals['bb'] * dampers['bb'].transpose((1, 0))
        )

        return new_lambdas

    def _update_lambdas(self, new_lambdas: dict[str, NDArray]) -> None:
        self.cc_lambda_data.l1_aa = new_lambdas['aa']
        self.cc_lambda_data.l1_bb = new_lambdas['bb']

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

    def _print_lambda_iteration_report(
        self,
        iter_idx: int,
        residuals_norm: float,
    ) -> None:
        if self.verbose == 0:
            return

        e_fmt = '12.6f'
        print(f"Iteration {iter_idx + 1:>2d}:", end='')
        print(f' {residuals_norm:{e_fmt}}', end='')
        if self.diis is not None:
            if iter_idx + 1 >= self.diis.START_DIIS_AT_ITER:
                print(' DIIS', end='')
        print('')

from dataclasses import dataclass
from itertools import product

from numpy.typing import NDArray

from chem.ccsd.containers import GHF_CCSD_Data, GHF_CCSD_Lambda_Data
from chem.ccsd.equations.ghf.cc_residuals.doubles import get_doubles_residual
from chem.ccsd.equations.ghf.cc_residuals.singles import get_singles_residual
from chem.ccsd.equations.ghf.energy.energy import get_ghf_ccsd_energy
from chem.ccsd.equations.ghf.lmbd.singles import get_lambda_singles_residual
from chem.ccsd.equations.ghf.lmbd.doubles import get_lambda_doubles_residual
from chem.ccsd.equations.ghf.dipole_moment.edm import (
    get_mux, get_muy, get_muz
)
from chem.ccsd.equations.ghf.util import GHF_Generators_Input
from chem.hf.ghf_data import GHF_Data
from chem.meta.coordinates import Descartes
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
    t_amp_print_threshold: float = 0.01


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

    def solve_lambda_equations(self) -> None:
        MAX_CCSD_ITER = self.CONFIG.max_iterations
        ENERGY_CONVERGENCE = self.CONFIG.energy_convergence
        RESIDUALS_CONVERGENCE = self.CONFIG.residuals_convergence
        if self.cc_solved is False:
            self.solve_cc_equations()
        CC_ENERGY = self.get_energy()

        self.initialize_lambda()

        for iter_idx in range(MAX_CCSD_ITER):
            old_pseudoenergy = self._calculate_lambda_pseudoenergy()

            residuals = self.calculate_lambda_residuals(CC_ENERGY)
            new_lambdas = self.calculate_new_lambdas(residuals)
            self.update_lambdas(new_lambdas)

            new_pseudoenergy = self._calculate_lambda_pseudoenergy()
            pseudoenergy_change = abs(new_pseudoenergy - old_pseudoenergy)
            residuals_norm = float(self.get_residuals_norm(residuals))
            self.print_lambda_iteration_report(
                iter_idx, residuals_norm, new_pseudoenergy,
                pseudoenergy_change,
            )
            residuals_converged = residuals_norm < RESIDUALS_CONVERGENCE
            pseudoenergy_converged = pseudoenergy_change < ENERGY_CONVERGENCE
            if residuals_converged and pseudoenergy_converged:
                break
        else:
            raise RuntimeError("Lambda-GHF_CCSD didn't converge.")
        self.lambda_cc_solved = True

    def _get_electronic_electric_dipole_moment(self) -> dict[Descartes, float]:
        if self.lambda_cc_solved is False:
            self.solve_lambda_equations()
        kwargs = GHF_Generators_Input(
            ghf_data=self.ghf_data,
            ghf_ccsd_data=self.data,
        )
        electronic_electric_dipole_moment = {
            Descartes.x: float(get_mux(**kwargs)),
            Descartes.y: float(get_muy(**kwargs)),
            Descartes.z: float(get_muz(**kwargs)),
        }
        return electronic_electric_dipole_moment

    def get_energy(self) -> float:
        # TODO: the energy calculation is used before the CC equations are
        # solved but it might be better if the user-facing interface shows an
        # error if this is used
        # Move this get_energy to _get_energy and in get_energy check if the
        # energy check if cc_solved is True and raise and solve them first if
        # not solved yet.
        ghf_ccsd_energy = get_ghf_ccsd_energy(
            ghf_data=self.ghf_data,
            ghf_ccsd_data=self.data,
        )
        return float(ghf_ccsd_energy)

    def print_leading_t_amplitudes(self) -> None:
        top_t1 = self._find_leading_t1_amplitudes()
        top_t2 = self._find_leading_t2_amplitudes()

        top_t1.sort(key=lambda x: abs(x['amp']), reverse=True)
        top_t2.sort(key=lambda x: abs(x['amp']), reverse=True)

        THRESHOLD = self.CONFIG.t_amp_print_threshold
        with np.printoptions(precision=3, suppress=True):
            print(f"t1 amplitudes greater than {THRESHOLD:.0e}:")
            print(f"{'v':>3s} {'o':>3s} {'t1[v,o]':^7s}")
            for top in top_t1:
                print(f'{top['v']:>3d} {top['o']:>3d} {top['amp']:+7.3f}')
            print(f'Norm the t1 = {np.linalg.norm(self.data.t1):.3f}')

            print(f"t2 amplitudes greater than {THRESHOLD:.0e}:")
            print(
                f"{'vl':>3s} {'vr':>3s} {'ol':>3s} {'or':>3s}"
                f" {'t2[vl,vr,ol,or]'}"
            )
            for top in top_t2:
                print(
                    f'{top['vl']:>3d} {top['vr']:>3d} {top['ol']:>3d}'
                    f' {top['or']:>3d} {top['amp']:+7.3f}'
                )
            print(f'Norm the t2 = {np.linalg.norm(self.data.t2):.3f}')

    def print_leading_lambda_amplitudes(self) -> None:
        if not self.lambda_cc_solved:
            print("No lambda amplitudes available for printing.")
            print("Lambda equations not solved.")
            return
        assert self.data.lmbda is not None

        top_l1 = self._find_leading_l1_amplitudes()
        top_l2 = self._find_leading_l2_amplitudes()

        top_l1.sort(key=lambda x: abs(x['amp']), reverse=True)
        top_l2.sort(key=lambda x: abs(x['amp']), reverse=True)

        THRESHOLD = self.CONFIG.t_amp_print_threshold
        with np.printoptions(precision=3, suppress=True):
            print(f"l1 amplitudes greater than {THRESHOLD:.0e}:")
            print(f"{'o':>3s} {'v':>3s} {'l1[o,v]':^7s}")
            for top in top_l1:
                print(f'{top['o']:>3d} {top['v']:>3d} {top['amp']:+7.3f}')
            print(f'Norm the l1 = {np.linalg.norm(self.data.lmbda.l1):.3f}')

            print(f"l2 amplitudes greater than {THRESHOLD:.0e}:")
            print(
                f"{'ol':>3s} {'or':>3s} {'vl':>3s} {'vr':>3s}"
                f" {'l2[ol,or,vl,vr]'}"
            )
            for top in top_l2:
                print(
                    f'{top['ol']:>3d} {top['or']:>3d}'
                    f' {top['vl']:>3d} {top['vr']:>3d}'
                    f' {top['amp']:+7.3f}'
                )
            print(f'Norm the l2 = {np.linalg.norm(self.data.lmbda.l2):.3f}')

    def _find_leading_t1_amplitudes(self) -> list[dict[str, int | float]]:
        t1 = self.data.t1
        no = self.ghf_data.no
        nv = self.ghf_data.nv
        top_t1 = []
        THRESHOLD = self.CONFIG.t_amp_print_threshold
        for v, o in product(range(nv), range(no)):
            amp = t1[v, o]
            if abs(amp) > THRESHOLD:
                top_t1.append({'v': v, 'o': o, 'amp': amp})
        return top_t1

    def _find_leading_t2_amplitudes(self) -> list[dict[str, int | float]]:
        t2 = self.data.t2
        no = self.ghf_data.no
        nv = self.ghf_data.nv
        THRESHOLD = self.CONFIG.t_amp_print_threshold
        top_t2 = []
        for virl, virr, occl, occr in product(
            range(nv), range(nv), range(no), range(no)
        ):
            amp = t2[virl, virr, occl, occr]
            if abs(amp) > THRESHOLD:
                top_t2.append({
                    'vl': virl,
                    'vr': virr,
                    'ol': occl,
                    'or': occr,
                    'amp': amp
                })
        return top_t2

    def _find_leading_l1_amplitudes(self) -> list[dict[str, int | float]]:
        assert self.data.lmbda is not None
        l1 = self.data.lmbda.l1
        no = self.ghf_data.no
        nv = self.ghf_data.nv
        top_l1 = []
        THRESHOLD = self.CONFIG.t_amp_print_threshold
        for o, v in product(range(no), range(nv)):
            amp = l1[o, v]
            if abs(amp) > THRESHOLD:
                top_l1.append({'v': v, 'o': o, 'amp': amp})
        return top_l1

    def _find_leading_l2_amplitudes(self) -> list[dict[str, int | float]]:
        assert self.data.lmbda is not None
        l2 = self.data.lmbda.l2
        no = self.ghf_data.no
        nv = self.ghf_data.nv
        THRESHOLD = self.CONFIG.t_amp_print_threshold
        top_l2 = []
        for occl, occr, virl, virr in product(
            range(no), range(no), range(nv), range(nv)
        ):
            amp = l2[occl, occr, virl, virr]
            if abs(amp) > THRESHOLD:
                top_l2.append({
                    'vl': virl,
                    'vr': virr,
                    'ol': occl,
                    'or': occr,
                    'amp': amp
                })
        return top_l2

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

    def initialize_lambda(self):
        if self.data.lmbda is not None:
            msg = "Re-initializing GHF CCSD Lambda."
            raise RuntimeError(msg)

        self.data.lmbda = GHF_CCSD_Lambda_Data(
            l1=self.data.t1.copy().transpose((1, 0)),
            l2=self.data.t2.copy().transpose((2, 3, 0, 1)),
        )

    def _calculate_lambda_pseudoenergy(self) -> float:
        assert self.data.lmbda is not None
        l1 = self.data.lmbda.l1
        l2 = self.data.lmbda.l2
        f = self.ghf_data.f
        g = self.ghf_data.g
        o = self.ghf_data.o
        v = self.ghf_data.v

        pseudo_energy = np.einsum('ia,ai->', l1, f[v, o], optimize=True)
        pseudo_energy += 0.25 * np.einsum(
            'ijab,abij->', l2, g[v, v, o, o], optimize=True
        )
        pseudo_energy += 0.5 * np.einsum(
            'ia,jb,abij->', l1, l1, g[v, v, o, o], optimize=True
        )
        return float(pseudo_energy)

    def calculate_lambda_residuals(self, CC_ENERGY: float):
        residuals = dict()

        kwargs = GHF_Generators_Input(
            ghf_data=self.ghf_data,
            ghf_ccsd_data=self.data,
        )

        if self.data.lmbda is None:
            raise RuntimeError("GHF CCSD Lambda uninitialized.")

        # I don't know how to do this in pdaggerq better, the rhs just has the
        # energy times a coefficients that I subtract here
        residuals['singles'] = (
            get_lambda_singles_residual(**kwargs)
            - CC_ENERGY * self.data.lmbda.l1
        )
        residuals['doubles'] = (
            get_lambda_doubles_residual(**kwargs)
            - CC_ENERGY * self.data.lmbda.l2
        )

        return residuals

    def calculate_new_lambdas(
        self,
        residuals: dict[str, NDArray],
    ) -> dict[str, NDArray]:
        new_lambdas = dict()
        if self.data.lmbda is None:
            raise RuntimeError("GHF CCSD Lambda uninitialized.")

        lmbda = self.data.lmbda
        dampers = self.dampers

        new_lambdas['singles'] = (
            lmbda.l1
            +
            residuals['singles'] * dampers['singles'].transpose((1, 0))
        )

        its_oovv_now = (2, 3, 0, 1)
        new_lambdas['doubles'] = (
            lmbda.l2
            +
            residuals['doubles'] * dampers['doubles'].transpose(its_oovv_now)
        )

        return new_lambdas

    def update_lambdas(self, new_lambdas: dict[str, NDArray]) -> None:
        if self.data.lmbda is None:
            raise RuntimeError("GHF CCSD Lambda uninitialized.")

        lmbda = self.data.lmbda
        lmbda.l1 = new_lambdas['singles']
        lmbda.l2 = new_lambdas['doubles']

    def print_lambda_iteration_report(
        self, iter_idx: int, residuals_norm: float, pseudoenergy: float,
        pseudoenergy_change: float,
    ):
        if self.CONFIG.verbose == 0:
            return

        e_fmt = '12.6f'
        print(f"Iteration {iter_idx + 1:>2d}:", end='')
        print(f' {residuals_norm:{e_fmt}}', end='')
        print(f' {pseudoenergy:{e_fmt}}', end='')
        print(f' {pseudoenergy_change:{e_fmt}}', end='')
        # TODO:
        # if self.diis is not None:
        #     if iter_idx + 1 >= self.diis.START_DIIS_AT_ITER:
        #         print(' DIIS', end='')
        print('')

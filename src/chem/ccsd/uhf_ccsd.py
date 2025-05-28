from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from chem.ccsd.containers import UHF_CCSD_Data
from chem.ccsd.equations.uhf_ccsd_energy import (
    get_energy,
)
from chem.ccsd.equations.uhf_ccsd_singles_res import (
    get_singles_residual_aa,
    get_singles_residual_bb,
)
from chem.ccsd.equations.uhf_ccsd_doubles_res import (
    get_doubles_residual_aaaa,
    get_doubles_residual_abab,
    get_doubles_residual_abba,
    get_doubles_residual_baab,
    get_doubles_residual_baba,
    get_doubles_residual_bbbb,
)
from chem.ccsd.equations.util import GeneratorsInput
from chem.hf.intermediates_builders import Intermediates


@dataclass
class DIIS:
    """ Direct Inversion in the Iterative Space (DIIS) by P. Pulay.

    [1] P. Pulay, Improved SCF Convergence Acceleration, Journal of
    Computational Chemistry 3, 556 (1982).
    [2] P. Pulay, Convergence Acceleration of Iterative Sequences. The Case of
    SCF Iteration, Chemical Physics Letters 73, 393 (1980).
    """
    def __init__(self, noa: int, nva: int, nob: int, nvb: int) -> None:
        self.STORAGE_SIZE: int = 10
        self.START_DIIS_AT_ITER: int = 2
        self.current_iteration: int = 0
        self.noa = noa
        self.nva = nva
        self.nob = nob
        self.nvb = nvb
        self.problem_dim = nva * noa + nvb * nob
        self.problem_dim += nva * nva * noa * noa
        self.problem_dim += nva * nvb * noa * nob
        self.problem_dim += nvb * nvb * nob * nob

        # vectors are stored as columns of the matrices
        # access the last vector with matrix[:, -1]
        self.residuals_matrix = np.zeros(shape=(self.problem_dim, 0))
        self.guesses_matrix = np.zeros(shape=(self.problem_dim, 0))

    def find_next_guess(
        self,
        guess: dict[str, NDArray],
        residual: dict[str, NDArray],
    ) -> dict[str, NDArray]:

        self.residuals_matrix = np.hstack([self.residuals_matrix, residual])
        self.guesses_matrix = np.hstack([self.guesses_matrix, guess])
        self.current_iteration += 1

        if self.current_iteration < self.START_DIIS_AT_ITER:
            return guess

        # TODO: automate this and don't overstore. Merge this step with the
        # addition of the new guess/residual.
        if self.residuals_matrix.shape[1] > self.STORAGE_SIZE:
            self.residuals_matrix = self.residuals_matrix[:, 1:]

        if self.guesses_matrix.shape[1] > self.STORAGE_SIZE:
            self.residuals_matrix = self.residuals_matrix[:, 1:]

        matrix, rhs = self.build_linear_problem()
        c = np.linalg.solve(matrix, rhs)
        assert self.residuals_matrix.shape[1] == c.shape[0]
        new_guess = self.residuals_matrix @ c
        return self.unfold(new_guess)

    def unfold(self, vector):
        """ Revert the flattening of the input vector, i.e., return somethign
        like this
        unfolded = {
            'aa': NDarray,
            'bb': NDarray,
            'aaaa': NDarray,
            'abab': NDarray,
            'abba': NDarray,
            'baab': NDarray,
            'baba': NDarray,
            'bbbb': NDarray,
        }
        """
        noa = self.noa
        nob = self.nob
        nva = self.nva
        nvb = self.nvb

    def update_state(
        self,
        residuals: dict[str, NDArray],
        guesses: dict[str, NDArray],
    ) -> None:
        """ New iteration begins

        """
        self.current_iteration += 1

    def _add_new_residual(self, residual):
        """
        Hoping that the new residuals/guesses looks like this
        ```
        residuals = {
            'aa': NDarray,
            'bb': NDarray,
            'aaaa': NDarray,
            'abab': NDarray,
            'abba': NDarray,
            'baab': NDarray,
            'baba': NDarray,
            'bbbb': NDarray,
        }
        ```
        """
        noa = self.noa
        nob = self.nob
        nva = self.nva
        nvb = self.nvb

        if self.residuals_matrix.shape[1] == self.STORAGE_SIZE:
            current_residuals = self.residuals_matrix[:, 1:]
        else:
            current_residuals = self.residuals_matrix

        # TODO: make sure this makes sense
        self.residuals_matrix = np.hstack(
            (
                current_residuals,
                np.vstack(
                    (
                        residual['aa'].reshape(nva * noa),
                        residual['bb'].reshape(nvb * nob),
                        residual['aaaa'].reshape(nva * nva * noa * noa),
                        residual['abab'].reshape(nva * nvb * noa * nob),
                        residual['bbbb'].reshape(nvb * nvb * nob * nob),
                    )
                ).reshape(-1, 1),
            )
        )

    def build_linear_problem(self) -> tuple[NDArray, NDArray]:
        dim = self.problem_dim
        matrix = self.residuals_matrix @ self.residuals_matrix.T
        matrix = np.vstack((
            matrix,
            -1 * np.ones((1, dim))
        ))
        matrix = np.hstack((
            matrix, -1 * np.ones((dim + 1, 1))
        ))
        matrix[-1][-1] = 0.0
        rhs = np.zeros((dim + 1, 1))
        rhs[-1][0] = -1
        return matrix, rhs


class UHF_CCSD:

    def __init__(self, scf_data: Intermediates) -> None:
        self.scf_data = scf_data
        noa = scf_data.noa
        nva = scf_data.nmo - noa
        nob = scf_data.nob
        nvb = scf_data.nmo - nob

        self.data = UHF_CCSD_Data(
            t1_aa = np.zeros(shape=(nva, noa)),
            t1_bb = np.zeros(shape=(nvb, nob)),
            t2_aaaa = np.zeros(shape=(nva, nva, noa, noa)),
            t2_abab = np.zeros(shape=(nva, nvb, noa, nob)),
            t2_bbbb = np.zeros(shape=(nvb, nvb, nob, nob)),
            # spin-changing terms that do not appear in the CC equations
            # but appear as residues
            t2_abba = np.zeros(shape=(nva, nvb, nob, noa)),
            t2_baab = np.zeros(shape=(nvb, nva, noa, nob)),
            t2_baba = np.zeros(shape=(nvb, nva, nob, noa)),
        )

        self.dampers = self.build_dampers(shift_1e=0.1)

        # UI
        self.verbose = 0


    def solve_cc_equations(self):
        MAX_CCSD_ITER = 50
        ENERGY_CONVERGENCE = 1e-6
        RESIDUALS_CONVERGENCE = 1e-6

        # diis = DIIS(noa=self.noa, nva=self.nva, nob=self.nob, nvb=self.nvb)

        for iter_idx in range(MAX_CCSD_ITER):
            residuals = self.calculate_residuals()
            new_t_amps = self.calculate_new_amplitudes(residuals)

            old_energy = self.get_energy()
            self.update_t_amps(new_t_amps)
            current_energy = self.get_energy()
            energy_change = current_energy - old_energy
            residuals_norm = self.get_residuals_norm(residuals)
            self.print_iteration_report(
                iter_idx + 1, current_energy, energy_change, residuals_norm
            )

            energy_converged = np.abs(energy_change) < ENERGY_CONVERGENCE
            residuals_converged = residuals_norm < RESIDUALS_CONVERGENCE

            if energy_converged and residuals_converged:
                break
            old_energy = current_energy
        else:
            raise RuntimeError("CCSD didn't converge")

    def get_residuals_norm(self, residuals):
        return sum(np.linalg.norm(residual) for residual in residuals.values())

    def print_the_largest_element(self, array, header: str = ''):
        max_index = np.unravel_index(np.argmax(np.abs(array)), array.shape)
        max_element = array[max_index]
        if header != '':
            print(header + ' ', end='')
        print(f'max value = {max_element:.6f}', end='')
        print(f' at index {tuple(int(idx) for idx in max_index)}')

    def print_iteration_report(
        self, iter_idx, current_energy, energy_change, residuals_norm,
    ):
        if self.verbose == 0:
            return

        e_fmt = '12.6f'
        print(f"Iteration {iter_idx:>2d}:", end='')
        print(f' {current_energy:{e_fmt}}', end='')
        print(f' {energy_change:{e_fmt}}', end='')
        print(f' {residuals_norm:{e_fmt}}')

    def update_t_amps(self, new_t_amps):
        self.data.t1_aa = new_t_amps['aa']
        self.data.t1_bb = new_t_amps['bb']
        self.data.t2_aaaa = new_t_amps['aaaa']
        self.data.t2_abab = new_t_amps['abab']
        self.data.t2_bbbb = new_t_amps['bbbb']
        # spin-changing terms
        self.data.t2_abba = new_t_amps['abba']
        self.data.t2_baab = new_t_amps['baab']
        self.data.t2_baba = new_t_amps['baba']

    def calculate_residuals(self):
        residuals = dict()

        kwargs = GeneratorsInput(
            uhf_scf_data=self.scf_data,
            uhf_ccsd_data=self.data,
        )

        residuals['aa'] = get_singles_residual_aa(**kwargs)
        residuals['bb'] = get_singles_residual_bb(**kwargs)

        residuals['aaaa'] = get_doubles_residual_aaaa(**kwargs)
        residuals['abab'] = get_doubles_residual_abab(**kwargs)
        residuals['abba'] = get_doubles_residual_abba(**kwargs)
        residuals['baab'] = get_doubles_residual_baab(**kwargs)
        residuals['baba'] = get_doubles_residual_baba(**kwargs)
        residuals['bbbb'] = get_doubles_residual_bbbb(**kwargs)

        return residuals

    def calculate_new_amplitudes(self, residuals):
        new_t_amps = dict()
        new_t_amps['aa'] =\
            self.data.t1_aa + residuals['aa'] * self.dampers['aa']
        new_t_amps['bb'] =\
            self.data.t1_bb + residuals['bb'] * self.dampers['bb']
        new_t_amps['aaaa'] =\
            self.data.t2_aaaa + residuals['aaaa'] * self.dampers['aaaa']
        new_t_amps['abab'] =\
            self.data.t2_abab + residuals['abab'] * self.dampers['abab']
        new_t_amps['bbbb'] =\
            self.data.t2_bbbb + residuals['bbbb'] * self.dampers['bbbb']
        # spin-changing terms
        new_t_amps['abba'] =\
            self.data.t2_abba + residuals['abba'] * self.dampers['abba']
        new_t_amps['baab'] =\
            self.data.t2_baab + residuals['baab'] * self.dampers['baab']
        new_t_amps['baba'] =\
            self.data.t2_baba + residuals['baba'] * self.dampers['baba']

        return new_t_amps

    def get_energy(self) -> float:
        uhf_ccsd_energy = get_energy(
            uhf_scf_data=self.scf_data,
            uhf_ccsd_data=self.data,
        )
        return float(uhf_ccsd_energy)

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
            'aaaa': 1.0 / (
                - fock_energy_a[va, new_axis, new_axis, new_axis]
                - fock_energy_a[new_axis, va, new_axis, new_axis]
                + fock_energy_a[new_axis, new_axis, oa, new_axis]
                + fock_energy_a[new_axis, new_axis, new_axis, oa]
                - shift_2e
            ),
            'abab': 1.0 / (
                - fock_energy_a[va, new_axis, new_axis, new_axis]
                - fock_energy_b[new_axis, vb, new_axis, new_axis]
                + fock_energy_a[new_axis, new_axis, oa, new_axis]
                + fock_energy_b[new_axis, new_axis, new_axis, ob]
                - shift_2e
            ),
            'bbbb': 1.0 / (
                - fock_energy_b[vb, new_axis, new_axis, new_axis]
                - fock_energy_b[new_axis, vb, new_axis, new_axis]
                + fock_energy_b[new_axis, new_axis, ob, new_axis]
                + fock_energy_b[new_axis, new_axis, new_axis, ob]
                - shift_2e
            ),
            # spin-changing terms
            'abba': 1.0 / (
                - fock_energy_a[va, new_axis, new_axis, new_axis]
                - fock_energy_b[new_axis, vb, new_axis, new_axis]
                + fock_energy_b[new_axis, new_axis, ob, new_axis]
                + fock_energy_a[new_axis, new_axis, new_axis, oa]
                - shift_2e
            ),
            'baab': 1.0 / (
                - fock_energy_b[vb, new_axis, new_axis, new_axis]
                - fock_energy_a[new_axis, va, new_axis, new_axis]
                + fock_energy_a[new_axis, new_axis, oa, new_axis]
                + fock_energy_b[new_axis, new_axis, new_axis, ob]
                - shift_2e
            ),
            'baba': 1.0 / (
                - fock_energy_b[vb, new_axis, new_axis, new_axis]
                - fock_energy_a[new_axis, va, new_axis, new_axis]
                + fock_energy_b[new_axis, new_axis, ob, new_axis]
                + fock_energy_a[new_axis, new_axis, new_axis, oa]
                - shift_2e
            ),
        }

        return dampers

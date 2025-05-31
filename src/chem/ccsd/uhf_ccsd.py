from dataclasses import dataclass
import functools
import numpy as np
from numpy.typing import NDArray
from chem.ccsd.containers import UHF_CCSD_Data, UHF_CCSD_Lambda_Data
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
from chem.ccsd.equations.lmbd.lambda_singles_res import (
    get_lambda_singles_res_aa,
    get_lambda_singles_res_bb,
)
from chem.ccsd.equations.lmbd.lambda_doubles_res import (
    get_lambda_doubles_res_aaaa,
    get_lambda_doubles_res_abab,
    get_lambda_doubles_res_bbbb,
)
from chem.ccsd.equations.util import GeneratorsInput
from chem.hf.intermediates_builders import Intermediates


@dataclass
class Alt_DIIS:
    """ Direct Inversion in the Iterative Space (DIIS) by P. Pulay.

    [1] P. Pulay, Convergence Acceleration of Iterative Sequences. The Case of
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
        self.problem_dim = nva * noa + nvb * nob   # aa + bb
        self.problem_dim += nva * nva * noa * noa  # aaaa
        self.problem_dim += nva * nvb * noa * nob  # abab
        self.problem_dim += nvb * nvb * nob * nob  # bbbb

        # vectors are stored as columns of the matrices
        # access the last vector with matrix[:, -1]
        self.residuals_matrix = np.zeros(shape=(self.problem_dim, 0))
        self.guesses_matrix = np.zeros(shape=(self.problem_dim, 0))

    def find_next_guess(
        self,
        guess: dict[str, NDArray],
        residual: dict[str, NDArray],
    ) -> dict[str, NDArray]:

        self.update_state(residual, guess)

        if self.current_iteration < self.START_DIIS_AT_ITER:
            return guess

        matrix, rhs = self.build_linear_problem()
        np.set_printoptions(precision=3)
        c = np.linalg.solve(matrix, rhs)
        coefficients = c[:-1, :]
        assert self.guesses_matrix.shape[1] == coefficients.shape[0]
        new_guess = self.guesses_matrix @ coefficients
        return self._unflatten(new_guess, self.save_for_later_guess)

    def update_state(
        self,
        new_residual: dict[str, NDArray],
        new_guess: dict[str, NDArray],
    ) -> None:
        self.current_iteration += 1

        self.blocks_in_use = {'aa', 'bb', 'aaaa', 'abab', 'bbbb'}

        self.residuals_matrix, self.save_for_later_res = self._add_new_vector(
            self.residuals_matrix, new_residual
        )
        self.guesses_matrix, self.save_for_later_guess = self._add_new_vector(
            self.guesses_matrix, new_guess
        )

    def _add_new_vector(
        self,
        vectors_store: NDArray,
        new_vector: dict[str, NDArray],
    ) -> tuple[NDArray, dict[str, NDArray]]:
        """ Flattens the new_vector and adds it to the storage. """

        if vectors_store.shape[1] == self.STORAGE_SIZE:
            current_store = vectors_store[:, 1:]
        else:
            current_store = vectors_store

        flattened_vector, save_for_later = self._flatten(new_vector)
        vectors_store = np.hstack(
            (current_store, flattened_vector)
        )
        return vectors_store, save_for_later

    def _flatten(
        self, new_vector: dict[str, NDArray],
    ) -> tuple[NDArray, dict[str, NDArray]]:
        noa = self.noa
        nob = self.nob
        nva = self.nva
        nvb = self.nvb

        flattened_vector = np.hstack(
            (
                new_vector['aa'].reshape(nva * noa),
                new_vector['bb'].reshape(nvb * nob),
                new_vector['aaaa'].reshape(nva * nva * noa * noa),
                new_vector['abab'].reshape(nva * nvb * noa * nob),
                new_vector['bbbb'].reshape(nvb * nvb * nob * nob),
            )
        ).reshape(-1, 1)

        save_for_later  = {
            block: new_vector[block] 
            for block in (set(new_vector.keys()) - self.blocks_in_use)
        }

        return flattened_vector, save_for_later

    def _unflatten(
        self,
        vector: NDArray,
        saved_reminder: None | dict[str, NDArray] = None,
    ) -> dict[str, NDArray]:
        """ Revert the `_flatten`, i.e., return something like this:
        ```
        {
            'aa': NDarray,
            'bb': NDarray,
            'aaaa': NDarray,
            'abab': NDarray,
            'abba': NDarray,
        }
        ```
        """
        noa = self.noa
        nob = self.nob
        nva = self.nva
        nvb = self.nvb

        blocks = [
            'aa', 'bb', 'aaaa', 'abab', 'bbbb',
        ]

        shapes = {
            'aa': (nva, noa),
            'bb': (nvb, nob),
            'aaaa': (nva, nva, noa, noa),
            'abab': (nva, nvb, noa, nob),
            'bbbb': (nvb, nvb, nob, nob),
        }

        dims = {
            block: functools.reduce(lambda x, y: x * y, shapes[block],1)
            for block in blocks
        }

        dim_sum = 0
        slices = {}
        for block in blocks:
            block_dim = dims[block]
            slices[block] = slice(dim_sum, dim_sum + block_dim)
            dim_sum += block_dim

        assert len(vector.shape) == 2
        assert vector.shape[0] == dim_sum
        assert vector.shape[1] == 1

        unflatten = {
            block: vector[slices[block]].reshape(shapes[block])
            for block in blocks
        }

        if saved_reminder is not None:
            unflatten = unflatten | saved_reminder

        return unflatten

    def build_linear_problem(self) -> tuple[NDArray, NDArray]:
        b_matrix_dim = self.residuals_matrix.shape[1]
        matrix = self.residuals_matrix.T @ self.residuals_matrix
        matrix = np.vstack((
            matrix,
            -1 * np.ones((1, b_matrix_dim))
        ))
        matrix = np.hstack((
            matrix, -1 * np.ones((b_matrix_dim + 1, 1))
        ))
        matrix[-1][-1] = 0.0
        rhs = np.zeros((b_matrix_dim + 1, 1))
        rhs[-1, 0] = -1
        return matrix, rhs


@dataclass
class DIIS:
    """ Direct Inversion in the Iterative Space (DIIS) by P. Pulay.

    [1] P. Pulay, Convergence Acceleration of Iterative Sequences. The Case of
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
        self.problem_dim = nva * noa + nvb * nob   # aa + bb
        self.problem_dim += nva * nva * noa * noa  # aaaa
        self.problem_dim += nva * nvb * noa * nob  # abab
        self.problem_dim += nvb * nvb * nob * nob  # bbbb
        # Extra terms
        self.problem_dim += nva * nvb * nob * noa  # abba
        self.problem_dim += nvb * nva * noa * nob  # baab
        self.problem_dim += nvb * nva * nob * noa  # baba

        # vectors are stored as columns of the matrices
        # access the last vector with matrix[:, -1]
        self.residuals_matrix = np.zeros(shape=(self.problem_dim, 0))
        self.guesses_matrix = np.zeros(shape=(self.problem_dim, 0))

    def find_next_guess(
        self,
        guess: dict[str, NDArray],
        residual: dict[str, NDArray],
    ) -> dict[str, NDArray]:

        self.update_state(residual, guess)

        if self.current_iteration < self.START_DIIS_AT_ITER:
            return guess

        matrix, rhs = self.build_linear_problem()
        np.set_printoptions(precision=3)
        c = np.linalg.solve(matrix, rhs)
        coefficients = c[:-1, :]
        assert self.guesses_matrix.shape[1] == coefficients.shape[0]
        new_guess = self.guesses_matrix @ coefficients
        return self._unflatten(new_guess)

    def update_state(
        self,
        new_residual: dict[str, NDArray],
        new_guess: dict[str, NDArray],
    ) -> None:
        self.current_iteration += 1
        self.residuals_matrix = self._add_new_vector(
            self.residuals_matrix, new_residual
        )
        self.guesses_matrix = self._add_new_vector(
            self.guesses_matrix, new_guess
        )

    def _add_new_vector(
        self,
        vectors_store: NDArray,
        new_vector: dict[str, NDArray]
    ) -> NDArray:
        """ Flattens the new_vector and adds it to the storage. """

        if vectors_store.shape[1] == self.STORAGE_SIZE:
            current_store = vectors_store[:, 1:]
        else:
            current_store = vectors_store

        flattened_vector = self._flatten(new_vector)
        vectors_store = np.hstack(
            (current_store, flattened_vector)
        )
        return vectors_store

    def _flatten(self, new_vector: dict[str, NDArray]) -> NDArray:
        noa = self.noa
        nob = self.nob
        nva = self.nva
        nvb = self.nvb
        flattened_vector = np.hstack(
            (
                new_vector['aa'].reshape(nva * noa),
                new_vector['bb'].reshape(nvb * nob),
                new_vector['aaaa'].reshape(nva * nva * noa * noa),
                new_vector['abab'].reshape(nva * nvb * noa * nob),
                new_vector['bbbb'].reshape(nvb * nvb * nob * nob),
                # spin-changing terms
                new_vector['abba'].reshape(nva * nvb * nob * noa),
                new_vector['baab'].reshape(nvb * nva * noa * nob), 
                new_vector['baba'].reshape(nvb * nva * nob * noa),
            )
        ).reshape(-1, 1)
        return flattened_vector

    def _unflatten(self, vector: NDArray) -> dict[str, NDArray]:
        """ Revert the `_flatten`, i.e., return something like this:
        ```
        {
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

        blocks = [
            'aa', 'bb', 'aaaa', 'abab', 'bbbb',
            # redundant terms
            'abba', 'baab', 'baba',
        ]

        shapes = {
            'aa': (nva, noa),
            'bb': (nvb, nob),
            'aaaa': (nva, nva, noa, noa),
            'abab': (nva, nvb, noa, nob),
            'bbbb': (nvb, nvb, nob, nob),
            'abba': (nva, nvb, nob, noa),
            'baab': (nvb, nva, noa, nob),
            'baba': (nvb, nva, nob, noa),
        }

        dims = {
            block: functools.reduce(lambda x, y: x * y, shapes[block],1)
            for block in blocks
        }

        dim_sum = 0
        slices = {}
        for block in blocks:
            block_dim = dims[block]
            slices[block] = slice(dim_sum, dim_sum + block_dim)
            dim_sum += block_dim

        assert len(vector.shape) == 2
        assert vector.shape[0] == dim_sum
        assert vector.shape[1] == 1

        unflatten = {
            block: vector[slices[block]].reshape(shapes[block])
            for block in blocks
        }

        return unflatten

    def build_linear_problem(self) -> tuple[NDArray, NDArray]:
        b_matrix_dim = self.residuals_matrix.shape[1]
        matrix = self.residuals_matrix.T @ self.residuals_matrix
        matrix = np.vstack((
            matrix,
            -1 * np.ones((1, b_matrix_dim))
        ))
        matrix = np.hstack((
            matrix, -1 * np.ones((b_matrix_dim + 1, 1))
        ))
        matrix[-1][-1] = 0.0
        rhs = np.zeros((b_matrix_dim + 1, 1))
        rhs[-1, 0] = -1
        return matrix, rhs


class UHF_CCSD:

    def __init__(
        self,
        scf_data: Intermediates,
        shift_1e: float = 0.0,
        shift_2e: float = 0.0,
        use_diis: bool = True,
    ) -> None:
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

        self.dampers = self.build_dampers(shift_1e=shift_1e, shift_2e=shift_2e)

        self.cc_solved = False
        self.lambda_cc_solved = False
        
        if use_diis is True:
            self.diis = DIIS(noa, nva, nob, nvb)
            # self.diis = Alt_DIIS(noa, nva, nob, nvb)
        else:
            self.diis = None

        # UI
        self.verbose = 0


    def solve_cc_equations(self):
        MAX_CCSD_ITER = 50
        ENERGY_CONVERGENCE = 1e-6
        RESIDUALS_CONVERGENCE = 1e-6

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

    def solve_lambda_equations(self):
        MAX_CCSD_ITER = 25
        RESIDUALS_CONVERGENCE = 1e-6
        CC_ENERGY = self.get_energy()
        if self.cc_solved is False:
            self.solve_cc_equations()

        self.initialize_lambda()

        for iter_idx in range(MAX_CCSD_ITER):
            residuals = self.calculate_lambda_residuals(CC_ENERGY)
            new_lambdas = self.calculate_new_lambdas(residuals)
            self.update_lambdas(new_lambdas)
            residuals_norm = float(self.get_residuals_norm(residuals))
            self.print_lambda_iteration_report(iter_idx, residuals_norm)
            residuals_converged = residuals_norm < RESIDUALS_CONVERGENCE
            if residuals_converged:
                break
        else:
            raise RuntimeError("Lambda-UHF_CCSD didn't converge.")
        self.lambda_cc_solved = True

    def initialize_lambda(self):
        if self.data.lmbda is not None:
            msg = "Initializing UHF CCSD Lambda, while it is already there."
            raise RuntimeError(msg)

        self.data.lmbda = UHF_CCSD_Lambda_Data(
            l1_aa=self.data.t1_aa.copy().transpose((1, 0)),
            l1_bb=self.data.t1_bb.copy().transpose((1, 0)),
            l2_aaaa=self.data.t2_aaaa.copy().transpose((2, 3, 0, 1)),
            l2_abab=self.data.t2_abab.copy().transpose((2, 3, 0, 1)),
            l2_bbbb=self.data.t2_bbbb.copy().transpose((2, 3, 0, 1)),
        )

        import sys 
        print(f'{self.data.lmbda.l1_aa.shape=}', file=sys.stderr)
        print(f'{self.data.lmbda.l2_aaaa.shape=}', file=sys.stderr)

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
        print(f"Iteration {iter_idx + 1:>2d}:", end='')
        print(f' {current_energy:{e_fmt}}', end='')
        print(f' {energy_change:{e_fmt}}', end='')
        print(f' {residuals_norm:{e_fmt}}', end='')
        if self.diis is not None:
            if iter_idx + 1 >= self.diis.START_DIIS_AT_ITER:
                print(' DIIS', end='')
        print('')

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

    def calculate_lambda_residuals(self, CC_ENERGY: float):
        residuals = dict()

        kwargs = GeneratorsInput(
            uhf_scf_data=self.scf_data,
            uhf_ccsd_data=self.data,
        )
        
        if self.data.lmbda is None:
            raise RuntimeError("UHF CCSD Lambda uninitialized.")

        # I don't know how to do this in pdaggerq better, the rhs just has the
        # energy times a coefficients that I subtract here
        residuals['aa'] = ( get_lambda_singles_res_aa(**kwargs) 
            - CC_ENERGY * self.data.lmbda.l1_aa )
        residuals['bb'] = ( get_lambda_singles_res_bb(**kwargs)
            - CC_ENERGY * self.data.lmbda.l1_bb )

        residuals['aaaa'] = ( get_lambda_doubles_res_aaaa(**kwargs)
            - CC_ENERGY * self.data.lmbda.l2_aaaa )
        residuals['abab'] = ( get_lambda_doubles_res_abab(**kwargs)
            - CC_ENERGY * self.data.lmbda.l2_abab )
        residuals['bbbb'] = ( get_lambda_doubles_res_bbbb(**kwargs)
            - CC_ENERGY * self.data.lmbda.l2_bbbb )

        return residuals

    def calculate_new_lambdas(self, residuals):
        new_lambdas = dict()
        if self.data.lmbda is None:
            raise RuntimeError("UHF CCSD Lambda uninitialized.")

        lmbda = self.data.lmbda
        dampers = self.dampers

        new_lambdas['aa'] = ( lmbda.l1_aa 
             + residuals['aa'] * dampers['aa'].transpose((1, 0)) )
        new_lambdas['bb'] = ( lmbda.l1_bb 
             + residuals['bb'] * dampers['bb'].transpose((1, 0)) )

        new_lambdas['aaaa'] = (
            lmbda.l2_aaaa 
            + residuals['aaaa'] * dampers['aaaa'].transpose((2, 3, 0, 1)) )
        new_lambdas['abab'] = (
            lmbda.l2_abab
            + residuals['abab'] * dampers['abab'].transpose((2, 3, 0, 1)) )
        new_lambdas['bbbb'] = (
            lmbda.l2_bbbb
            + residuals['bbbb'] * dampers['bbbb'].transpose((2, 3, 0, 1)) )

        return new_lambdas

    def update_lambdas(self, new_lambdas):
        if self.data.lmbda is None:
            raise RuntimeError("UHF CCSD Lambda uninitialized.")
        
        lmbda = self.data.lmbda
        lmbda.l1_aa = new_lambdas['aa']
        lmbda.l1_bb = new_lambdas['bb']
        lmbda.l2_aaaa = new_lambdas['aaaa']
        lmbda.l2_abab = new_lambdas['abab']
        lmbda.l2_bbbb = new_lambdas['bbbb']

    def print_lambda_iteration_report(
            self, iter_idx: int, residuals_norm: float,
    ):
        if self.verbose == 0:
            return

        e_fmt = '12.6f'
        print(f"Iteration {iter_idx + 1:>2d}:", end='')
        print(f' {residuals_norm:{e_fmt}}', end='')
        # if self.diis is not None:
        #     if iter_idx + 1 >= self.diis.START_DIIS_AT_ITER:
        #         print(' DIIS', end='')
        print('')

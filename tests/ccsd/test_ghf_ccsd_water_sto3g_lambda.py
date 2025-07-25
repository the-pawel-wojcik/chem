from chem.ccsd.ghf_ccsd import GHF_CCSD, GHF_CCSD_Config
from chem.hf.electronic_structure import ResultHF
from chem.hf.ghf_data import GHF_Data, wfn_to_GHF_Data
import numpy as np
import pytest


@pytest.fixture(scope='session')
def ghf_data(water_sto3g: ResultHF) -> GHF_Data:
    return wfn_to_GHF_Data(water_sto3g.wfn)


@pytest.fixture(scope='session')
def nuclear_repulsion_energy(water_sto3g: ResultHF) -> float:
    nuclear_repulsion_energy = water_sto3g.molecule.nuclear_repulsion_energy()
    return nuclear_repulsion_energy


def test_lambda_solver(
    ghf_data: GHF_Data,
    nuclear_repulsion_energy: float,
) -> None:
    ccsd = GHF_CCSD(
        ghf_data,
        config=GHF_CCSD_Config(
            verbose=1,
            use_diis=False,
            max_iterations=100,
            energy_convergence=1e-5,
            residuals_convergence=1e-5,
        )
    )
    print()
    print('Solving the GHF-CCSD Equations.')
    ccsd.solve_cc_equations()
    print('GHF-CCSD Converged.')
    ccsd.print_leading_t_amplitudes()
    cc_energy = ccsd.get_energy()
    print(f'Electronic GHF-CCSD energy = {cc_energy:.5f} Ha.')
    total_cc_energy = cc_energy + nuclear_repulsion_energy
    print(f'Total GHF-CCSD energy =      {total_cc_energy:.5f} Ha.')
    print()
    print('Solving the GHF-CCSD Lambda Equations.')
    ccsd.solve_lambda_equations()
    print('GHF-CCSD Lambda equations solved.')
    no_occupations = ccsd._get_no_occupations()
    BENCH_NO_OCCUPATIONS = [
        0.99999901, 0.99999901, 0.99921936, 0.99921936, 0.99896584, 0.99896584,
        0.98659275, 0.98659275, 0.98493343, 0.98493343, 0.01555294, 0.01555294,
        0.01473737, 0.01473737,
    ]
    assert np.allclose(no_occupations, BENCH_NO_OCCUPATIONS, atol=1e-6)
    # compare against RHF-CCSD printout from Psi4
    assert np.isclose(sum(no_occupations[2*2:2*2+2]), 1.998, atol=1e-3)
    assert np.isclose(sum(no_occupations[3*2:3*2+2]), 1.973, atol=1e-3)
    assert np.isclose(sum(no_occupations[4*2:4*2+2]), 1.970, atol=1e-3)
    assert np.isclose(sum(no_occupations[5*2:5*2+2]), 0.031, atol=1e-3)
    assert np.isclose(sum(no_occupations[6*2:6*2+2]), 0.029, atol=1e-3)

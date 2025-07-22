import pytest
from chem.ccsd.ghf_ccsd import GHF_CCSD, GHF_CCSD_Config
from chem.hf.electronic_structure import ResultHF
from chem.hf.ghf_data import GHF_Data, wfn_to_GHF_Data


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
    cc_energy = ccsd.get_energy()
    print(f'Electronic GHF-CCSD energy = {cc_energy:.5f} Ha.')
    total_cc_energy = cc_energy + nuclear_repulsion_energy
    print(f'Total GHF-CCSD energy =      {total_cc_energy:.5f} Ha.')
    print()
    print('Solving the GHF-CCSD Lambda Equations')
    ccsd.solve_lambda_equations()

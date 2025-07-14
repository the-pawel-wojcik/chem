from chem.ccsd.ghf_ccsd import GHF_CCSD, GHF_CCSD_Config
from chem.hf.electronic_structure import ResultHF
from chem.hf.ghf_data import GHF_Data, wfn_to_GHF_Data
import pytest
import numpy as np


@pytest.fixture(scope='session')
def nuclear_repulsion_energy(water_sto3g: ResultHF) -> float:
    nuclear_repulsion_energy = water_sto3g.molecule.nuclear_repulsion_energy()
    return nuclear_repulsion_energy


@pytest.fixture(scope='session')
def ghf_data(water_sto3g: ResultHF) -> GHF_Data:
    return wfn_to_GHF_Data(water_sto3g.wfn)


@pytest.mark.parametrize(
    argnames='ghf_ccsd_config',
    argvalues= [
        pytest.param(GHF_CCSD_Config(verbose=1, use_diis=False), id='no_diis'),
        # pytest.param(GHF_CCSD_Config(verbose=1), id='default'),
    ]
)
def test_ccsd_energy(
    ghf_data: GHF_Data,
    nuclear_repulsion_energy: float,
    ghf_ccsd_config: GHF_CCSD_Config,
) -> None:
    ccsd = GHF_CCSD(ghf_data, ghf_ccsd_config)
    ccsd.solve_cc_equations()
    ghf_ccsd_energy = ccsd.get_energy()
    ghf_ccsd_total_energy = ghf_ccsd_energy + nuclear_repulsion_energy
    assert np.isclose(ghf_ccsd_energy, -83.9266502349831, atol=1e-5)
    assert np.isclose(ghf_ccsd_total_energy, -75.02028564818042, atol=1e-5)


# def test_ccsd_diis_energy(
#     ghf_data: GHF_Data,
#     nuclear_repulsion_energy: float
# ):
#     ccsd = GHF_CCSD(ghf_data, GHF_CCSD_Config(verbose=1))
#     ccsd.solve_cc_equations()
#     ghf_ccsd_energy = ccsd.get_energy()
#     ghf_ccsd_total_energy = ghf_ccsd_energy + nuclear_repulsion_energy
#     assert np.isclose(ghf_ccsd_energy, -83.9266502349831, atol=1e-5)
#     assert np.isclose(ghf_ccsd_total_energy, -75.02028564818042, atol=1e-5)


def test_lambda_solver(ghf_data: GHF_Data) -> None:
    ccsd = GHF_CCSD(ghf_data, config=GHF_CCSD_Config(verbose=True))
    ccsd.solve_lambda_equations()

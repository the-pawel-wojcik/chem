from chem.ccsd.uhf_ccsd import UHF_CCSD, UHF_CCSD_Config
from chem.hf.electronic_structure import hf, ResultHF
from chem.hf.intermediates_builders import Intermediates, extract_intermediates
import pytest
import numpy as np


@pytest.fixture(scope='session')
def hf_result() -> ResultHF:
    """ Geometry from CCCBDB: HF/STO-3G """
    geometry = """
    0 1
    O1	0.0000   0.0000   0.1272
    H2	0.0000   0.7581  -0.5086
    H3	0.0000  -0.7581  -0.5086
    symmetry c1
    """
    hf_result = hf(geometry=geometry, basis='sto-3g')
    return hf_result


@pytest.fixture(scope='session')
def nuclear_repulsion_energy(hf_result: ResultHF) -> float:
    nuclear_repulsion_energy = hf_result.molecule.nuclear_repulsion_energy()
    return nuclear_repulsion_energy


@pytest.fixture(scope='session')
def intermediates(hf_result) -> Intermediates:
    return extract_intermediates(hf_result.wfn)


@pytest.mark.parametrize(
    argnames='uhf_ccsd_config',
    argvalues= [
        pytest.param(UHF_CCSD_Config(verbose=1, use_diis=False), id='no_diis'),
        pytest.param(UHF_CCSD_Config(verbose=1), id='default'),
    ]
)
def test_ccsd_energy(
    intermediates: Intermediates,
    nuclear_repulsion_energy: float,
    uhf_ccsd_config: UHF_CCSD_Config,
):
    ccsd = UHF_CCSD(intermediates, uhf_ccsd_config)
    ccsd.solve_cc_equations()
    uhf_ccsd_energy = ccsd.get_energy()
    uhf_ccsd_total_energy = uhf_ccsd_energy + nuclear_repulsion_energy
    assert np.isclose(uhf_ccsd_energy, -83.9266502349831, atol=1e-5)
    assert np.isclose(uhf_ccsd_total_energy, -75.02028564818042, atol=1e-5)


def test_ccsd_diis_energy(
    intermediates: Intermediates,
    nuclear_repulsion_energy: float
):
    ccsd = UHF_CCSD(intermediates, UHF_CCSD_Config(verbose=1))
    ccsd.solve_cc_equations()
    uhf_ccsd_energy = ccsd.get_energy()
    uhf_ccsd_total_energy = uhf_ccsd_energy + nuclear_repulsion_energy
    assert np.isclose(uhf_ccsd_energy, -83.9266502349831, atol=1e-5)
    assert np.isclose(uhf_ccsd_total_energy, -75.02028564818042, atol=1e-5)


def test_ccsd_lambda(intermediates: Intermediates):
    ccsd = UHF_CCSD(intermediates)
    ccsd.solve_lambda_equations()  # solves the CC equations first if unsolved

from numpy.linalg import svdvals
from chem.ccsd.ghf_ccsd import GHF_CCSD, GHF_CCSD_Config
from chem.hf.containers import ResultHF
from chem.hf.ghf_data import wfn_to_GHF_Data
from chem.ccsd.equations.ghf.opdm.manual_opdm import get_opdm
import numpy as np
from numpy.linalg._linalg import SVDResult
import pytest


@pytest.fixture(scope='session')
def ghf_ccsd_water_sto3g(water_sto3g: ResultHF) -> GHF_CCSD:
    ghf_ccsd_config = GHF_CCSD_Config(
        verbose=0,
        use_diis=False,
        max_iterations=50,
        energy_convergence=1e-10,
        residuals_convergence=1e-10,
        shift_1e=0.,
        shift_2e=0.,
    )
    ghf_data = wfn_to_GHF_Data(water_sto3g.wfn)
    ccsd = GHF_CCSD(ghf_data, ghf_ccsd_config)
    ccsd.solve_lambda_equations()
    return ccsd


def test_opdm(ghf_ccsd_water_sto3g: GHF_CCSD) -> None:
    ccsd = ghf_ccsd_water_sto3g
    opdm = get_opdm(ccsd.ghf_data, ccsd.data)

    np.set_printoptions(precision=3, suppress=True)
    assert not np.allclose(opdm, opdm.T, atol=1e-4)
    # TODO: is the CC OPDM really expected to be non-symmetric

    # TODO: test these occupations against expected values
    svalues = ccsd._get_no_occupations()
    SVD_PRINT_THRESH = 1e-3
    ccsd.print_no_occupations(threshold=SVD_PRINT_THRESH)

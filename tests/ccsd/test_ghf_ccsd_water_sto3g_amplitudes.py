import pytest
import numpy as np
from chem.ccsd.ghf_ccsd import GHF_CCSD, GHF_CCSD_Config
from chem.hf.electronic_structure import ResultHF
from chem.hf.ghf_data import GHF_Data, wfn_to_GHF_Data


@pytest.fixture(scope='session')
def ghf_data(water_sto3g: ResultHF) -> GHF_Data:
    return wfn_to_GHF_Data(water_sto3g.wfn)


def test_ccsd_print_leading_t_amplitudes(
    ghf_data: GHF_Data,
) -> None:
    ccsd = GHF_CCSD(
        ghf_data,
        GHF_CCSD_Config(
            energy_convergence=1e-10,
            residuals_convergence=1e-10,
            t_amp_print_threshold=1e-2,
        )
    )
    ccsd.solve_cc_equations()
    top_t1 = ccsd._find_leading_t1_amplitudes()
    assert len(top_t1) == 2
    assert top_t1[0]['v'] == 0
    assert top_t1[0]['o'] == 7
    assert np.isclose(top_t1[0]['amp'], 0.014, atol=1e-3)
    assert top_t1[1]['v'] == 1
    assert top_t1[1]['o'] == 6
    assert np.isclose(top_t1[1]['amp'], 0.014, atol=1e-3)

    top_t2 = ccsd._find_leading_t2_amplitudes()

    assert len(top_t2) == 88
    assert top_t2[62]['vl'] == 2
    assert top_t2[62]['vr'] == 3
    assert top_t2[62]['ol'] == 4
    assert top_t2[62]['or'] == 5
    assert np.isclose(top_t2[62]['amp'], +0.085, atol=1e-3)

    assert top_t2[83]['vl'] == 3
    assert top_t2[83]['vr'] == 2
    assert top_t2[83]['ol'] == 5
    assert top_t2[83]['or'] == 4
    assert np.isclose(top_t2[83]['amp'], +0.085, atol=1e-3)

    assert top_t2[63]['vl'] == 2
    assert top_t2[63]['vr'] == 3
    assert top_t2[63]['ol'] == 5
    assert top_t2[63]['or'] == 4
    assert np.isclose(top_t2[63]['amp'], -0.085, atol=1e-3)

    assert top_t2[82]['vl'] == 3
    assert top_t2[82]['vr'] == 2
    assert top_t2[82]['ol'] == 4
    assert top_t2[82]['or'] == 5
    assert np.isclose(top_t2[82]['amp'], -0.085, atol=1e-3)
    ccsd.print_leading_t_amplitudes()


def test_ccsd_print_leading_lambda_amplitudes(
    ghf_data: GHF_Data,
) -> None:
    ccsd = GHF_CCSD(
        ghf_data,
        GHF_CCSD_Config(
            energy_convergence=1e-10,
            residuals_convergence=1e-10,
            t_amp_print_threshold=1e-2,
        )
    )
    ccsd.solve_lambda_equations()
    ccsd.print_leading_lambda_amplitudes()
    top_l1 = ccsd._find_leading_l1_amplitudes()
    assert len(top_l1) == 2
    assert top_l1[0]['o'] == 6
    assert top_l1[0]['v'] == 1
    assert np.isclose(top_l1[0]['amp'], 0.014, atol=1e-3)
    assert top_l1[1]['o'] == 7
    assert top_l1[1]['v'] == 0
    assert np.isclose(top_l1[1]['amp'], 0.014, atol=1e-3)

    top_l2 = ccsd._find_leading_l2_amplitudes()
    assert len(top_l2) == 80

import pytest
from chem.hf.intermediates_builders import extract_intermediates, Intermediates
from chem.hf.electronic_structure import hf, ResultHF
from numpy import einsum
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


def get_uhf_energy(uhf_data: Intermediates) -> float:
    f_aa = uhf_data.f_aa
    f_bb = uhf_data.f_bb
    g_aaaa = uhf_data.g_aaaa
    g_abab = uhf_data.g_abab
    g_bbbb = uhf_data.g_bbbb
    oa = uhf_data.oa
    ob = uhf_data.ob

    #  1.00 f_aa(i,i)
    energy =  1.00 * einsum('ii', f_aa[oa, oa])
    #  1.00 f_bb(i,i)
    energy +=  1.00 * einsum('ii', f_bb[ob, ob])
    # -0.50 <j,i||j,i>_aaaa
    energy += -0.50 * einsum('jiji', g_aaaa[oa, oa, oa, oa])
    # -0.50 <j,i||j,i>_abab
    energy += -0.50 * einsum('jiji', g_abab[oa, ob, oa, ob])
    # -0.50 <i,j||i,j>_abab
    energy += -0.50 * einsum('ijij', g_abab[oa, ob, oa, ob])
    # -0.50 <j,i||j,i>_bbbb
    energy += -0.50 * einsum('jiji', g_bbbb[ob, ob, ob, ob])
    return float(energy)


def test_constructor(hf_result: ResultHF):
    extract_intermediates(hf_result.wfn)


def test_energies(hf_result: ResultHF):
    uhf_data = extract_intermediates(hf_result.wfn)
    fock_diagonal = np.array([
        -20.25157669, -1.25754098, -0.59385141, -0.45972452, -0.3926153,
        0.58177897, 0.69266369,
    ])
    assert np.allclose(uhf_data.f_aa.diagonal(), fock_diagonal, atol=1e-7)
    energy = get_uhf_energy(uhf_data)
    assert np.isclose(energy, -83.87226577852897, atol=1e-7)

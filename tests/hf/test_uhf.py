from chem.hf.intermediates_builders import extract_intermediates, Intermediates
from chem.hf.electronic_structure import ResultHF
from numpy import einsum
import numpy as np


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


def test_constructor(water_sto3g: ResultHF):
    extract_intermediates(water_sto3g.wfn)


def test_energies(water_sto3g: ResultHF):
    uhf_data = extract_intermediates(water_sto3g.wfn)
    fock_diagonal = np.array([
        -20.25157699, -1.25754837, -0.59385451, -0.45972972, -0.39261692,
        0.58179264, 0.69267285,
    ])
    assert np.allclose(uhf_data.f_aa.diagonal(), fock_diagonal, atol=1e-7)
    energy = get_uhf_energy(uhf_data)
    assert np.isclose(energy, -83.87226577852897, atol=1e-7)

from numpy import einsum
from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.ccs.containers import UHF_CCS_Data


def get_ccs_energy(
    uhf_data: Intermediates,
    uhf_ccs_data: UHF_CCS_Data,
) -> NDArray:
    f_aa = uhf_data.f_aa
    f_bb = uhf_data.f_bb
    g_aaaa = uhf_data.g_aaaa
    g_abab = uhf_data.g_abab
    g_bbbb = uhf_data.g_bbbb
    va = uhf_data.va
    vb = uhf_data.vb
    oa = uhf_data.oa
    ob = uhf_data.ob
    t1_aa = uhf_ccs_data.t1_aa
    t1_bb = uhf_ccs_data.t1_bb
    
    ccs_energy =  1.00 * einsum('ii', f_aa[oa, oa])
    ccs_energy +=  1.00 * einsum('ii', f_bb[ob, ob])
    ccs_energy +=  1.00 * einsum('ia,ai', f_aa[oa, va], t1_aa)
    ccs_energy +=  1.00 * einsum('ia,ai', f_bb[ob, vb], t1_bb)
    ccs_energy += -0.50 * einsum('jiji', g_aaaa[oa, oa, oa, oa])
    ccs_energy += -0.50 * einsum('jiji', g_abab[oa, ob, oa, ob])
    ccs_energy += -0.50 * einsum('ijij', g_abab[oa, ob, oa, ob])
    ccs_energy += -0.50 * einsum('jiji', g_bbbb[ob, ob, ob, ob])
    ccs_energy += -0.50 * einsum('jiab,ai,bj', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    ccs_energy +=  0.50 * einsum('ijab,ai,bj', g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    ccs_energy +=  0.50 * einsum('jiba,ai,bj', g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    ccs_energy += -0.50 * einsum('jiab,ai,bj', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    return ccs_energy

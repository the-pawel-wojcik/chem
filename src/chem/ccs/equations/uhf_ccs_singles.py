from numpy import einsum
from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.ccs.containers import UHF_CCS_Data


def get_uhf_ccs_singles_residuals_aa(
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
    
    uhf_ccs_singles_residuals_aa =  1.00 * einsum('ai->ai', f_aa[va, oa])
    uhf_ccs_singles_residuals_aa += -1.00 * einsum('ji,aj->ai', f_aa[oa, oa], t1_aa)
    uhf_ccs_singles_residuals_aa +=  1.00 * einsum('ab,bi->ai', f_aa[va, va], t1_aa)
    uhf_ccs_singles_residuals_aa += -1.00 * einsum('jb,aj,bi->ai', f_aa[oa, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    uhf_ccs_singles_residuals_aa +=  1.00 * einsum('jabi,bj->ai', g_aaaa[oa, va, va, oa], t1_aa)
    uhf_ccs_singles_residuals_aa +=  1.00 * einsum('ajib,bj->ai', g_abab[va, ob, oa, vb], t1_bb)
    uhf_ccs_singles_residuals_aa +=  1.00 * einsum('kjbi,ak,bj->ai', g_aaaa[oa, oa, va, oa], t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    uhf_ccs_singles_residuals_aa += -1.00 * einsum('kjib,ak,bj->ai', g_abab[oa, ob, oa, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    uhf_ccs_singles_residuals_aa +=  1.00 * einsum('jabc,bj,ci->ai', g_aaaa[oa, va, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    uhf_ccs_singles_residuals_aa +=  1.00 * einsum('ajcb,bj,ci->ai', g_abab[va, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    uhf_ccs_singles_residuals_aa +=  1.00 * einsum('kjbc,ak,bj,ci->ai', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    uhf_ccs_singles_residuals_aa += -1.00 * einsum('kjcb,ak,bj,ci->ai', g_abab[oa, ob, va, vb], t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    return uhf_ccs_singles_residuals_aa


def get_uhf_ccs_singles_residuals_bb(
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
    
    uhf_ccs_singles_residuals_bb =  1.00 * einsum('ai->ai', f_bb[vb, ob])
    uhf_ccs_singles_residuals_bb += -1.00 * einsum('ji,aj->ai', f_bb[ob, ob], t1_bb)
    uhf_ccs_singles_residuals_bb +=  1.00 * einsum('ab,bi->ai', f_bb[vb, vb], t1_bb)
    uhf_ccs_singles_residuals_bb += -1.00 * einsum('jb,aj,bi->ai', f_bb[ob, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    uhf_ccs_singles_residuals_bb +=  1.00 * einsum('jabi,bj->ai', g_abab[oa, vb, va, ob], t1_aa)
    uhf_ccs_singles_residuals_bb +=  1.00 * einsum('jabi,bj->ai', g_bbbb[ob, vb, vb, ob], t1_bb)
    uhf_ccs_singles_residuals_bb += -1.00 * einsum('jkbi,ak,bj->ai', g_abab[oa, ob, va, ob], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    uhf_ccs_singles_residuals_bb +=  1.00 * einsum('kjbi,ak,bj->ai', g_bbbb[ob, ob, vb, ob], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    uhf_ccs_singles_residuals_bb +=  1.00 * einsum('jabc,bj,ci->ai', g_abab[oa, vb, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    uhf_ccs_singles_residuals_bb +=  1.00 * einsum('jabc,bj,ci->ai', g_bbbb[ob, vb, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    uhf_ccs_singles_residuals_bb += -1.00 * einsum('jkbc,ak,bj,ci->ai', g_abab[oa, ob, va, vb], t1_bb, t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    uhf_ccs_singles_residuals_bb +=  1.00 * einsum('kjbc,ak,bj,ci->ai', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    return uhf_ccs_singles_residuals_bb

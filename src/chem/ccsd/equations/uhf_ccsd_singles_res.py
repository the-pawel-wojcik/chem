from numpy import einsum
from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.ccsd.uhf_ccsd import UHF_CCSD_Data


def get_singles_residual_aa(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    f_aa = uhf_scf_data.f_aa
    f_bb = uhf_scf_data.f_bb
    g_aaaa = uhf_scf_data.g_aaaa
    g_abab = uhf_scf_data.g_abab
    g_bbbb = uhf_scf_data.g_bbbb
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    singles_res_aa =  1.00 * einsum('ai->ai', f_aa[va, oa])
    singles_res_aa += -1.00 * einsum('ji,aj->ai', f_aa[oa, oa], t1_aa)
    singles_res_aa +=  1.00 * einsum('ab,bi->ai', f_aa[va, va], t1_aa)
    singles_res_aa += -1.00 * einsum('jb,baij->ai', f_aa[oa, va], t2_aaaa)
    singles_res_aa +=  1.00 * einsum('jb,abij->ai', f_bb[ob, vb], t2_abab)
    singles_res_aa += -1.00 * einsum('jb,aj,bi->ai', f_aa[oa, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    singles_res_aa +=  1.00 * einsum('jabi,bj->ai', g_aaaa[oa, va, va, oa], t1_aa)
    singles_res_aa +=  1.00 * einsum('ajib,bj->ai', g_abab[va, ob, oa, vb], t1_bb)
    singles_res_aa += -0.50 * einsum('kjbi,bakj->ai', g_aaaa[oa, oa, va, oa], t2_aaaa)
    singles_res_aa += -0.50 * einsum('kjib,abkj->ai', g_abab[oa, ob, oa, vb], t2_abab)
    singles_res_aa += -0.50 * einsum('jkib,abjk->ai', g_abab[oa, ob, oa, vb], t2_abab)
    singles_res_aa += -0.50 * einsum('jabc,bcij->ai', g_aaaa[oa, va, va, va], t2_aaaa)
    singles_res_aa +=  0.50 * einsum('ajbc,bcij->ai', g_abab[va, ob, va, vb], t2_abab)
    singles_res_aa +=  0.50 * einsum('ajcb,cbij->ai', g_abab[va, ob, va, vb], t2_abab)
    singles_res_aa +=  1.00 * einsum('kjbc,caik,bj->ai', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    singles_res_aa += -1.00 * einsum('kjcb,caik,bj->ai', g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    singles_res_aa +=  1.00 * einsum('jkbc,acik,bj->ai', g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    singles_res_aa += -1.00 * einsum('kjbc,acik,bj->ai', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    singles_res_aa +=  0.50 * einsum('kjbc,cakj,bi->ai', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    singles_res_aa += -0.50 * einsum('kjbc,ackj,bi->ai', g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    singles_res_aa += -0.50 * einsum('jkbc,acjk,bi->ai', g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    singles_res_aa +=  0.50 * einsum('kjbc,aj,bcik->ai', g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    singles_res_aa += -0.50 * einsum('jkbc,aj,bcik->ai', g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    singles_res_aa += -0.50 * einsum('jkcb,aj,cbik->ai', g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    singles_res_aa +=  1.00 * einsum('kjbi,ak,bj->ai', g_aaaa[oa, oa, va, oa], t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    singles_res_aa += -1.00 * einsum('kjib,ak,bj->ai', g_abab[oa, ob, oa, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    singles_res_aa +=  1.00 * einsum('jabc,bj,ci->ai', g_aaaa[oa, va, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    singles_res_aa +=  1.00 * einsum('ajcb,bj,ci->ai', g_abab[va, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    singles_res_aa +=  1.00 * einsum('kjbc,ak,bj,ci->ai', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    singles_res_aa += -1.00 * einsum('kjcb,ak,bj,ci->ai', g_abab[oa, ob, va, vb], t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    return singles_res_aa


def get_singles_residual_bb(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    f_aa = uhf_scf_data.f_aa
    f_bb = uhf_scf_data.f_bb
    g_aaaa = uhf_scf_data.g_aaaa
    g_abab = uhf_scf_data.g_abab
    g_bbbb = uhf_scf_data.g_bbbb
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    
    singles_res_bb =  1.00 * einsum('ai->ai', f_bb[vb, ob])
    singles_res_bb += -1.00 * einsum('ji,aj->ai', f_bb[ob, ob], t1_bb)
    singles_res_bb +=  1.00 * einsum('ab,bi->ai', f_bb[vb, vb], t1_bb)
    singles_res_bb +=  1.00 * einsum('jb,baji->ai', f_aa[oa, va], t2_abab)
    singles_res_bb += -1.00 * einsum('jb,baij->ai', f_bb[ob, vb], t2_bbbb)
    singles_res_bb += -1.00 * einsum('jb,aj,bi->ai', f_bb[ob, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    singles_res_bb +=  1.00 * einsum('jabi,bj->ai', g_abab[oa, vb, va, ob], t1_aa)
    singles_res_bb +=  1.00 * einsum('jabi,bj->ai', g_bbbb[ob, vb, vb, ob], t1_bb)
    singles_res_bb += -0.50 * einsum('kjbi,bakj->ai', g_abab[oa, ob, va, ob], t2_abab)
    singles_res_bb += -0.50 * einsum('jkbi,bajk->ai', g_abab[oa, ob, va, ob], t2_abab)
    singles_res_bb += -0.50 * einsum('kjbi,bakj->ai', g_bbbb[ob, ob, vb, ob], t2_bbbb)
    singles_res_bb +=  0.50 * einsum('jabc,bcji->ai', g_abab[oa, vb, va, vb], t2_abab)
    singles_res_bb +=  0.50 * einsum('jacb,cbji->ai', g_abab[oa, vb, va, vb], t2_abab)
    singles_res_bb += -0.50 * einsum('jabc,bcij->ai', g_bbbb[ob, vb, vb, vb], t2_bbbb)
    singles_res_bb += -1.00 * einsum('kjbc,caki,bj->ai', g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    singles_res_bb +=  1.00 * einsum('kjcb,caki,bj->ai', g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    singles_res_bb += -1.00 * einsum('jkbc,caik,bj->ai', g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    singles_res_bb +=  1.00 * einsum('kjbc,caik,bj->ai', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    singles_res_bb += -0.50 * einsum('kjcb,cakj,bi->ai', g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    singles_res_bb += -0.50 * einsum('jkcb,cajk,bi->ai', g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    singles_res_bb +=  0.50 * einsum('kjbc,cakj,bi->ai', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    singles_res_bb += -0.50 * einsum('kjbc,aj,bcki->ai', g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    singles_res_bb += -0.50 * einsum('kjcb,aj,cbki->ai', g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    singles_res_bb +=  0.50 * einsum('kjbc,aj,bcik->ai', g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    singles_res_bb += -1.00 * einsum('jkbi,ak,bj->ai', g_abab[oa, ob, va, ob], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    singles_res_bb +=  1.00 * einsum('kjbi,ak,bj->ai', g_bbbb[ob, ob, vb, ob], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    singles_res_bb +=  1.00 * einsum('jabc,bj,ci->ai', g_abab[oa, vb, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    singles_res_bb +=  1.00 * einsum('jabc,bj,ci->ai', g_bbbb[ob, vb, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    singles_res_bb += -1.00 * einsum('jkbc,ak,bj,ci->ai', g_abab[oa, ob, va, vb], t1_bb, t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    singles_res_bb +=  1.00 * einsum('kjbc,ak,bj,ci->ai', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    return singles_res_bb

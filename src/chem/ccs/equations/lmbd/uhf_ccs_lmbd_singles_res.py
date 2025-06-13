from numpy import einsum
from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.ccs.containers import UHF_CCS_Data, UHF_CCS_Lambda_Data


def get_uhf_ccs_lambda_singles_res_aa(
    uhf_data: Intermediates,
    uhf_ccs_data: UHF_CCS_Data,
    uhf_ccs_lambda_data: UHF_CCS_Lambda_Data,
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
    l1_aa = uhf_ccs_lambda_data.l1_aa
    l1_bb = uhf_ccs_lambda_data.l1_bb
    
    uhf_ccs_lambda_singles_res_aa =  1.00 * einsum('ia->ia', f_aa[oa, va])
    uhf_ccs_lambda_singles_res_aa +=  1.00 * einsum('jj,ia->ia', f_aa[oa, oa], l1_aa)
    uhf_ccs_lambda_singles_res_aa +=  1.00 * einsum('jj,ia->ia', f_bb[ob, ob], l1_aa)
    uhf_ccs_lambda_singles_res_aa += -1.00 * einsum('ij,ja->ia', f_aa[oa, oa], l1_aa)
    uhf_ccs_lambda_singles_res_aa +=  1.00 * einsum('ba,ib->ia', f_aa[va, va], l1_aa)
    uhf_ccs_lambda_singles_res_aa +=  1.00 * einsum('jb,bj,ia->ia', f_aa[oa, va], t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    uhf_ccs_lambda_singles_res_aa +=  1.00 * einsum('jb,bj,ia->ia', f_bb[ob, vb], t1_bb, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    uhf_ccs_lambda_singles_res_aa += -1.00 * einsum('ib,bj,ja->ia', f_aa[oa, va], t1_aa, l1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    uhf_ccs_lambda_singles_res_aa += -1.00 * einsum('ja,bj,ib->ia', f_aa[oa, va], t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    uhf_ccs_lambda_singles_res_aa += -0.50 * einsum('kjkj,ia->ia', g_aaaa[oa, oa, oa, oa], l1_aa)
    uhf_ccs_lambda_singles_res_aa += -0.50 * einsum('kjkj,ia->ia', g_abab[oa, ob, oa, ob], l1_aa)
    uhf_ccs_lambda_singles_res_aa += -0.50 * einsum('jkjk,ia->ia', g_abab[oa, ob, oa, ob], l1_aa)
    uhf_ccs_lambda_singles_res_aa += -0.50 * einsum('kjkj,ia->ia', g_bbbb[ob, ob, ob, ob], l1_aa)
    uhf_ccs_lambda_singles_res_aa +=  1.00 * einsum('ibaj,jb->ia', g_aaaa[oa, va, va, oa], l1_aa)
    uhf_ccs_lambda_singles_res_aa +=  1.00 * einsum('ibaj,jb->ia', g_abab[oa, vb, va, ob], l1_bb)
    uhf_ccs_lambda_singles_res_aa += -1.00 * einsum('ijba,bj->ia', g_aaaa[oa, oa, va, va], t1_aa)
    uhf_ccs_lambda_singles_res_aa +=  1.00 * einsum('ijab,bj->ia', g_abab[oa, ob, va, vb], t1_bb)
    uhf_ccs_lambda_singles_res_aa +=  1.00 * einsum('ikbj,bk,ja->ia', g_aaaa[oa, oa, va, oa], t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    uhf_ccs_lambda_singles_res_aa += -1.00 * einsum('ikjb,bk,ja->ia', g_abab[oa, ob, oa, vb], t1_bb, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    uhf_ccs_lambda_singles_res_aa += -1.00 * einsum('ikaj,bk,jb->ia', g_aaaa[oa, oa, va, oa], t1_aa, l1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    uhf_ccs_lambda_singles_res_aa += -1.00 * einsum('ikaj,bk,jb->ia', g_abab[oa, ob, va, ob], t1_bb, l1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    uhf_ccs_lambda_singles_res_aa +=  1.00 * einsum('jbca,cj,ib->ia', g_aaaa[oa, va, va, va], t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    uhf_ccs_lambda_singles_res_aa +=  1.00 * einsum('bjac,cj,ib->ia', g_abab[va, ob, va, vb], t1_bb, l1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    uhf_ccs_lambda_singles_res_aa += -1.00 * einsum('ibca,cj,jb->ia', g_aaaa[oa, va, va, va], t1_aa, l1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    uhf_ccs_lambda_singles_res_aa +=  1.00 * einsum('ibac,cj,jb->ia', g_abab[oa, vb, va, vb], t1_bb, l1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    uhf_ccs_lambda_singles_res_aa += -0.50 * einsum('kjbc,bj,ck,ia->ia', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    uhf_ccs_lambda_singles_res_aa +=  0.50 * einsum('jkbc,bj,ck,ia->ia', g_abab[oa, ob, va, vb], t1_aa, t1_bb, l1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    uhf_ccs_lambda_singles_res_aa +=  0.50 * einsum('kjcb,bj,ck,ia->ia', g_abab[oa, ob, va, vb], t1_bb, t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    uhf_ccs_lambda_singles_res_aa += -0.50 * einsum('kjbc,bj,ck,ia->ia', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, l1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    uhf_ccs_lambda_singles_res_aa +=  1.00 * einsum('ikbc,bk,cj,ja->ia', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    uhf_ccs_lambda_singles_res_aa += -1.00 * einsum('ikcb,bk,cj,ja->ia', g_abab[oa, ob, va, vb], t1_bb, t1_aa, l1_aa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    uhf_ccs_lambda_singles_res_aa +=  1.00 * einsum('kjca,bk,cj,ib->ia', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, l1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    uhf_ccs_lambda_singles_res_aa += -1.00 * einsum('kjac,bk,cj,ib->ia', g_abab[oa, ob, va, vb], t1_aa, t1_bb, l1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    uhf_ccs_lambda_singles_res_aa +=  1.00 * einsum('ikca,bk,cj,jb->ia', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, l1_aa, optimize=['einsum_path', (2, 3), (1, 2), (0, 1)])
    uhf_ccs_lambda_singles_res_aa += -1.00 * einsum('ikac,bk,cj,jb->ia', g_abab[oa, ob, va, vb], t1_bb, t1_bb, l1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    return uhf_ccs_lambda_singles_res_aa


def get_uhf_ccs_lambda_singles_res_bb(
    uhf_data: Intermediates,
    uhf_ccs_data: UHF_CCS_Data,
    uhf_ccs_lambda_data: UHF_CCS_Lambda_Data,
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
    l1_aa = uhf_ccs_lambda_data.l1_aa
    l1_bb = uhf_ccs_lambda_data.l1_bb
    
    uhf_ccs_lambda_singles_res_bb =  1.00 * einsum('ia->ia', f_bb[ob, vb])
    uhf_ccs_lambda_singles_res_bb +=  1.00 * einsum('jj,ia->ia', f_aa[oa, oa], l1_bb)
    uhf_ccs_lambda_singles_res_bb +=  1.00 * einsum('jj,ia->ia', f_bb[ob, ob], l1_bb)
    uhf_ccs_lambda_singles_res_bb += -1.00 * einsum('ij,ja->ia', f_bb[ob, ob], l1_bb)
    uhf_ccs_lambda_singles_res_bb +=  1.00 * einsum('ba,ib->ia', f_bb[vb, vb], l1_bb)
    uhf_ccs_lambda_singles_res_bb +=  1.00 * einsum('jb,bj,ia->ia', f_aa[oa, va], t1_aa, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    uhf_ccs_lambda_singles_res_bb +=  1.00 * einsum('jb,bj,ia->ia', f_bb[ob, vb], t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    uhf_ccs_lambda_singles_res_bb += -1.00 * einsum('ib,bj,ja->ia', f_bb[ob, vb], t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    uhf_ccs_lambda_singles_res_bb += -1.00 * einsum('ja,bj,ib->ia', f_bb[ob, vb], t1_bb, l1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    uhf_ccs_lambda_singles_res_bb += -0.50 * einsum('kjkj,ia->ia', g_aaaa[oa, oa, oa, oa], l1_bb)
    uhf_ccs_lambda_singles_res_bb += -0.50 * einsum('kjkj,ia->ia', g_abab[oa, ob, oa, ob], l1_bb)
    uhf_ccs_lambda_singles_res_bb += -0.50 * einsum('jkjk,ia->ia', g_abab[oa, ob, oa, ob], l1_bb)
    uhf_ccs_lambda_singles_res_bb += -0.50 * einsum('kjkj,ia->ia', g_bbbb[ob, ob, ob, ob], l1_bb)
    uhf_ccs_lambda_singles_res_bb +=  1.00 * einsum('bija,jb->ia', g_abab[va, ob, oa, vb], l1_aa)
    uhf_ccs_lambda_singles_res_bb +=  1.00 * einsum('ibaj,jb->ia', g_bbbb[ob, vb, vb, ob], l1_bb)
    uhf_ccs_lambda_singles_res_bb +=  1.00 * einsum('jiba,bj->ia', g_abab[oa, ob, va, vb], t1_aa)
    uhf_ccs_lambda_singles_res_bb += -1.00 * einsum('ijba,bj->ia', g_bbbb[ob, ob, vb, vb], t1_bb)
    uhf_ccs_lambda_singles_res_bb += -1.00 * einsum('kibj,bk,ja->ia', g_abab[oa, ob, va, ob], t1_aa, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    uhf_ccs_lambda_singles_res_bb +=  1.00 * einsum('ikbj,bk,ja->ia', g_bbbb[ob, ob, vb, ob], t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    uhf_ccs_lambda_singles_res_bb += -1.00 * einsum('kija,bk,jb->ia', g_abab[oa, ob, oa, vb], t1_aa, l1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    uhf_ccs_lambda_singles_res_bb += -1.00 * einsum('ikaj,bk,jb->ia', g_bbbb[ob, ob, vb, ob], t1_bb, l1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    uhf_ccs_lambda_singles_res_bb +=  1.00 * einsum('jbca,cj,ib->ia', g_abab[oa, vb, va, vb], t1_aa, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    uhf_ccs_lambda_singles_res_bb +=  1.00 * einsum('jbca,cj,ib->ia', g_bbbb[ob, vb, vb, vb], t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    uhf_ccs_lambda_singles_res_bb +=  1.00 * einsum('bica,cj,jb->ia', g_abab[va, ob, va, vb], t1_aa, l1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
    uhf_ccs_lambda_singles_res_bb += -1.00 * einsum('ibca,cj,jb->ia', g_bbbb[ob, vb, vb, vb], t1_bb, l1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
    uhf_ccs_lambda_singles_res_bb += -0.50 * einsum('kjbc,bj,ck,ia->ia', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, l1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    uhf_ccs_lambda_singles_res_bb +=  0.50 * einsum('jkbc,bj,ck,ia->ia', g_abab[oa, ob, va, vb], t1_aa, t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    uhf_ccs_lambda_singles_res_bb +=  0.50 * einsum('kjcb,bj,ck,ia->ia', g_abab[oa, ob, va, vb], t1_bb, t1_aa, l1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    uhf_ccs_lambda_singles_res_bb += -0.50 * einsum('kjbc,bj,ck,ia->ia', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    uhf_ccs_lambda_singles_res_bb += -1.00 * einsum('kibc,bk,cj,ja->ia', g_abab[oa, ob, va, vb], t1_aa, t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    uhf_ccs_lambda_singles_res_bb +=  1.00 * einsum('ikbc,bk,cj,ja->ia', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, l1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    uhf_ccs_lambda_singles_res_bb += -1.00 * einsum('jkca,bk,cj,ib->ia', g_abab[oa, ob, va, vb], t1_bb, t1_aa, l1_bb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    uhf_ccs_lambda_singles_res_bb +=  1.00 * einsum('kjca,bk,cj,ib->ia', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, l1_bb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    uhf_ccs_lambda_singles_res_bb += -1.00 * einsum('kica,bk,cj,jb->ia', g_abab[oa, ob, va, vb], t1_aa, t1_aa, l1_aa, optimize=['einsum_path', (2, 3), (1, 2), (0, 1)])
    uhf_ccs_lambda_singles_res_bb +=  1.00 * einsum('ikca,bk,cj,jb->ia', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, l1_bb, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    return uhf_ccs_lambda_singles_res_bb

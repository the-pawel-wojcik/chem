from numpy import einsum
from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.ccsd.containers import UHF_CCSD_Data


def get_energy(
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
    
    uhf_ccsd_energy =  1.00 * einsum('ii', f_aa[oa, oa])
    uhf_ccsd_energy +=  1.00 * einsum('ii', f_bb[ob, ob])
    uhf_ccsd_energy +=  1.00 * einsum('ia,ai', f_aa[oa, va], t1_aa)
    uhf_ccsd_energy +=  1.00 * einsum('ia,ai', f_bb[ob, vb], t1_bb)
    uhf_ccsd_energy += -0.50 * einsum('jiji', g_aaaa[oa, oa, oa, oa])
    uhf_ccsd_energy += -0.50 * einsum('jiji', g_abab[oa, ob, oa, ob])
    uhf_ccsd_energy += -0.50 * einsum('ijij', g_abab[oa, ob, oa, ob])
    uhf_ccsd_energy += -0.50 * einsum('jiji', g_bbbb[ob, ob, ob, ob])
    uhf_ccsd_energy +=  0.250 * einsum('jiab,abji', g_aaaa[oa, oa, va, va], t2_aaaa)
    uhf_ccsd_energy +=  0.250 * einsum('jiab,abji', g_abab[oa, ob, va, vb], t2_abab)
    uhf_ccsd_energy +=  0.250 * einsum('ijab,abij', g_abab[oa, ob, va, vb], t2_abab)
    uhf_ccsd_energy +=  0.250 * einsum('jiba,baji', g_abab[oa, ob, va, vb], t2_abab)
    uhf_ccsd_energy +=  0.250 * einsum('ijba,baij', g_abab[oa, ob, va, vb], t2_abab)
    uhf_ccsd_energy +=  0.250 * einsum('jiab,abji', g_bbbb[ob, ob, vb, vb], t2_bbbb)
    uhf_ccsd_energy += -0.50 * einsum('jiab,ai,bj', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    uhf_ccsd_energy +=  0.50 * einsum('ijab,ai,bj', g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    uhf_ccsd_energy +=  0.50 * einsum('jiba,ai,bj', g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    uhf_ccsd_energy += -0.50 * einsum('jiab,ai,bj', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    return uhf_ccsd_energy

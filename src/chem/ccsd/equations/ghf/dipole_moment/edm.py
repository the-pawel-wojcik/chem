from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.ghf_ccsd import GHF_CCSD_Data
from chem.meta.coordinates import Descartes


def get_mux(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: () """
    h = ghf_data.mu[Descartes.x]
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    t2 = ghf_ccsd_data.t2
    if ghf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in GHF_CCSD_Data")
    l1 = ghf_ccsd_data.lmbda.l1
    l2 = ghf_ccsd_data.lmbda.l2
    
    mux =  1.00 * einsum('ii', h[o, o])
    mux +=  1.00 * einsum('ai,ia', h[v, o], l1)
    mux +=  1.00 * einsum('ia,ai', h[o, v], t1)
    mux += -1.00 * einsum('ji,aj,ia', h[o, o], t1, l1, optimize=['einsum_path', (0, 1), (0, 1)])
    mux +=  1.00 * einsum('ab,bi,ia', h[v, v], t1, l1, optimize=['einsum_path', (0, 1), (0, 1)])
    mux += -1.00 * einsum('jb,baij,ia', h[o, v], t2, l1, optimize=['einsum_path', (0, 1), (0, 1)])
    mux += -0.50 * einsum('kj,baik,ijba', h[o, o], t2, l2, optimize=['einsum_path', (1, 2), (0, 1)])
    mux +=  0.50 * einsum('bc,caij,ijba', h[v, v], t2, l2, optimize=['einsum_path', (1, 2), (0, 1)])
    mux += -0.50 * einsum('kc,baik,cj,ijba', h[o, v], t2, t1, l2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    mux += -0.50 * einsum('kc,caij,bk,ijba', h[o, v], t2, t1, l2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    mux += -1.00 * einsum('jb,aj,bi,ia', h[o, v], t1, t1, l1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    return mux


def get_muy(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: () """
    h = ghf_data.mu[Descartes.y]
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    t2 = ghf_ccsd_data.t2
    if ghf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in GHF_CCSD_Data")
    l1 = ghf_ccsd_data.lmbda.l1
    l2 = ghf_ccsd_data.lmbda.l2
    
    muy =  1.00 * einsum('ii', h[o, o])
    muy +=  1.00 * einsum('ai,ia', h[v, o], l1)
    muy +=  1.00 * einsum('ia,ai', h[o, v], t1)
    muy += -1.00 * einsum('ji,aj,ia', h[o, o], t1, l1, optimize=['einsum_path', (0, 1), (0, 1)])
    muy +=  1.00 * einsum('ab,bi,ia', h[v, v], t1, l1, optimize=['einsum_path', (0, 1), (0, 1)])
    muy += -1.00 * einsum('jb,baij,ia', h[o, v], t2, l1, optimize=['einsum_path', (0, 1), (0, 1)])
    muy += -0.50 * einsum('kj,baik,ijba', h[o, o], t2, l2, optimize=['einsum_path', (1, 2), (0, 1)])
    muy +=  0.50 * einsum('bc,caij,ijba', h[v, v], t2, l2, optimize=['einsum_path', (1, 2), (0, 1)])
    muy += -0.50 * einsum('kc,baik,cj,ijba', h[o, v], t2, t1, l2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    muy += -0.50 * einsum('kc,caij,bk,ijba', h[o, v], t2, t1, l2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    muy += -1.00 * einsum('jb,aj,bi,ia', h[o, v], t1, t1, l1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    return muy


def get_muz(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: () """
    h = ghf_data.mu[Descartes.z]
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    t2 = ghf_ccsd_data.t2
    if ghf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in GHF_CCSD_Data")
    l1 = ghf_ccsd_data.lmbda.l1
    l2 = ghf_ccsd_data.lmbda.l2
    
    muz =  1.00 * einsum('ii', h[o, o])
    muz +=  1.00 * einsum('ai,ia', h[v, o], l1)
    muz +=  1.00 * einsum('ia,ai', h[o, v], t1)
    muz += -1.00 * einsum('ji,aj,ia', h[o, o], t1, l1, optimize=['einsum_path', (0, 1), (0, 1)])
    muz +=  1.00 * einsum('ab,bi,ia', h[v, v], t1, l1, optimize=['einsum_path', (0, 1), (0, 1)])
    muz += -1.00 * einsum('jb,baij,ia', h[o, v], t2, l1, optimize=['einsum_path', (0, 1), (0, 1)])
    muz += -0.50 * einsum('kj,baik,ijba', h[o, o], t2, l2, optimize=['einsum_path', (1, 2), (0, 1)])
    muz +=  0.50 * einsum('bc,caij,ijba', h[v, v], t2, l2, optimize=['einsum_path', (1, 2), (0, 1)])
    muz += -0.50 * einsum('kc,baik,cj,ijba', h[o, v], t2, t1, l2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    muz += -0.50 * einsum('kc,caij,bk,ijba', h[o, v], t2, t1, l2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    muz += -1.00 * einsum('jb,aj,bi,ia', h[o, v], t1, t1, l1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    return muz

import numpy as np
from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.ghf_ccsd import GHF_CCSD_Data


def get_opdm(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('t', 'u') """
    kd =  ghf_data.identity_singles
    nmo = ghf_data.nmo
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    t2 = ghf_ccsd_data.t2
    if ghf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in GHF_CCSD_Data")
    l1 = ghf_ccsd_data.lmbda.l1
    l2 = ghf_ccsd_data.lmbda.l2
    
    opdm = np.zeros(shape=(nmo, nmo))
    opdm[o, o] +=  1.00 * einsum('ij', kd[o, o])
    opdm[o, v] +=  1.00 * einsum('ia->ia', l1)
    opdm[v, o] +=  1.00 * einsum('ai->ai', t1)
    opdm[o, o] += -1.00 * einsum('ai,ja->ij', t1, l1)
    opdm[v, v] +=  1.00 * einsum('bi,ia->ba', t1, l1)
    opdm[v, o] += -1.00 * einsum('baij,ia->bj', t2, l1)
    opdm[o, o] += -0.50 * einsum('baij,ikba->jk', t2, l2)
    opdm[v, v] +=  0.50 * einsum('caij,ijba->cb', t2, l2)
    opdm[v, o] += -0.50 * einsum(
        'baik,cj,ijba->ck', t2, t1, l2,
        optimize=['einsum_path', (0, 2), (0, 1)]
    )
    opdm[v, o] += -0.50 * einsum(
        'caij,bk,ijba->ck', t2, t1, l2,
        optimize=['einsum_path', (0, 2), (0, 1)]
    )
    opdm[o, v] += -1.00 * einsum(
        'aj,bi,ia->jb', t1, t1, l1,
        optimize=['einsum_path', (0, 2), (0, 1)]
    )
    return opdm

from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.ghf_ccsd import GHF_CCSD_Data
from chem.meta.coordinates import Descartes


def get_opdm(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    """ tensor_subscripts: ('t', 'u') """
    kd =  ghf_data.identity_singles
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    t2 = ghf_ccsd_data.t2
    if ghf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in GHF_CCSD_Data")
    l1 = ghf_ccsd_data.lmbda.l1
    l2 = ghf_ccsd_data.lmbda.l2
    
    opdm =  1.00 * einsum('ij', kd[o, o])
    opdm +=  1.00 * einsum('ia', l1)
    opdm +=  1.00 * einsum('ai', t1)
    opdm += -1.00 * einsum('ai,ja', t1, l1)
    opdm +=  1.00 * einsum('bi,ia', t1, l1)
    opdm += -1.00 * einsum('baij,ia', t2, l1)
    opdm += -0.50 * einsum('baij,ikba', t2, l2)
    opdm +=  0.50 * einsum('caij,ijba', t2, l2)
    opdm += -0.50 * einsum('baik,cj,ijba', t2, t1, l2, optimize=['einsum_path', (0, 2), (0, 1)])
    opdm += -0.50 * einsum('caij,bk,ijba', t2, t1, l2, optimize=['einsum_path', (0, 2), (0, 1)])
    opdm += -1.00 * einsum('aj,bi,ia', t1, t1, l1, optimize=['einsum_path', (0, 2), (0, 1)])
    return opdm

from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.containers import GHF_CCSD_Data


def get_ghf_ccsd_energy(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    f = ghf_data.f
    g = ghf_data.g
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    t2 = ghf_ccsd_data.t2
    
    ghf_ccsd_energy = 1.00 * einsum('ii', f[o, o])
    ghf_ccsd_energy += 1.00 * einsum('ia,ai', f[o, v], t1)
    ghf_ccsd_energy += -0.50 * einsum('jiji', g[o, o, o, o])
    ghf_ccsd_energy += 0.250 * einsum('jiab,abji', g[o, o, v, v], t2)
    ghf_ccsd_energy += -0.50 * einsum('jiab,ai,bj', g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    return ghf_ccsd_energy

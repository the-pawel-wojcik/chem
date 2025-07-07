from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.containers import GHF_CCSD_Data


def get_singles_residual(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    f = ghf_data.f
    g = ghf_data.g
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    t2 = ghf_ccsd_data.t2
    
    singles_residual =  1.00 * einsum('ai->ai', f[v, o])
    singles_residual += -1.00 * einsum('ji,aj->ai', f[o, o], t1)
    singles_residual +=  1.00 * einsum('ab,bi->ai', f[v, v], t1)
    singles_residual += -1.00 * einsum('jb,baij->ai', f[o, v], t2)
    singles_residual += -1.00 * einsum('jb,aj,bi->ai', f[o, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    singles_residual +=  1.00 * einsum('jabi,bj->ai', g[o, v, v, o], t1)
    singles_residual += -0.50 * einsum('kjbi,bakj->ai', g[o, o, v, o], t2)
    singles_residual += -0.50 * einsum('jabc,bcij->ai', g[o, v, v, v], t2)
    singles_residual +=  1.00 * einsum('kjbc,caik,bj->ai', g[o, o, v, v], t2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    singles_residual +=  0.50 * einsum('kjbc,cakj,bi->ai', g[o, o, v, v], t2, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    singles_residual +=  0.50 * einsum('kjbc,aj,bcik->ai', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    singles_residual +=  1.00 * einsum('kjbi,ak,bj->ai', g[o, o, v, o], t1, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    singles_residual +=  1.00 * einsum('jabc,bj,ci->ai', g[o, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    singles_residual +=  1.00 * einsum('kjbc,ak,bj,ci->ai', g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    return singles_residual

from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.containers import GHF_CCSD_Data


def get_doubles_residual(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    f = ghf_data.f
    g = ghf_data.g
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    t2 = ghf_ccsd_data.t2
    
    contracted_intermediate = -1.00 * einsum('kj,abik->abij', f[o, o], t2)
    doubles_residual =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ac,cbij->abij', f[v, v], t2)
    doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,abik,cj->abij', f[o, v], t2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kc,ak,cbij->abij', f[o, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    doubles_residual +=  1.00 * einsum('abij->abij', g[v, v, o, o])
    contracted_intermediate =  1.00 * einsum('kaij,bk->abij', g[o, v, o, o], t1)
    doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('abcj,ci->abij', g[v, v, v, o], t1)
    doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    doubles_residual +=  0.50 * einsum('lkij,ablk->abij', g[o, o, o, o], t2)
    contracted_intermediate =  1.00 * einsum('kacj,cbik->abij', g[o, v, v, o], t2)
    doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    doubles_residual +=  0.50 * einsum('abcd,cdij->abij', g[v, v, v, v], t2)
    contracted_intermediate =  1.00 * einsum('lkcj,abil,ck->abij', g[o, o, v, o], t2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('lkcj,ablk,ci->abij', g[o, o, v, o], t2, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('lkcj,ak,cbil->abij', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kacd,dbij,ck->abij', g[o, v, v, v], t2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kacd,dbik,cj->abij', g[o, v, v, v], t2, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    contracted_intermediate =  0.50 * einsum('kacd,bk,cdij->abij', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    doubles_residual += -1.00 * einsum('lkij,ak,bl->abij', g[o, o, o, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('kacj,bk,ci->abij', g[o, v, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    doubles_residual += -1.00 * einsum('abcd,cj,di->abij', g[v, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate = -0.50 * einsum('lkcd,abil,cdjk->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    doubles_residual +=  0.250 * einsum('lkcd,ablk,cdij->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_residual += -0.50 * einsum('lkcd,calk,dbij->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('lkcd,cajk,dbil->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    doubles_residual += -0.50 * einsum('lkcd,caij,dblk->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('lkcd,abil,ck,dj->abij', g[o, o, v, v], t2, t1, t1, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkcd,al,dbij,ck->abij', g[o, o, v, v], t1, t2, t1, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    doubles_residual += -0.50 * einsum('lkcd,ablk,cj,di->abij', g[o, o, v, v], t2, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('lkcd,ak,dbil,cj->abij', g[o, o, v, v], t1, t2, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    doubles_residual += -0.50 * einsum('lkcd,ak,bl,cdij->abij', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    contracted_intermediate = -1.00 * einsum('lkcj,ak,bl,ci->abij', g[o, o, v, o], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('kacd,bk,cj,di->abij', g[o, v, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    doubles_residual +=  1.00 * einsum('lkcd,ak,bl,cj,di->abij', g[o, o, v, v], t1, t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    return doubles_residual

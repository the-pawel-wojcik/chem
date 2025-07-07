from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.containers import GHF_CCSD_Data


def get_lambda_singles_residual(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    f = ghf_data.f
    g = ghf_data.g
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    t2 = ghf_ccsd_data.t2
    if ghf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in GHF_CCSD_Data")
    l1 = ghf_ccsd_data.lmbda.l1
    l2 = ghf_ccsd_data.lmbda.l2
    
    lambda_singles_residual =  1.00 * einsum('ia->ia', f[o, v])
    lambda_singles_residual +=  1.00 * einsum('jj,ia->ia', f[o, o], l1)
    lambda_singles_residual += -1.00 * einsum('ij,ja->ia', f[o, o], l1)
    lambda_singles_residual +=  1.00 * einsum('ba,ib->ia', f[v, v], l1)
    lambda_singles_residual += -1.00 * einsum('bj,ijba->ia', f[v, o], l2)
    lambda_singles_residual +=  1.00 * einsum('jb,bj,ia->ia', f[o, v], t1, l1, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_singles_residual += -1.00 * einsum('ib,bj,ja->ia', f[o, v], t1, l1, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_singles_residual += -1.00 * einsum('ja,bj,ib->ia', f[o, v], t1, l1, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_singles_residual +=  1.00 * einsum('kj,bk,ijba->ia', f[o, o], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_singles_residual += -1.00 * einsum('bc,cj,ijba->ia', f[v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_singles_residual +=  1.00 * einsum('kc,cbjk,ijba->ia', f[o, v], t2, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_singles_residual +=  0.50 * einsum('ic,cbjk,jkba->ia', f[o, v], t2, l2, optimize=['einsum_path', (1, 2), (0, 1)])
    lambda_singles_residual +=  0.50 * einsum('ka,cbjk,ijcb->ia', f[o, v], t2, l2, optimize=['einsum_path', (1, 2), (0, 1)])
    lambda_singles_residual +=  1.00 * einsum('kc,bk,cj,ijba->ia', f[o, v], t1, t1, l2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lambda_singles_residual += -0.50 * einsum('kjkj,ia->ia', g[o, o, o, o], l1)
    lambda_singles_residual +=  1.00 * einsum('ibaj,jb->ia', g[o, v, v, o], l1)
    lambda_singles_residual +=  0.50 * einsum('ibjk,jkba->ia', g[o, v, o, o], l2)
    lambda_singles_residual +=  0.50 * einsum('cbaj,ijcb->ia', g[v, v, v, o], l2)
    lambda_singles_residual += -1.00 * einsum('ijba,bj->ia', g[o, o, v, v], t1)
    lambda_singles_residual +=  1.00 * einsum('ikbj,bk,ja->ia', g[o, o, v, o], t1, l1, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_singles_residual += -1.00 * einsum('ikaj,bk,jb->ia', g[o, o, v, o], t1, l1, optimize=['einsum_path', (1, 2), (0, 1)])
    lambda_singles_residual +=  1.00 * einsum('jbca,cj,ib->ia', g[o, v, v, v], t1, l1, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_singles_residual += -1.00 * einsum('ibca,cj,jb->ia', g[o, v, v, v], t1, l1, optimize=['einsum_path', (1, 2), (0, 1)])
    lambda_singles_residual += -0.50 * einsum('iljk,bl,jkba->ia', g[o, o, o, o], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_singles_residual += -1.00 * einsum('kbcj,ck,ijba->ia', g[o, v, v, o], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_singles_residual +=  1.00 * einsum('ibck,cj,jkba->ia', g[o, v, v, o], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_singles_residual +=  1.00 * einsum('kcaj,bk,ijcb->ia', g[o, v, v, o], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_singles_residual += -0.50 * einsum('cbda,dj,ijcb->ia', g[v, v, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_singles_residual +=  0.250 * einsum('kjbc,bckj,ia->ia', g[o, o, v, v], t2, l1, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_singles_residual += -0.50 * einsum('ikbc,bcjk,ja->ia', g[o, o, v, v], t2, l1, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_singles_residual += -0.50 * einsum('kjca,cbkj,ib->ia', g[o, o, v, v], t2, l1, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_singles_residual +=  1.00 * einsum('ikca,cbjk,jb->ia', g[o, o, v, v], t2, l1, optimize=['einsum_path', (1, 2), (0, 1)])
    lambda_singles_residual +=  0.50 * einsum('lkcj,cblk,ijba->ia', g[o, o, v, o], t2, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_singles_residual += -1.00 * einsum('ilck,cbjl,jkba->ia', g[o, o, v, o], t2, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_singles_residual +=  0.250 * einsum('lkaj,cblk,ijcb->ia', g[o, o, v, o], t2, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_singles_residual += -0.50 * einsum('ilak,cbjl,jkcb->ia', g[o, o, v, o], t2, l2, optimize=['einsum_path', (1, 2), (0, 1)])
    lambda_singles_residual +=  0.50 * einsum('kbcd,cdjk,ijba->ia', g[o, v, v, v], t2, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_singles_residual +=  0.250 * einsum('ibcd,cdjk,jkba->ia', g[o, v, v, v], t2, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_singles_residual += -1.00 * einsum('kcda,dbjk,ijcb->ia', g[o, v, v, v], t2, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_singles_residual += -0.50 * einsum('icda,dbjk,jkcb->ia', g[o, v, v, v], t2, l2, optimize=['einsum_path', (1, 2), (0, 1)])
    lambda_singles_residual += -1.00 * einsum('lkcd,dbjl,ck,ijba->ia', g[o, o, v, v], t2, t1, l2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    lambda_singles_residual += -0.50 * einsum('ilcd,dbjk,cl,jkba->ia', g[o, o, v, v], t2, t1, l2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lambda_singles_residual += -0.50 * einsum('lkda,cbjl,dk,ijcb->ia', g[o, o, v, v], t2, t1, l2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lambda_singles_residual += -0.50 * einsum('lkcd,dblk,cj,ijba->ia', g[o, o, v, v], t2, t1, l2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lambda_singles_residual +=  1.00 * einsum('ilcd,dbjl,ck,jkba->ia', g[o, o, v, v], t2, t1, l2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lambda_singles_residual += -0.250 * einsum('lkda,cblk,dj,ijcb->ia', g[o, o, v, v], t2, t1, l2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lambda_singles_residual +=  0.50 * einsum('ilda,cbjl,dk,jkcb->ia', g[o, o, v, v], t2, t1, l2, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    lambda_singles_residual += -0.50 * einsum('lkcd,bk,cdjl,ijba->ia', g[o, o, v, v], t1, t2, l2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    lambda_singles_residual += -0.250 * einsum('ilcd,bl,cdjk,jkba->ia', g[o, o, v, v], t1, t2, l2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lambda_singles_residual +=  1.00 * einsum('lkda,dbjl,ck,ijcb->ia', g[o, o, v, v], t2, t1, l2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lambda_singles_residual +=  0.50 * einsum('ilda,dbjk,cl,jkcb->ia', g[o, o, v, v], t2, t1, l2, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    lambda_singles_residual += -0.50 * einsum('kjbc,bj,ck,ia->ia', g[o, o, v, v], t1, t1, l1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lambda_singles_residual +=  1.00 * einsum('ikbc,bk,cj,ja->ia', g[o, o, v, v], t1, t1, l1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lambda_singles_residual +=  1.00 * einsum('kjca,bk,cj,ib->ia', g[o, o, v, v], t1, t1, l1, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    lambda_singles_residual +=  1.00 * einsum('ikca,bk,cj,jb->ia', g[o, o, v, v], t1, t1, l1, optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    lambda_singles_residual += -1.00 * einsum('lkcj,bl,ck,ijba->ia', g[o, o, v, o], t1, t1, l2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    lambda_singles_residual += -1.00 * einsum('ilck,bl,cj,jkba->ia', g[o, o, v, o], t1, t1, l2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lambda_singles_residual += -0.50 * einsum('lkaj,bl,ck,ijcb->ia', g[o, o, v, o], t1, t1, l2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lambda_singles_residual += -1.00 * einsum('kbcd,ck,dj,ijba->ia', g[o, v, v, v], t1, t1, l2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lambda_singles_residual += -0.50 * einsum('ibcd,ck,dj,jkba->ia', g[o, v, v, v], t1, t1, l2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lambda_singles_residual += -1.00 * einsum('kcda,bk,dj,ijcb->ia', g[o, v, v, v], t1, t1, l2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lambda_singles_residual += -1.00 * einsum('lkcd,bl,ck,dj,ijba->ia', g[o, o, v, v], t1, t1, t1, l2, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    lambda_singles_residual +=  0.50 * einsum('ilcd,bl,ck,dj,jkba->ia', g[o, o, v, v], t1, t1, t1, l2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    lambda_singles_residual +=  0.50 * einsum('lkda,bl,ck,dj,ijcb->ia', g[o, o, v, v], t1, t1, t1, l2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    return lambda_singles_residual

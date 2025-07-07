from numpy import einsum
from numpy.typing import NDArray
from chem.hf.ghf_data import GHF_Data
from chem.ccsd.containers import GHF_CCSD_Data


def get_lambda_doubles_residual(
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
    
    contracted_intermediate = -1.00 * einsum('ja,ib->ijab', f[o, v], l1)
    lambda_doubles_residual =  1.00000 * contracted_intermediate + -1.00000 * einsum('ijab->jiab', contracted_intermediate)  + -1.00000 * einsum('ijab->ijba', contracted_intermediate)  +  1.00000 * einsum('ijab->jiba', contracted_intermediate) 
    lambda_doubles_residual +=  1.00 * einsum('kk,ijab->ijab', f[o, o], l2)
    contracted_intermediate = -1.00 * einsum('jk,ikab->ijab', f[o, o], l2)
    lambda_doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('ijab->jiab', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ca,ijcb->ijab', f[v, v], l2)
    lambda_doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('ijab->ijba', contracted_intermediate) 
    lambda_doubles_residual +=  1.00 * einsum('kc,ck,ijab->ijab', f[o, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate = -1.00 * einsum('jc,ck,ikab->ijab', f[o, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('ijab->jiab', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('ka,ck,ijcb->ijab', f[o, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('ijab->ijba', contracted_intermediate) 
    lambda_doubles_residual += -0.50 * einsum('lklk,ijab->ijab', g[o, o, o, o], l2)
    lambda_doubles_residual +=  1.00 * einsum('ijab->ijab', g[o, o, v, v])
    contracted_intermediate = -1.00 * einsum('ijak,kb->ijab', g[o, o, v, o], l1)
    lambda_doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('ijab->ijba', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jcab,ic->ijab', g[o, v, v, v], l1)
    lambda_doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('ijab->jiab', contracted_intermediate) 
    lambda_doubles_residual +=  0.50 * einsum('ijkl,klab->ijab', g[o, o, o, o], l2)
    contracted_intermediate =  1.00 * einsum('jcak,ikcb->ijab', g[o, v, v, o], l2)
    lambda_doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('ijab->jiab', contracted_intermediate)  + -1.00000 * einsum('ijab->ijba', contracted_intermediate)  +  1.00000 * einsum('ijab->jiba', contracted_intermediate) 
    lambda_doubles_residual +=  0.50 * einsum('dcab,ijdc->ijab', g[v, v, v, v], l2)
    contracted_intermediate =  1.00 * einsum('jkca,ck,ib->ijab', g[o, o, v, v], t1, l1, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('ijab->jiab', contracted_intermediate)  + -1.00000 * einsum('ijab->ijba', contracted_intermediate)  +  1.00000 * einsum('ijab->jiba', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('ijca,ck,kb->ijab', g[o, o, v, v], t1, l1, optimize=['einsum_path', (1, 2), (0, 1)])
    lambda_doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('ijab->ijba', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jkab,ck,ic->ijab', g[o, o, v, v], t1, l1, optimize=['einsum_path', (1, 2), (0, 1)])
    lambda_doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('ijab->jiab', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jlck,cl,ikab->ijab', g[o, o, v, o], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('ijab->jiab', contracted_intermediate) 
    lambda_doubles_residual +=  1.00 * einsum('ijcl,ck,klab->ijab', g[o, o, v, o], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate = -1.00 * einsum('jlak,cl,ikcb->ijab', g[o, o, v, o], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('ijab->jiab', contracted_intermediate)  + -1.00000 * einsum('ijab->ijba', contracted_intermediate)  +  1.00000 * einsum('ijab->jiba', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('kcda,dk,ijcb->ijab', g[o, v, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('ijab->ijba', contracted_intermediate) 
    contracted_intermediate = -1.00 * einsum('jcda,dk,ikcb->ijab', g[o, v, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('ijab->jiab', contracted_intermediate)  + -1.00000 * einsum('ijab->ijba', contracted_intermediate)  +  1.00000 * einsum('ijab->jiba', contracted_intermediate) 
    lambda_doubles_residual +=  1.00 * einsum('kdab,ck,ijdc->ijab', g[o, v, v, v], t1, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_doubles_residual +=  0.250 * einsum('lkcd,cdlk,ijab->ijab', g[o, o, v, v], t2, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate = -0.50 * einsum('jlcd,cdkl,ikab->ijab', g[o, o, v, v], t2, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('ijab->jiab', contracted_intermediate) 
    lambda_doubles_residual +=  0.250 * einsum('ijcd,cdkl,klab->ijab', g[o, o, v, v], t2, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate = -0.50 * einsum('lkda,dclk,ijcb->ijab', g[o, o, v, v], t2, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('ijab->ijba', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('jlda,dckl,ikcb->ijab', g[o, o, v, v], t2, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    lambda_doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('ijab->jiab', contracted_intermediate)  + -1.00000 * einsum('ijab->ijba', contracted_intermediate)  +  1.00000 * einsum('ijab->jiba', contracted_intermediate) 
    contracted_intermediate = -0.50 * einsum('ijda,dckl,klcb->ijab', g[o, o, v, v], t2, l2, optimize=['einsum_path', (1, 2), (0, 1)])
    lambda_doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('ijab->ijba', contracted_intermediate) 
    lambda_doubles_residual +=  0.250 * einsum('lkab,dclk,ijdc->ijab', g[o, o, v, v], t2, l2, optimize=['einsum_path', (0, 1), (0, 1)])
    contracted_intermediate = -0.50 * einsum('jlab,dckl,ikdc->ijab', g[o, o, v, v], t2, l2, optimize=['einsum_path', (1, 2), (0, 1)])
    lambda_doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('ijab->jiab', contracted_intermediate) 
    lambda_doubles_residual += -0.50 * einsum('lkcd,ck,dl,ijab->ijab', g[o, o, v, v], t1, t1, l2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    contracted_intermediate =  1.00 * einsum('jlcd,cl,dk,ikab->ijab', g[o, o, v, v], t1, t1, l2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    lambda_doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('ijab->jiab', contracted_intermediate) 
    contracted_intermediate =  1.00 * einsum('lkda,cl,dk,ijcb->ijab', g[o, o, v, v], t1, t1, l2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    lambda_doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('ijab->ijba', contracted_intermediate) 
    lambda_doubles_residual += -0.50 * einsum('ijcd,cl,dk,klab->ijab', g[o, o, v, v], t1, t1, l2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    contracted_intermediate =  1.00 * einsum('jlda,cl,dk,ikcb->ijab', g[o, o, v, v], t1, t1, l2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    lambda_doubles_residual +=  1.00000 * contracted_intermediate + -1.00000 * einsum('ijab->jiab', contracted_intermediate)  + -1.00000 * einsum('ijab->ijba', contracted_intermediate)  +  1.00000 * einsum('ijab->jiba', contracted_intermediate) 
    lambda_doubles_residual += -0.50 * einsum('lkab,cl,dk,ijdc->ijab', g[o, o, v, v], t1, t1, l2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    return lambda_doubles_residual

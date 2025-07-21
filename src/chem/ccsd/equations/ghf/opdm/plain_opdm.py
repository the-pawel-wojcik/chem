One-particle density matrix
<Λ|p†q|CC>
opdm +=  1.00 * einsum('ij', kd[o, o])
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

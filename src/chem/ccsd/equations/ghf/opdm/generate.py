from chem.ccsd.equations.ghf.printer import (
    DefineSections,
    print_imports,
    print_to_numpy,
)
import pdaggerq


def build_opdm_expression():
    pq = pdaggerq.pq_helper('fermi')
    pq.set_left_operators([['1'], ['l1'], ['l2']])
    pq.add_st_operator(1.0, ['a*(t)', 'a(u)'], ['t1', 't2'])
    pq.simplify()
    return pq


def main():
    pq = build_opdm_expression()
    print_imports()
    print_to_numpy(
        pq=pq,
        tensor_name='opdm',
        tensor_subscripts=('t', 'u'),
        defines_exclude={DefineSections.FLUCTUATION, DefineSections.FOCK},
    )


if __name__ == "__main__":
    main()

from chem.ccsd.equations.ghf.printer import (
    print_to_numpy,
    print_imports,
    DefineSections,
)
from chem.meta.coordinates import Descartes
import pdaggerq


def main():
    print_imports()
    for component in Descartes:
        pq = pdaggerq.pq_helper('fermi')
        pq.set_left_operators([['1'], ['l1'], ['l2']])
        pq.add_st_operator(1.0, ['h'], ['t1', 't2'])
        pq.simplify()
        extra_definitions = (
            f'h = ghf_data.mu[Descartes.{component}]',
        )
        print_to_numpy(
            pq,
            tensor_name=f'mu{component}',
            tensor_subscripts=(),
            defines_exclude={
                DefineSections.IDENTITY,
                DefineSections.FOCK,
                DefineSections.FLUCTUATION,
            },
            extra_definitions=extra_definitions,
        )


if __name__ == "__main__":
    main()

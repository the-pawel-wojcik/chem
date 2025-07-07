import argparse

import pdaggerq
from pdaggerq.parser import contracted_strings_to_tensor_terms


TAB = '    '


def print_imports() -> None:
    print('from numpy import einsum')
    print('from numpy.typing import NDArray')
    print('from chem.hf.ghf_data import GHF_Data')
    print('from chem.ccsd.containers import GHF_CCSD_Data')


def print_function_header(quantity: str) -> None:

    if not quantity.isidentifier():
        raise ValueError('Argument must be a valid python isidentifier.')

    body = f'''\n\ndef get_{quantity}(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,
) -> NDArray:
    f = ghf_data.f
    g = ghf_data.g
    v = ghf_data.v
    o = ghf_data.o
    t1 = ghf_ccsd_data.t1
    t2 = ghf_ccsd_data.t2
    '''

    print(body)


def numpy_print_singles_residual(pq):
    quantity = 'singles_residual'
    print_imports()

    terms = pq.strings()
    tensor_terms = contracted_strings_to_tensor_terms(terms)

    print_function_header(quantity=quantity,)
    for my_term in tensor_terms:
        einsum_terms = my_term.einsum_string(
            output_variables=('a', 'i'),
            update_val=quantity,
        )
        print(f"{TAB}{einsum_terms}")
    print(f'{TAB}return {quantity}')


def numpy_print_doubles_residual(pq):
    print_imports()
    quantity = 'doubles_residual'
    terms = pq.strings()
    tensor_terms = contracted_strings_to_tensor_terms(terms)

    print_function_header(quantity=quantity)
    for my_term in tensor_terms:
        einsum_terms = my_term.einsum_string(
            output_variables=('a', 'b', 'i', 'j'),
            update_val=quantity,
        )
        for print_term in einsum_terms.split('\n'):
            print(f"{TAB}{print_term}")
    print(f'{TAB}return {quantity}')


def build_singles():
    pq = pdaggerq.pq_helper('fermi')
    pq.set_left_operators([['a*(i)', 'a(a)']])
    pq.add_st_operator(1.0, ['f'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['v'], ['t1', 't2'])
    pq.simplify()
    return pq


def build_doubles():
    pq = pdaggerq.pq_helper('fermi')
    # The order changes everything. Crazy.
    pq.set_left_operators([['a*(i)', 'a*(j)', 'a(b)', 'a(a)']])
    pq.add_st_operator(1.0, ['f'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['v'], ['t1', 't2'])
    pq.simplify()
    return pq


def main():
    parser = argparse.ArgumentParser()
    options = parser.add_mutually_exclusive_group()
    options.add_argument('--singles', default=False, action='store_true')
    options.add_argument('--doubles', default=False, action='store_true')
    args = parser.parse_args()

    if args.singles:
        pq = build_singles()
        numpy_print_singles_residual(pq)

    if args.doubles:
        pq = build_doubles()
        numpy_print_doubles_residual(pq)


if __name__ == "__main__":
    main()

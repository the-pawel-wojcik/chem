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
    if ghf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in GHF_CCSD_Data")
    l1 = ghf_ccsd_data.lmbda.l1
    l2 = ghf_ccsd_data.lmbda.l2
    '''
    print(body)


def print_singles_in_numpy(pq):
    tensor_name = 'lambda_singles_residual'
    print_imports()

    terms = pq.strings()
    tensor_terms = contracted_strings_to_tensor_terms(terms)

    print_function_header(
        quantity=tensor_name,
    )
    for my_term in tensor_terms:
        einsum_terms = my_term.einsum_string(
            output_variables=('i', 'a'),
            update_val=tensor_name,
        )
        print(f"{TAB}{einsum_terms}")
    print(f'{TAB}return {tensor_name}')


def print_doubles_in_numpy(pq):
    tensor_name = 'lambda_doubles_residual'
    print_imports()

    terms = pq.strings()
    tensor_terms = contracted_strings_to_tensor_terms(terms)

    print_function_header(
        quantity=tensor_name,
    )

    output_variables = ('i', 'j', 'b', 'a')
    for my_term in tensor_terms:
        einsum_terms = my_term.einsum_string(
            output_variables=output_variables,
            update_val=tensor_name,
        )
        for print_term in einsum_terms.split('\n'):
            print(f"{TAB}{print_term}")

    print(f'{TAB}return {tensor_name}')


def build_singles():
    """
    <0| (1 + l1 + l2) e^{-T} H e^T a^† i |0>
    """
    pq = pdaggerq.pq_helper('fermi')
    pq.set_left_operators([['1'], ['l1'], ['l2']])
    pq.set_right_operators([['a*(a)', 'a(i)']])
    pq.add_st_operator(1.0, ['f'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['v'], ['t1', 't2'])
    pq.simplify()
    return pq


def build_doubles():
    pq = pdaggerq.pq_helper('fermi')
    pq.set_left_operators([['1'], ['l1'], ['l2']])
    pq.set_right_operators([['a*(a)', 'a*(b)', 'a(j)', 'a(i)']])
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
        print_singles_in_numpy(pq)

    if args.doubles:
        pq = build_doubles()
        print_doubles_in_numpy(pq)


if __name__ == "__main__":
    main()

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


def numpy_print_energy(pq):
    out_var = 'ghf_ccsd_energy'
    terms = pq.strings()
    tensor_terms = contracted_strings_to_tensor_terms(terms)

    print_imports()
    print_function_header('ghf_ccsd_energy')

    for my_term in tensor_terms:
        einsum_terms = my_term.einsum_string(
            output_variables=(),
            update_val=out_var,
        )
        print(f'{TAB}{einsum_terms}')
    print(f'{TAB}return {out_var}')


def build_energy():
    pq = pdaggerq.pq_helper('fermi')
    pq.add_st_operator(1.0, ['f'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['v'], ['t1', 't2'])
    pq.simplify()
    return pq


def main():
    pq = build_energy()
    numpy_print_energy(pq)


if __name__ == "__main__":
    main()

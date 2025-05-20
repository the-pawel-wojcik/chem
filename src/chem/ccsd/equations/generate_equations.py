import itertools
import pdaggerq
from pdaggerq.parser import contracted_strings_to_tensor_terms

TAB = '    '

def print_imports() -> None:
    print('from numpy import einsum')
    print('from numpy.typing import NDArray')
    print('from chem.hf.intermediates_builders import Intermediates')


def print_function_header(quantity: str, spin_subscript: str = '') -> None:

    if not quantity.isidentifier():
        raise ValueError('Argument must be a valid python isidentifier.')
    if spin_subscript != '' and not spin_subscript.isidentifier():
        raise ValueError('Argument must be a valid python isidentifier.')

    if spin_subscript != '':
        spin_subscript = '_' + spin_subscript

    body = f'''\n\ndef get_{quantity}{spin_subscript}(
    intermediates: Intermediates,
    t1_aa: NDArray,
    t1_bb: NDArray,
    t2_aaaa: NDArray,
    t2_abab: NDArray,
    t2_bbbb: NDArray,
) -> NDArray:
    f_aa = intermediates.f_aa
    f_bb = intermediates.f_bb
    g_aaaa = intermediates.g_aaaa
    g_abab = intermediates.g_abab
    g_bbbb = intermediates.g_bbbb
    va = intermediates.va
    vb = intermediates.vb
    oa = intermediates.oa
    ob = intermediates.ob
    '''
    print(body)


def numpy_print_energy(pq):
    out_var = 'uhf_ccsd_energy'
    terms = pq.strings(spin_labels={})
    tensor_terms = contracted_strings_to_tensor_terms(terms)

    print_imports()
    print_function_header('energy')

    for my_term in tensor_terms:
        einsum_terms = my_term.einsum_string(
            output_variables=(),
            update_val=out_var,
        )
        print(f'{TAB}{einsum_terms}')
    print(f'{TAB}return {out_var}')


def numpy_print_singles_uhf(pq):
    print_imports()
    for spin_mix in itertools.product(['a', 'b'], repeat=2):
        spin_labels = {
            'a': spin_mix[0],
            'i': spin_mix[1],
        }

        terms = pq.strings(spin_labels=spin_labels)
        tensor_terms = contracted_strings_to_tensor_terms(terms)
        if len(tensor_terms) == 0:
            continue

        print_function_header(
            quantity='singles_residual',
            spin_subscript=''.join(spin_mix)
        )
        out_var = 'singles_res_' + ''.join(spin_mix)
        for my_term in tensor_terms:
            einsum_terms = my_term.einsum_string(
                output_variables=('a', 'i'),
                update_val=out_var,
            )
            print(f"{TAB}{einsum_terms}")
        print(f'{TAB}return {out_var}')


def numpy_print_doubles_uhf(pq):
    print_imports()
    for spin_mix in itertools.product(['a', 'b'], repeat=4):
        out_var = 'doubles_res_' + ''.join(spin_mix)
        spin_labels = {
            'a': spin_mix[0],
            'b': spin_mix[1],
            'i': spin_mix[2],
            'j': spin_mix[3],
        }
        terms = pq.strings(spin_labels=spin_labels)
        tensor_terms = contracted_strings_to_tensor_terms(terms)
        if len(tensor_terms) == 0:
            continue

        print_function_header(
            quantity='doubles_residual',
            spin_subscript=''.join(spin_mix)
        )
        for my_term in tensor_terms:
            einsum_terms = my_term.einsum_string(
                output_variables=('a', 'b', 'i', 'j'),
                update_val=out_var
            )
            for print_term in einsum_terms.split('\n'):
                print(f"{TAB}{print_term}")
        print(f'{TAB}return {out_var}')


def build_energy():
    pq = pdaggerq.pq_helper('fermi')
    pq.add_st_operator(1.0, ['f'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['v'], ['t1', 't2'])
    pq.simplify()
    return pq


def build_singles():
    pq = pdaggerq.pq_helper('fermi')
    pq.set_left_operators([['a*(i)', 'a(a)']])
    pq.add_st_operator(1.0, ['f'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['v'], ['t1', 't2'])
    pq.simplify()
    return pq


def build_doubles():
    pq = pdaggerq.pq_helper('fermi')
    pq.set_left_operators([['a*(j)', 'a*(i)', 'a(a)', 'a(b)']])
    pq.add_st_operator(1.0, ['f'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['v'], ['t1', 't2'])
    pq.simplify()
    return pq


def main():
    do_energy = True
    do_singles = False
    do_doubles = False

    if do_energy is True:
        pq = build_energy()
        numpy_print_energy(pq)

    if do_singles is True:
        pq = build_singles()
        numpy_print_singles_uhf(pq)

    elif do_doubles is True:
        pq = build_doubles()
        numpy_print_doubles_uhf(pq)


if __name__ == "__main__":
    main()

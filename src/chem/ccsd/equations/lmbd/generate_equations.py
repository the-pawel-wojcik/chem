import itertools
import pdaggerq
from pdaggerq.parser import contracted_strings_to_tensor_terms

TAB = '    '

def print_imports() -> None:
    print('from numpy import einsum')
    print('from numpy.typing import NDArray')
    print('from chem.hf.intermediates_builders import Intermediates')
    print('from chem.ccsd.uhf_ccsd import UHF_CCSD_Data')


def print_function_header(quantity: str, spin_subscript: str = '') -> None:

    if not quantity.isidentifier():
        raise ValueError('Argument must be a valid python isidentifier.')
    if spin_subscript != '' and not spin_subscript.isidentifier():
        raise ValueError('Argument must be a valid python isidentifier.')

    if spin_subscript != '':
        spin_subscript = '_' + spin_subscript

    body = f'''\n\ndef get_{quantity}{spin_subscript}(
    uhf_scf_data: Intermediates,
    uhf_ccsd_data: UHF_CCSD_Data,
) -> NDArray:
    f_aa = uhf_scf_data.f_aa
    f_bb = uhf_scf_data.f_bb
    g_aaaa = uhf_scf_data.g_aaaa
    g_abab = uhf_scf_data.g_abab
    g_bbbb = uhf_scf_data.g_bbbb
    va = uhf_scf_data.va
    vb = uhf_scf_data.vb
    oa = uhf_scf_data.oa
    ob = uhf_scf_data.ob
    t1_aa = uhf_ccsd_data.t1_aa
    t1_bb = uhf_ccsd_data.t1_bb
    t2_aaaa = uhf_ccsd_data.t2_aaaa
    t2_abab = uhf_ccsd_data.t2_abab
    t2_bbbb = uhf_ccsd_data.t2_bbbb
    if uhf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in UHF_CCSD_Data")
    l1_aa = uhf_ccsd_data.lmbda.l1_aa
    l1_bb = uhf_ccsd_data.lmbda.l1_bb
    l2_aaaa = uhf_ccsd_data.lmbda.l2_aaaa
    l2_abab = uhf_ccsd_data.lmbda.l2_abab
    l2_bbbb = uhf_ccsd_data.lmbda.l2_bbbb
    '''
    print(body)


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
            quantity='lambda_singles_residual',
            spin_subscript=''.join(spin_mix)
        )
        out_var = 'lambda_singles_res_' + ''.join(spin_mix)
        for my_term in tensor_terms:
            einsum_terms = my_term.einsum_string(
                output_variables=('a', 'i'),
                update_val=out_var,
            )
            print(f"{TAB}{einsum_terms}")
        print(f'{TAB}return {out_var}')


def numpy_print_doubles_uhf(pq):
    tensor_name = 'lambda_doubles_res'
    print_imports()
    for spin_mix in itertools.product(['a', 'b'], repeat=4):
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

        spin_subscript = ''.join(spin_labels.values())
        print_function_header(
            quantity=tensor_name,
            spin_subscript=spin_subscript,
        )

        out_var = tensor_name + '_' + spin_subscript
        for my_term in tensor_terms:
            einsum_terms = my_term.einsum_string(
                output_variables=('a', 'b', 'j', 'i'),
                update_val=out_var
            )
            for print_term in einsum_terms.split('\n'):
                print(f"{TAB}{print_term}")

        print(f'{TAB}return {out_var}')


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
    do_singles = False
    do_doubles = True

    if do_singles is True:
        pq = build_singles()
        numpy_print_singles_uhf(pq)

    elif do_doubles is True:
        pq = build_doubles()
        numpy_print_doubles_uhf(pq)


if __name__ == "__main__":
    main()

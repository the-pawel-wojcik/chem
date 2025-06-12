import pdaggerq
import itertools
from pdaggerq.parser import contracted_strings_to_tensor_terms


TAB = '    '


def print_imports() -> None:
    print('from numpy import einsum')
    print('from numpy.typing import NDArray')
    print('from chem.hf.intermediates_builders import Intermediates')
    print('from chem.ccs.containers import UHF_CCS_Data')


def print_function_header(quantity: str, spin_subscript: str = '') -> None:

    if not quantity.isidentifier():
        raise ValueError('Argument must be a valid python isidentifier.')
    if spin_subscript != '' and not spin_subscript.isidentifier():
        raise ValueError('Argument must be a valid python isidentifier.')

    if spin_subscript != '':
        spin_subscript = '_' + spin_subscript

    body = f'''\n\ndef get_{quantity}{spin_subscript}(
    uhf_scf_data: Intermediates,
    uhf_ccs_data: UHF_CCS_Data,
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
    t1_aa = uhf_ccs_data.t1_aa
    t1_bb = uhf_ccs_data.t1_bb
    '''
    print(body)


def energy():
    tensor_name = 'ccs_energy'
    pq = pdaggerq.pq_helper('fermi')
    pq.add_st_operator(1.0, ['f'], ['t1'])
    pq.add_st_operator(1.0, ['v'], ['t1'])
    pq.simplify()
    strings = pq.strings(spin_labels={})
    tensor_terms = contracted_strings_to_tensor_terms(strings)
    print_imports()
    print_function_header(tensor_name)
    for tensor in tensor_terms:
        numpyfied = tensor.einsum_string(
            output_variables=(),
            update_val=tensor_name,
        )
        print(f'{TAB}{numpyfied}')
    print(f'{TAB}return {tensor_name}')


def singles():
    tensor_name = 'uhf_ccs_singles_residuals'
    tensor_subscripts = ('a', 'i')

    pq = pdaggerq.pq_helper('fermi')
    pq.set_left_operators([['e1(i,a)']])
    pq.add_st_operator(1.0, ['f'], ['t1'])
    pq.add_st_operator(1.0, ['v'], ['t1'])
    pq.simplify()

    print_imports()
    subscripts_count = len(tensor_subscripts)
    for spin_mix in itertools.product(['a', 'b'], repeat=subscripts_count):
        spin_labels = {
            subscript: spin_mix[idx] for idx, subscript
            in enumerate(tensor_subscripts)
        }
        strings = pq.strings(spin_labels=spin_labels)
        if len(strings) == 0:
            continue
        tensor_terms = contracted_strings_to_tensor_terms(strings)
        spin_subscript = ''.join(spin_mix)
        out_var_name = tensor_name + '_' + spin_subscript
        print_function_header(out_var_name)
        for tensor in tensor_terms:
            numpyfied = tensor.einsum_string(
                output_variables=tuple(tensor_subscripts),
                update_val=out_var_name,
            )
            print(f'{TAB}{numpyfied}')
        print(f'{TAB}return {out_var_name}')


if __name__ == '__main__':
    # energy()
    singles()

import pdaggerq
import itertools
from pdaggerq.parser import contracted_strings_to_tensor_terms


TAB = '    '


def print_imports() -> None:
    print('from numpy import einsum')
    print('from numpy.typing import NDArray')
    print('from chem.hf.intermediates_builders import Intermediates')
    print('from chem.ccs.containers import UHF_CCS_Data, UHF_CCS_Lambda_Data')


def print_function_header(quantity: str, spin_subscript: str = '') -> None:

    if not quantity.isidentifier():
        raise ValueError('Argument must be a valid python isidentifier.')
    if spin_subscript != '' and not spin_subscript.isidentifier():
        raise ValueError('Argument must be a valid python isidentifier.')

    if spin_subscript != '':
        spin_subscript = '_' + spin_subscript

    body = f'''\n\ndef get_{quantity}{spin_subscript}(
    uhf_data: Intermediates,
    uhf_ccs_data: UHF_CCS_Data,
    uhf_ccs_lambda_data: UHF_CCS_Lambda_Data,
) -> NDArray:
    f_aa = uhf_data.f_aa
    f_bb = uhf_data.f_bb
    g_aaaa = uhf_data.g_aaaa
    g_abab = uhf_data.g_abab
    g_bbbb = uhf_data.g_bbbb
    va = uhf_data.va
    vb = uhf_data.vb
    oa = uhf_data.oa
    ob = uhf_data.ob
    t1_aa = uhf_ccs_data.t1_aa
    t1_bb = uhf_ccs_data.t1_bb
    l1_aa = uhf_ccs_lambda_data.l1_aa
    l1_bb = uhf_ccs_lambda_data.l1_bb
    '''
    print(body)


def singles():
    """
    <0| (1 + l1 + l2) e^{-T} H e^T a^† i |0>
    """
    tensor_name = 'uhf_ccs_lambda_singles_res'
    tensor_subscripts = ('i', 'a')

    pq = pdaggerq.pq_helper('fermi')
    pq.set_left_operators([['1'], ['l1']])
    pq.set_right_operators([['e1(a,i)']])
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
    singles()

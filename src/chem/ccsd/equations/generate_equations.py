import itertools
import pdaggerq


def numpy_print_energy_uhf(pq):
    print('import numpy as np')
    print('from numpy import einsum')
    print()
    from pdaggerq.parser import contracted_strings_to_tensor_terms

    out_var = 'uhf_ccsd'
    print(f'{out_var} = np.zeros()  # TODO')

    terms = pq.strings(spin_labels={})
    tensor_terms = contracted_strings_to_tensor_terms(terms)

    for my_term in tensor_terms:
        einsum_terms = my_term.einsum_string(
            output_variables=(),
            update_val=out_var,
        )
        print(f"{einsum_terms}")


TAB = '    '
VARS_FILLER = '''
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

SINGLES_HEADER= {
    'aa': '''
def get_singles_residual_aa(''' + VARS_FILLER,  # )
    'bb':'''
def get_singles_residual_bb(''' + VARS_FILLER,  # )
    'ab':'# TODO: CLEANUP',
    'ba':'# TODO: CLEANUP',
}


def numpy_print_singles_uhf(pq):
    print('from numpy import einsum')
    print('from numpy.typing import NDArray')
    print('from chem.hf.intermediates_builders import Intermediates')
    print()

    from pdaggerq.parser import contracted_strings_to_tensor_terms
    for spin_mix in itertools.product(['a', 'b'], repeat=2):
        spin_labels = {
            'a': spin_mix[0],
            'i': spin_mix[1],
        }
        
        print(SINGLES_HEADER[''.join(spin_mix)])
        print()

        terms = pq.strings(spin_labels=spin_labels)
        tensor_terms = contracted_strings_to_tensor_terms(terms)

        out_var = 'singles_res_' + ''.join(spin_mix)
        for my_term in tensor_terms:
            einsum_terms = my_term.einsum_string(
                output_variables=('a', 'i'),
                update_val=out_var,
            )
            print(f"{TAB}{einsum_terms}")
        print(f'{TAB}return {out_var}\n\n')


DOUBLES_HEADER= {
    'aaaa': '''
def get_doubles_residual_aaaa(''' + VARS_FILLER,  # )
    'bbbb':'''
def get_doubles_residual_bbbb(''' + VARS_FILLER,  # )
    'abab': '''
def get_doubles_residual_abab(''' + VARS_FILLER,  # )
    'abba': '''
def get_doubles_residual_abba(''' + VARS_FILLER,  # )
    'baab': '''
def get_doubles_residual_baab(''' + VARS_FILLER,  # )
    'baba': '''
def get_doubles_residual_baba(''' + VARS_FILLER,  # )
    'aaab': '# TODO: CLEANUP',
    'aaba': '# TODO: CLEANUP',
    'aabb': '# TODO: CLEANUP',
    'abaa': '# TODO: CLEANUP',
    'abbb': '# TODO: CLEANUP',
    'baaa': '# TODO: CLEANUP',
    'babb': '# TODO: CLEANUP',
    'bbaa': '# TODO: CLEANUP',
    'bbab': '# TODO: CLEANUP',
    'bbba': '# TODO: CLEANUP',
}


def numpy_print_doubles_uhf(pq):
    print('from numpy import einsum')
    print('from numpy.typing import NDArray')
    print('from chem.hf.intermediates_builders import Intermediates')
    print()
    from pdaggerq.parser import contracted_strings_to_tensor_terms
    for spin_mix in itertools.product(['a', 'b'], repeat=4):
        out_var = 'doubles_res_' + ''.join(spin_mix)
        print(DOUBLES_HEADER[''.join(spin_mix)])
        print()
        spin_labels = {
            'a': spin_mix[0],
            'b': spin_mix[1],
            'i': spin_mix[2],
            'j': spin_mix[3],
        }
        terms = pq.strings(spin_labels=spin_labels)
        tensor_terms = contracted_strings_to_tensor_terms(terms)

        for my_term in tensor_terms:
            einsum_terms = my_term.einsum_string(
                output_variables=('a', 'b', 'i', 'j'),
                update_val=out_var
            )
            for print_term in einsum_terms.split('\n'):
                print(f"{TAB}{print_term}")
        print(f'{TAB}return {out_var}\n\n')


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
    do_energy = False
    do_singles = False
    do_doubles = True

    if do_energy is True:
        pq = build_energy()
        numpy_print_energy_uhf(pq)

    if do_singles is True:
        pq = build_singles()
        numpy_print_singles_uhf(pq)

    elif do_doubles is True:
        pq = build_doubles()
        numpy_print_doubles_uhf(pq)


if __name__ == "__main__":
    main()

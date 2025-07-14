from collections.abc import Iterable, Sequence
from pdaggerq.parser import contracted_strings_to_tensor_terms
from enum import Enum, auto

TAB = '    '


class DefineSections(Enum):
    FOCK = auto()
    FLUCTUATION = auto()
    IDENTITY = auto()
    SLICES = auto()
    CC_AMPS = auto()
    LAMBDA_AMPS = auto()


def print_imports() -> None:
    print('from numpy import einsum')
    print('from numpy.typing import NDArray')
    print('from chem.hf.ghf_data import GHF_Data')
    print('from chem.ccsd.ghf_ccsd import GHF_CCSD_Data')
    print('from chem.meta.coordinates import Descartes')


def print_function_header(
    quantity: str,
    comment: str = '',
    defines_exclude: set[DefineSections] | None = None,
    extra_definitions: None | Iterable[str] = None,
    extra_arguments: None | Iterable[str] = None,
) -> None:
    if not quantity.isidentifier():
        raise ValueError('Argument must be a valid python isidentifier.')

    body = f'''\n\ndef get_{quantity}(
    ghf_data: GHF_Data,
    ghf_ccsd_data: GHF_CCSD_Data,'''

    if extra_arguments is not None:
        for argument in extra_arguments:
            body += f'\n{TAB}{argument},'

    body += '''
) -> NDArray:'''

    if comment != '':
        body += f'''
    """ {comment} """'''

    if extra_definitions is not None:
        for definition in extra_definitions:
            body += f'\n{TAB}{definition}'

    if defines_exclude is None:
        defines_exclude = set()

    if DefineSections.FOCK not in defines_exclude:
        body += '''
    f = ghf_data.f'''

    if DefineSections.FLUCTUATION not in defines_exclude:
        body += '''
    g = ghf_data.g'''

    if DefineSections.IDENTITY not in defines_exclude:
        body += '''
    kd =  ghf_data.identity_singles'''

    if DefineSections.SLICES not in defines_exclude:
        body += '''
    v = ghf_data.v
    o = ghf_data.o'''

    if DefineSections.CC_AMPS not in defines_exclude:
        body += '''
    t1 = ghf_ccsd_data.t1
    t2 = ghf_ccsd_data.t2'''

    if DefineSections.LAMBDA_AMPS not in defines_exclude:
        body += '''
    if ghf_ccsd_data.lmbda is None:
        raise RuntimeError("Lambda amplitues missing in GHF_CCSD_Data")
    l1 = ghf_ccsd_data.lmbda.l1
    l2 = ghf_ccsd_data.lmbda.l2'''

    body += f'\n{TAB}'

    print(body)


def print_to_numpy(
    pq,
    tensor_name: str,
    tensor_subscripts: Sequence[str],
    defines_exclude: set[DefineSections] | None = None,
    extra_definitions: Iterable[str] | None = None,
    extra_arguments: Iterable[str] | None = None,
) -> None:
    terms = pq.strings()
    tensor_terms = contracted_strings_to_tensor_terms(terms)

    print_function_header(
        quantity=tensor_name,
        comment=f'tensor_subscripts: {tensor_subscripts}',
        defines_exclude=defines_exclude,
        extra_definitions=extra_definitions,
        extra_arguments=extra_arguments,
    )

    out_var = tensor_name
    for my_term in tensor_terms:
        einsum_terms = my_term.einsum_string(
            output_variables=tuple(tensor_subscripts),
            update_val=out_var,
        )
        for print_term in einsum_terms.split('\n'):
            print(f"{TAB}{print_term}")

    print(f'{TAB}return {out_var}')

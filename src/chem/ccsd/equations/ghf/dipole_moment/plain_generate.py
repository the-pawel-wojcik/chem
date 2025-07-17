import pdaggerq
from pdaggerq.parser import contracted_strings_to_tensor_terms


print('CC dipole moment: ', end='')
print('<Λ|μ|CC>')

pq = pdaggerq.pq_helper('fermi')
pq.set_left_operators([['1'], ['l1'], ['l2']])
pq.add_st_operator(1.0, ['h'], ['t1', 't2'])
pq.simplify()
terms = pq.strings()
for term in terms:
    print(term)

tensor_terms = contracted_strings_to_tensor_terms(terms)
for my_term in tensor_terms:
    einsum_terms = my_term.einsum_string(
        output_variables=(),
        update_val='mu_cc',
    )
    for print_term in einsum_terms.split('\n'):
        print(print_term)

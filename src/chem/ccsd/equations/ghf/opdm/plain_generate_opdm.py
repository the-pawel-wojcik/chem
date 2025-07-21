import pdaggerq
from pdaggerq.parser import contracted_strings_to_tensor_terms


print("One-particle density matrix")
print("<Λ|p†q|CC>")
pq = pdaggerq.pq_helper('fermi')
pq.set_left_operators([['1'], ['l1'], ['l2']])
pq.add_st_operator(1.0, ['a*(p)', 'a(1)'], ['t1', 't2'])
pq.simplify()
terms = pq.strings()
for term in terms:
    print(term)
tensor_terms = contracted_strings_to_tensor_terms(terms)
for my_term in tensor_terms:
    einsum_terms = my_term.einsum_string(
        output_variables=('t', 'u'),
        update_val='opdm',
    )
    for print_term in einsum_terms.split('\n'):
        print(f"{print_term}")

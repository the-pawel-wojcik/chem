from chem.hf.electronic_structure import scf
from chem.hf.intermediates_builders import extract_intermediates
from chem.ccsd.uhf_ccsd import UHF_CCSD


mol, scf_energy, wfn = scf()
intermediates = extract_intermediates(wfn)
ccsd = UHF_CCSD(intermediates)
ccsd.verbose = 1
ccsd.solve_cc_equations()

nuclear_repulsion_energy = mol.nuclear_repulsion_energy()
uhf_ccsd_total = ccsd.get_energy() + nuclear_repulsion_energy
print(f'Final UHF CCSD energy = {uhf_ccsd_total:.6f}')

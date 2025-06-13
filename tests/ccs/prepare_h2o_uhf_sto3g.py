import pickle

import psi4
from psi4.core import Molecule, Wavefunction
from chem.hf.intermediates_builders import extract_intermediates


def h2o_UHF_sto3g() -> tuple[Molecule, float, Wavefunction]:
    mol = psi4.geometry("""
    0 1
    O1  0.0000000   0.0000000   0.1271610
    H2  0.0000000   0.7580820  -0.5086420
    H3  0.0000000  -0.7580820  -0.5086420
    symmetry c1
    """)

    psi4.set_options({
        'basis': 'sto-3g',
        'scf_type': 'pk',
        'e_convergence': 1e-12,
        'd_convergence': 1e-12,
    })
    psi4.core.be_quiet()

    energy, wfn = psi4.energy('SCF', molecule=mol, return_wfn=True)
    return mol, energy, wfn


if __name__ == "__main__":
    mol, uhf_energy, wfn = h2o_UHF_sto3g()
    nuclear_repulsion_energy = mol.nuclear_repulsion_energy()
    uhf_data = extract_intermediates(wfn)
    data = {
        "nuclear_repulsion_energy": nuclear_repulsion_energy,
        "uhf_energy": uhf_energy,
        "uhf_data": uhf_data,
    }
    with open('pickles/h2o_uhf_sto3g.pickle', 'wb') as pickle_file:
        pickle.dump(data, pickle_file,)

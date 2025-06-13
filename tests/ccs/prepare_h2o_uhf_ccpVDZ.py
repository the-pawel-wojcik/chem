import pickle

import psi4
from psi4.core import Molecule, Wavefunction
from chem.hf.intermediates_builders import extract_intermediates


def h2o_UHF_ccpVDZ() -> tuple[Molecule, float, Wavefunction]:
    mol = psi4.geometry("""
    0 1
    O1	0.00000   0.00000   0.11572
    H2	0.00000   0.74879  -0.46288
    H3	0.00000  -0.74879  -0.46288
    symmetry c1
    """)

    psi4.set_options({
        'basis': 'cc-pVDZ',
        'scf_type': 'pk',
        'e_convergence': 1e-12,
        'd_convergence': 1e-12,
    })
    psi4.core.be_quiet()

    energy, wfn = psi4.energy('SCF', molecule=mol, return_wfn=True)
    return mol, energy, wfn


if __name__ == "__main__":
    mol, uhf_energy, wfn = h2o_UHF_ccpVDZ()
    nuclear_repulsion_energy = mol.nuclear_repulsion_energy()
    uhf_data = extract_intermediates(wfn)
    data = {
        "nuclear_repulsion_energy": nuclear_repulsion_energy,
        "uhf_energy": uhf_energy,
        "uhf_data": uhf_data,
    }
    with open('pickles/h2o_uhf_ccpVDZ.pickle', 'wb') as pickle_file:
        pickle.dump(data, pickle_file,)

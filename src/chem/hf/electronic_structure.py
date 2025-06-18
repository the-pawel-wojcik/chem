import psi4
from psi4.core import Molecule, Wavefunction


def scf() -> tuple[Molecule, float, Wavefunction]:
    """ For water at the cc-pVDZ geometry and in that basis set. """
    return hf(geometry="""
    0 1
    O1	0.00000   0.00000   0.11572
    H2	0.00000   0.74879  -0.46288
    H3	0.00000  -0.74879  -0.46288
    symmetry c1
    """, basis='cc-pvdz',)


def hf(geometry: str, basis: str) -> tuple[Molecule, float, Wavefunction]:
    """ The geometry string got to look like this
    ```
    0 1
    O1	0.00000   0.00000   0.11572
    H2	0.00000   0.74879  -0.46288
    H3	0.00000  -0.74879  -0.46288
    symmetry c1
    ```
    """

    mol: Molecule = psi4.geometry(geometry)

    psi4.set_options({'basis': basis,
                      'scf_type': 'pk',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12})

    psi4.core.be_quiet()

    # compute the Hartree-Fock energy and wavefunction
    energy, wfn = psi4.energy('SCF', molecule=mol, return_wfn=True)

    return mol, energy, wfn

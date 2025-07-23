from chem.ccsd.ghf_ccsd import GHF_CCSD, GHF_CCSD_Config
from chem.hf.containers import ResultHF
from chem.hf.electronic_structure import hf
from chem.hf.ghf_data import wfn_to_GHF_Data
from chem.meta.coordinates import Descartes
from psi4.core import Molecule


def get_nuclear_electric_dipole_moment(mol: Molecule) -> dict[Descartes, float]:
    atoms = mol.geometry().to_array()
    nEDM = {direction: 0.0 for direction in Descartes}
    for idx, atom in enumerate(atoms):
        charge = mol.charge(idx)
        nEDM[Descartes.x] += charge * atom[0]
        nEDM[Descartes.y] += charge * atom[1]
        nEDM[Descartes.z] += charge * atom[2]
    return nEDM


def main():
    """ Geometry from CCCBDB: HF/STO-3G """
    geometry = """
    0 1
    O  0.0  0.0000000  0.1271610
    H  0.0  0.7580820 -0.5086420
    H  0.0 -0.7580820 -0.5086420
    symmetry c1
    """
    basis='sto-3g'
    print(f'Molecule specified with: {geometry}')
    print(f'Basis set to {basis}')

    print('\n# SCF\n')
     
    print(f'Solving SCF.')
    hf_result: ResultHF = hf(geometry=geometry, basis=basis)
    print('SCF converged.')
    nre: float = hf_result.molecule.nuclear_repulsion_energy()
    scf_energy = hf_result.hf_energy
    strfmt = '25s'
    fltfmt = ' 18.12f'
    print(f'{"Nuclear repulsion energy:":{strfmt}} {nre:{fltfmt}} Ha')
    print(f'{"SCF energy:":{strfmt}} {scf_energy + nre:{fltfmt}} Ha')

    print('\n# CC\n')

    ghf_data = wfn_to_GHF_Data(hf_result.wfn)
    ccsd = GHF_CCSD(
        ghf_data,
        config=GHF_CCSD_Config(
            verbose=1,
            use_diis=False,
            max_iterations=100,
            energy_convergence=1e-12,
            residuals_convergence=1e-12,
            t_amp_print_threshold=5e-2,
        )
    )
    ccsd.solve_cc_equations()
    ccsd_energy = ccsd.get_energy()
    print(f'{"GHF-CCSD energy:":{strfmt}} {ccsd_energy + nre:{fltfmt}} Ha')

    print('\n# CC Properties\n')

    ccsd.solve_lambda_equations()
    ccsd.print_no_occupations()
    eEDM = ccsd.get_electronic_electric_dipole_moment()
    print("The electronic part of the electric dipole moment:")
    for coordinate, value in eEDM.items():
        print(f'{coordinate}: {value:{fltfmt}}')
    nEDM = get_nuclear_electric_dipole_moment(hf_result.molecule)
    print("The nuclear part of the electric dipole moment:")
    for coordinate, value in nEDM.items():
        print(f'{coordinate}: {value:{fltfmt}}')
    print("The electric dipole moment:")
    for coordinate in Descartes:
        print(f'{coordinate}: {eEDM[coordinate] + nEDM[coordinate]:{fltfmt}}')


if __name__ == "__main__":
    main()

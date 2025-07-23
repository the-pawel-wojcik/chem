from chem.ccsd.ghf_ccsd import GHF_CCSD, GHF_CCSD_Config
from chem.hf.containers import ResultHF
from chem.hf.electronic_structure import hf
from chem.hf.ghf_data import wfn_to_GHF_Data


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
    print(f'Solving SCF/{basis} for molecule specified with: {geometry}')
    hf_result: ResultHF = hf(geometry=geometry, basis=basis)
    print('SCF converged.')
    nre: float = hf_result.molecule.nuclear_repulsion_energy()
    scf_energy = hf_result.hf_energy
    strfmt = '25s'
    fltfmt = ' 18.12f'
    print(f'{"Nuclear repulsion energy:":{strfmt}} {nre:{fltfmt}} Ha')
    print(f'{"SCF energy:":{strfmt}} {scf_energy + nre:{fltfmt}} Ha')

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
    print('Solving GHF-CCSD equations.')
    ccsd.solve_cc_equations()
    print('GHF-CCSD equations solved.')
    ccsd.print_leading_t_amplitudes()
    ccsd_energy = ccsd.get_energy()
    print(f'{"GHF-CCSD energy:":{strfmt}} {ccsd_energy + nre:{fltfmt}} Ha')
    print('Solving GHF-CCSD Lambda equations.')
    ccsd.solve_lambda_equations()
    print('GHF-CCSD equations solved.')
    ccsd.print_leading_lambda_amplitudes()
    ccsd.print_no_occupations()


if __name__ == "__main__":
    main()

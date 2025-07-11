import math

from chem.ccs.containers import UHF_CCS_Lambda_Data
from chem.ccs.uhf_ccs import UHF_CCS
from chem.hf.intermediates_builders import extract_intermediates
from chem.hf.electronic_structure import ResultHF
import numpy as np


def test_cc_equations(water_ccpVDZ: ResultHF) -> None:
    nuclear_repulsion_energy = water_ccpVDZ.molecule.nuclear_repulsion_energy()
    uhf_total_energy = water_ccpVDZ.hf_energy
    uhf_data = extract_intermediates(water_ccpVDZ.wfn)

    uhf_energy = uhf_total_energy - nuclear_repulsion_energy

    ccs = UHF_CCS(scf_data=uhf_data, use_diis=False)
    ccs.verbose = 1
    ccs.solve_cc_equations()
    uhf_ccs_energy = ccs.get_energy() 
    uhf_ccs_total = uhf_ccs_energy + nuclear_repulsion_energy
    print(f'{'UHF energy =':<30s} {uhf_energy:.10f}')
    print(f'{'UHF CCS energy =':<30s} {uhf_ccs_energy:.10f}')
    print(f'{'UHF total energy =':<30s} {uhf_total_energy:.10f}')
    print(f'{'UHF CCS total energy =':<30s} {uhf_ccs_total:.10f}')

    t1_aa_norm = float(np.linalg.norm(ccs.data.t1_aa))
    t1_bb_norm = float(np.linalg.norm(ccs.data.t1_bb))
    assert math.isclose(t1_aa_norm, 0.0, abs_tol=1e-10)
    assert math.isclose(t1_bb_norm, 0.0, abs_tol=1e-10)
    # 9.300820 comes from CCCBDB for water @ HF/cc-pvDZ
    assert math.isclose(9.300820, nuclear_repulsion_energy, abs_tol=1e-6)
    assert math.isclose(-85.327873230, uhf_energy, abs_tol=1e-9)
    assert math.isclose(-85.327873230, uhf_ccs_energy, abs_tol=1e-9)
    assert math.isclose(-76.027053513, uhf_total_energy, abs_tol=1e-9)
    assert math.isclose(-76.027053513, uhf_ccs_total, abs_tol=1e-9)


def test_cc_lambda_equations(water_ccpVDZ: ResultHF) -> None:
    uhf_data = extract_intermediates(water_ccpVDZ.wfn)
    ccs = UHF_CCS(scf_data=uhf_data, use_diis=False)
    ccs.verbose = 1
    ccs.solve_cc_equations()
    ccs.solve_lambda_equations()
    uhf_ccs_lambda_data: UHF_CCS_Lambda_Data = ccs.cc_lambda_data
    lambda_norm = float(
        sum(
            np.linalg.norm(spin_component) for spin_component in
            [uhf_ccs_lambda_data.l1_aa, uhf_ccs_lambda_data.l1_bb]
        )
    )
    assert math.isclose(lambda_norm, 0., abs_tol=1e-8)

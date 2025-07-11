from chem.ccs.uhf_ccs import UHF_CCS
from chem.hf.intermediates_builders import extract_intermediates
from chem.hf.electronic_structure import ResultHF
import numpy as np


def test_residuals(water_sto3g: ResultHF) -> None:
    uhf_data = extract_intermediates(water_sto3g.wfn)
    ccs = UHF_CCS(scf_data=uhf_data, use_diis=False)
    residuals = ccs._calculate_residuals()
    residuals_norm = sum(np.linalg.norm(res) for res in residuals.values())
    assert np.isclose(residuals_norm, 0., atol=1e-6)


def test_energy(water_sto3g: ResultHF) -> None:
    nuclear_repulsion_energy = water_sto3g.molecule.nuclear_repulsion_energy()
    uhf_data = extract_intermediates(water_sto3g.wfn)

    ccs = UHF_CCS(scf_data=uhf_data, use_diis=False)
    ccs.verbose = 1
    ccs.solve_cc_equations()
    uhf_ccs_total = ccs.get_energy() + nuclear_repulsion_energy
    assert np.isclose(-74.96590119, uhf_ccs_total, atol=1e-6)

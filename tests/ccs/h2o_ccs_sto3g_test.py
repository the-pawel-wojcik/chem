import pickle

from chem.ccs.uhf_ccs import UHF_CCS
from chem.hf.intermediates_builders import Intermediates
import numpy as np


def test_residuals():
    with open('pickles/h2o_uhf_sto3g.pickle', 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    uhf_data: Intermediates = data['uhf_data']
    ccs = UHF_CCS(scf_data=uhf_data, use_diis=False)
    residuals = ccs._calculate_residuals()
    residuals_norm = sum(np.linalg.norm(res) for res in residuals.values())
    assert np.isclose(residuals_norm, 0., atol=1e-6)


def test_energy():
    with open('pickles/h2o_uhf_sto3g.pickle', 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    nuclear_repulsion_energy: float = data['nuclear_repulsion_energy']
    uhf_data: Intermediates = data['uhf_data']

    ccs = UHF_CCS(scf_data=uhf_data, use_diis=False)
    ccs.verbose = 1
    ccs.solve_cc_equations()
    uhf_ccs_total = ccs.get_energy() + nuclear_repulsion_energy
    assert np.isclose(-74.96590119, uhf_ccs_total, atol=1e-6)


if __name__ == "__main__":
    test_residuals()
    test_energy()

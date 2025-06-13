import math
import pickle

from chem.ccs.uhf_ccs import UHF_CCS
from chem.hf.intermediates_builders import Intermediates
import numpy as np


def test_cc_equations():
    with open('pickles/h2o_uhf_ccpVDZ.pickle', 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    nuclear_repulsion_energy: float = data['nuclear_repulsion_energy']
    uhf_total_energy: float = data['uhf_energy']
    uhf_data: Intermediates = data['uhf_data']

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
    assert math.isclose(9.3007568224, nuclear_repulsion_energy, abs_tol=1e-10)
    assert math.isclose(-85.3278103352, uhf_energy, abs_tol=1e-10)
    assert math.isclose(-85.3278103352, uhf_ccs_energy, abs_tol=1e-10)
    assert math.isclose(-76.0270535127, uhf_total_energy, abs_tol=1e-8)
    assert math.isclose(-76.0270535127, uhf_ccs_total, abs_tol=1e-8)


if __name__ == "__main__":
    test_cc_equations()

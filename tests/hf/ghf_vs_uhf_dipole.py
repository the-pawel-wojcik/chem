import pytest

from chem.hf.containers import ResultHF
from chem.hf.electronic_structure import hf
from chem.hf.ghf_data import GHF_Data, wfn_to_GHF_Data
from chem.hf.intermediates_builders import Intermediates, extract_intermediates
from chem.meta.coordinates import Descartes
import numpy as np


@pytest.fixture(scope='session')
def hf_result() -> ResultHF:
    """ Geometry from CCCBDB: HF/STO-3G """
    geometry = """
    0 1
    O  0.0  0.0000000  0.1271610
    H  0.0  0.7580820 -0.5086420
    H  0.0 -0.7580820 -0.5086420
    symmetry c1
    """
    hf_result = hf(geometry=geometry, basis='sto-3g')
    return hf_result


def test_uhf_vs_ghf_dipole(hf_result: ResultHF)-> None:
    ghf_data: GHF_Data = wfn_to_GHF_Data(hf_result.wfn)
    ghf_orbital_dipoles_z = ghf_data.mu[Descartes.z].diagonal()[ghf_data.o]
    ghf = ghf_orbital_dipoles_z[::2]
    uhf_data: Intermediates = extract_intermediates(hf_result.wfn)
    uhf_orbital_dipoles_az = uhf_data.mua_z.diagonal()[uhf_data.oa]
    uhf = uhf_orbital_dipoles_az
    assert np.allclose(ghf, uhf)

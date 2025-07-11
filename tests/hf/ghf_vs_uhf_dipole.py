from chem.hf.containers import ResultHF
from chem.hf.ghf_data import GHF_Data, wfn_to_GHF_Data
from chem.hf.intermediates_builders import Intermediates, extract_intermediates
from chem.meta.coordinates import Descartes
import numpy as np


def test_uhf_vs_ghf_dipole(water_sto3g: ResultHF)-> None:
    ghf_data: GHF_Data = wfn_to_GHF_Data(water_sto3g.wfn)
    ghf_orbital_dipoles_z = ghf_data.mu[Descartes.z].diagonal()[ghf_data.o]
    ghf = ghf_orbital_dipoles_z[::2]
    uhf_data: Intermediates = extract_intermediates(water_sto3g.wfn)
    uhf_orbital_dipoles_az = uhf_data.mua_z.diagonal()[uhf_data.oa]
    uhf = uhf_orbital_dipoles_az
    assert np.allclose(ghf, uhf)

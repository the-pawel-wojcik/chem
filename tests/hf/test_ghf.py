import pytest

from chem.hf.ghf_data import wfn_to_GHF_Data, GHF_Data
from chem.hf.electronic_structure import hf, ResultHF
from chem.meta.coordinates import Descartes
from numpy import einsum
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


def get_hf_energy(ghf_data: GHF_Data) -> float:
    f = ghf_data.f
    g = ghf_data.g
    o = ghf_data.o
    energy = 1.00 * einsum('ii', f[o, o])
    energy += -0.50 * einsum('jiji', g[o, o, o, o])
    return float(energy)


def test_constructor(hf_result: ResultHF):
    wfn_to_GHF_Data(hf_result.wfn)


def test_energies(hf_result: ResultHF):
    ghf_data = wfn_to_GHF_Data(hf_result.wfn)
    assert ghf_data.f.shape == (14, 14)
    fock_diagonal = np.array([-20.25157699, -20.25157699, -1.25754837,
    -1.25754837, -0.59385451, -0.59385451, -0.45972972, -0.45972972,
    -0.39261692, -0.39261692, 0.58179264, 0.58179264, 0.69267285, 0.69267285])
    assert np.allclose(fock_diagonal, ghf_data.f.diagonal(), atol=1e-7)
    energy = get_hf_energy(ghf_data)
    assert np.isclose(-83.87226577852897, energy, atol=1e-7)
    nre = hf_result.molecule.nuclear_repulsion_energy()
    assert np.isclose(nre, 8.906479, atol=1e-6)  # value from CCCBDB
    assert np.isclose(energy + nre, -74.965901, atol=1e-6)  # value from CCCBDB


def test_dipoles_shapes(hf_result: ResultHF):
    ghf_data = wfn_to_GHF_Data(hf_result.wfn)
    for direction in Descartes:
        mu_component =  ghf_data.mu[direction]
        assert mu_component.shape == (14, 14)

# TODO: test dipole values too

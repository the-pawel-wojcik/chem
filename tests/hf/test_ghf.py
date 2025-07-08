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
    O1	0.0000   0.0000   0.1272
    H2	0.0000   0.7581  -0.5086
    H3	0.0000  -0.7581  -0.5086
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
    fock_diagonal = np.array([
        -20.25157669, -20.25157669,  -1.25754098, -1.25754098,  -0.59385141,
        -0.59385141, -0.45972452,  -0.45972452, -0.3926153 ,  -0.3926153,
        0.58177897, 0.58177897, 0.69266369,   0.69266369
    ])
    assert np.allclose(fock_diagonal, ghf_data.f.diagonal(), atol=1e-7)
    energy = get_hf_energy(ghf_data)
    assert np.isclose(-83.87226577852897, energy, atol=1e-7)
    nre = hf_result.molecule.nuclear_repulsion_energy()
    # assert np.isclose(nre, 8.906479, atol=1e-5)  # value from CCCBDB
    assert np.isclose(energy + nre, -74.965901, atol=1e-5)  # value from CCCBDB


def test_dipoles_shapes(hf_result: ResultHF):
    ghf_data = wfn_to_GHF_Data(hf_result.wfn)
    for direction in Descartes:
        mu_component =  ghf_data.mu[direction]
        assert mu_component.shape == (14, 14)

# TODO: test dipole values too

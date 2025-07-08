from chem.hf.electronic_structure import hf, ResultHF
from chem.hf.ghf_data import wfn_to_GHF_Data, GHF_Data
from chem.hf.util import turn_GHF_Data_to_GHF_ov_data
import pytest
import numpy as np

from chem.meta.ghf_ccsd_mbe import GHF_CCSD_MBE


@pytest.fixture(scope='session')
def ghf_data() -> GHF_Data:
    """ Geometry from CCCBDB: HF/STO-3G """
    geometry = """
    0 1
    O1	0.0000   0.0000   0.1272
    H2	0.0000   0.7581  -0.5086
    H3	0.0000  -0.7581  -0.5086
    symmetry c1
    """
    hf_result: ResultHF = hf(geometry=geometry, basis='sto-3g')
    return wfn_to_GHF_Data(hf_result.wfn)


def test_dims_shapes_slices(ghf_data: GHF_Data) -> None:
    ghf_ov_data = turn_GHF_Data_to_GHF_ov_data(ghf_data)
    dims = ghf_ov_data.get_dims()
    shapes = ghf_ov_data.get_shapes()

    assert dims['singles'] == 40
    assert shapes['singles'] == (4, 10)

    assert dims['doubles'] == 1600
    assert shapes['doubles'] == (4, 4, 10, 10)

    slices = ghf_ov_data.get_slices()
    assert slices['singles'] == slice(0, 40, None)
    assert slices['doubles'] == slice(40, 1640, None)

    assert ghf_ov_data.get_vector_dim() == 1640


def test_go_around(ghf_data: GHF_Data):
    """ Test that the conversion from an NDArray to MBE and back is an identity
    transformation. """
    ghf_ov_data = turn_GHF_Data_to_GHF_ov_data(ghf_data)
    full_mbe_dim = ghf_ov_data.get_vector_dim()
    TODAY = 20250708
    rng = np.random.default_rng(seed=TODAY)
    for _ in range(10):
        test_vector = rng.random(size=full_mbe_dim)
        test_vector_mbe  = GHF_CCSD_MBE.from_NDArray(test_vector, ghf_ov_data)
        all_around = test_vector_mbe.flatten()

        assert test_vector.shape == all_around.shape
        assert np.allclose(test_vector, all_around)


def test_GHF_ov_data_constructor(ghf_data: GHF_Data):
    turn_GHF_Data_to_GHF_ov_data(ghf_data)

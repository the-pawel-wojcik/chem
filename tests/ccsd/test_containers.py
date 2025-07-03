from chem.hf.electronic_structure import hf
from chem.hf.intermediates_builders import (
    Intermediates,
    extract_intermediates,
)
from chem.hf.util import turn_NDArray_to_Spin_MBE, turn_UHF_Data_to_UHF_ov_data
from chem.meta.spin_mbe import E1_spin, E2_spin
import pytest
import numpy as np


@pytest.fixture(scope='session')
def intermediates() -> Intermediates:
    """ Geometry from CCCBDB: HF/STO-3G """
    geometry = """
    0 1
    O1	0.0000   0.0000   0.1272
    H2	0.0000   0.7581  -0.5086
    H3	0.0000  -0.7581  -0.5086
    symmetry c1
    """
    hf_result = hf(geometry=geometry, basis='sto-3g')
    return extract_intermediates(hf_result.wfn)


def test_dims_shapes_slices(intermediates: Intermediates):
    uhf_ov_data = turn_UHF_Data_to_UHF_ov_data(intermediates)
    dims = uhf_ov_data.get_dims()
    shapes = uhf_ov_data.get_shapes()

    for e1 in E1_spin:
        assert dims[e1] == 10
        assert shapes[e1] == (2, 5)

    for e2 in E2_spin:
        assert dims[e2] == 100
        assert shapes[e2] == (2, 2, 5, 5)

    slices = uhf_ov_data.get_slices()
    assert slices[E1_spin.aa] == slice(0, 10, None)
    assert slices[E1_spin.bb] == slice(10, 20, None)
    assert slices[E2_spin.aaaa] == slice(20, 120, None)
    assert slices[E2_spin.abab] == slice(120, 220, None)
    assert slices[E2_spin.abba] == slice(220, 320, None)
    assert slices[E2_spin.baab] == slice(320, 420, None)
    assert slices[E2_spin.baba] == slice(420, 520, None)
    assert slices[E2_spin.bbbb] == slice(520, 620, None)


def test_get_dims(intermediates: Intermediates):
    uhf_ov_data = turn_UHF_Data_to_UHF_ov_data(intermediates)
    assert uhf_ov_data.get_singles_dim() == 20
    assert uhf_ov_data.get_doubles_dim() == 600
    assert uhf_ov_data.get_vector_dim() == 620


def test_go_around(intermediates: Intermediates):
    """ Test that the conversion from an NDArray to MBE and back is an identity
    transformation. """
    uhf_ov_data = turn_UHF_Data_to_UHF_ov_data(intermediates)
    full_mbe_dim = uhf_ov_data.get_vector_dim()
    TODAY = 20250627
    rng = np.random.default_rng(seed=TODAY)
    for _ in range(10):
        test_vector = rng.random(size=full_mbe_dim)
        test_vector_mbe  = turn_NDArray_to_Spin_MBE(test_vector, uhf_ov_data)
        all_around = test_vector_mbe.flatten()

        assert test_vector.shape == all_around.shape
        assert np.allclose(test_vector, all_around)


def test_block_match(intermediates: Intermediates):
    """ Test that blocks land in the right spots when MBE-ed. """
    uhf_ov_data = turn_UHF_Data_to_UHF_ov_data(intermediates)
    full_mbe_dim = uhf_ov_data.get_vector_dim()
    slices = uhf_ov_data.get_slices()
    shapes = uhf_ov_data.get_shapes()
    TODAY = 20250627
    rng = np.random.default_rng(seed=TODAY)
    for _ in range(1):
        test_vector = rng.random(size=full_mbe_dim)
        test_vector_mbe = turn_NDArray_to_Spin_MBE(test_vector, uhf_ov_data)
        for block in E1_spin:
            manual_subblock = test_vector[slices[block]].reshape(shapes[block])
            assert np.allclose(manual_subblock, test_vector_mbe.singles[block])

        for block in E2_spin:
            manual_subblock = test_vector[slices[block]].reshape(shapes[block])
            assert np.allclose(manual_subblock, test_vector_mbe.doubles[block])


def test_UHF_ov_data_constructor(intermediates: Intermediates):
    turn_UHF_Data_to_UHF_ov_data(intermediates)

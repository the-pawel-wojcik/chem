from chem.ccsd.containers import E1_spin, E2_spin, Spin_MBE
from chem.hf.electronic_structure import hf
from chem.hf.intermediates_builders import Intermediates, extract_intermediates
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
    vector = Spin_MBE()
    dims, slices, shapes = vector.find_dims_slices_shapes(
        uhf_scf_data=intermediates,
    )
    for e1 in E1_spin:
        assert dims[e1] == 10
        assert shapes[e1] == (2, 5)

    for e2 in E2_spin:
        assert dims[e2] == 100
        assert shapes[e2] == (2, 2, 5, 5)

    assert slices[E1_spin.aa] == slice(0, 10, None)
    assert slices[E1_spin.bb] == slice(10, 20, None)
    assert slices[E2_spin.aaaa] == slice(20, 120, None)
    assert slices[E2_spin.abab] == slice(120, 220, None)
    assert slices[E2_spin.abba] == slice(220, 320, None)
    assert slices[E2_spin.baab] == slice(320, 420, None)
    assert slices[E2_spin.baba] == slice(420, 520, None)
    assert slices[E2_spin.bbbb] == slice(520, 620, None)


def test_get_dims(intermediates: Intermediates):
    vector = Spin_MBE()
    dims, _, _ = vector.find_dims_slices_shapes(
        uhf_scf_data=intermediates,
    )
    assert vector.get_singles_dim(dims) == 20
    assert vector.get_doubles_dim(dims) == 600
    assert vector.get_vector_dim(dims) == 620


def test_go_around(intermediates: Intermediates):
    """ Test that the conversion from an NDArray to MBE and back is an identity
    transformation. """
    dims, _, _ = Spin_MBE.find_dims_slices_shapes(
        uhf_scf_data=intermediates,
    )

    full_mbe_dim = sum(block_dim for block_dim in dims.values())

    TODAY = 20250627
    rng = np.random.default_rng(seed=TODAY)
    for _ in range(10):
        test_vector = rng.random(size=full_mbe_dim)
        test_vector_mbe = Spin_MBE.from_flattened_NDArray(
                test_vector, intermediates)
        all_around = test_vector_mbe.flatten()

        assert test_vector.shape == all_around.shape
        assert np.allclose(test_vector, all_around)

import pytest

from chem.hf.electronic_structure import hf, ResultHF
from chem.hf.ghf_data import wfn_to_GHF_Data, GHF_Data
from chem.hf.util import turn_GHF_Data_to_GHF_ov_data
from chem.meta.ghf_ccsd_mbe import GHF_CCSD_MBE, GHF_ov_data
import numpy as np


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


def test_matmul():
    """ Test that the flattened MBE matmul works as planned. """
    TODAY = 20250708

    # create a model matrix of a "singles, doubles" structure
    rng = np.random.default_rng(seed=TODAY)
    no = 10
    nv = 5
    dim_s = nv * no
    dim_d = nv * nv * no * no
    dim_ss = dim_s * dim_s
    dim_sd = dim_s * dim_d
    dim_ds = dim_d * dim_s
    dim_dd = dim_d * dim_d
    matrix_mbe = {
        'ss': rng.random(size=dim_ss).reshape(nv, no, nv, no),
        'sd': rng.random(size=dim_sd).reshape(nv, no, nv, nv, no, no),
        'ds': rng.random(size=dim_ds).reshape(nv, nv, no, no, nv, no),
        'dd': rng.random(size=dim_dd).reshape(nv, nv, no, no, nv, nv, no, no),
    }
    # This is the tested transformation
    # The point of this test to check if a matrix reshaped like this give the
    # correct outcome of the multiplication by a similarily "flattened" mbe
    # vector.
    matrix_nda = np.block([
        [
            matrix_mbe['ss'].reshape(dim_s, dim_s),
            matrix_mbe['sd'].reshape(dim_s, dim_d),
        ],
        [
            matrix_mbe['ds'].reshape(dim_d, dim_s),
            matrix_mbe['dd'].reshape(dim_d, dim_d),
        ],
    ])

    ghf_ov_data = GHF_ov_data(nmo=no+nv, no=no, nv=nv)
    full_mbe_dim = ghf_ov_data.get_vector_dim()
    for _ in range(10):
        test_vector = rng.random(size=full_mbe_dim)

        # This is the tested mat-mul
        # This "flattened" matrix multiplication is tested to result in the
        # same outcome as what the 'eigsum' version produces
        nda_result = matrix_nda @ test_vector

        test_vector_mbe  = GHF_CCSD_MBE.from_NDArray(test_vector, ghf_ov_data)
        mbe_result = GHF_CCSD_MBE(
            singles=(
                np.einsum(
                    'aibj,bj->ai',
                    matrix_mbe['ss'],
                    test_vector_mbe.singles,
                )
                +
                np.einsum(
                    'aibckj,bckj->ai',
                    matrix_mbe['sd'],
                    test_vector_mbe.doubles,
                )
            ),
            doubles=(
                np.einsum(
                    'abjick,ck->abji',
                    matrix_mbe['ds'],
                    test_vector_mbe.singles,
                )
                +
                np.einsum(
                    'abjicdlk,cdlk->abji',
                    matrix_mbe['dd'],
                    test_vector_mbe.doubles,
                )
            ),
        )
        assert np.allclose(mbe_result.flatten(), nda_result)


def test_GHF_ov_data_constructor(ghf_data: GHF_Data):
    turn_GHF_Data_to_GHF_ov_data(ghf_data)

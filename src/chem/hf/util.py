from numpy.typing import NDArray
from chem.hf.intermediates_builders import Intermediates
from chem.meta.spin_mbe import E1_spin, E2_spin, Spin_MBE, UHF_ov_data


def turn_UHF_Data_to_UHF_ov_data(uhf_data: Intermediates) -> UHF_ov_data:
    nmo = uhf_data.nmo
    noa = uhf_data.noa
    nva = nmo - noa
    nob = uhf_data.nob
    nvb = nmo - nob
    return UHF_ov_data(nmo, noa, nva, nob, nvb)


def turn_NDArray_to_Spin_MBE(
    vector: NDArray,
    uhf_ov_data: UHF_ov_data,
) -> Spin_MBE:
    assert len(vector.shape) == 1
    assert vector.shape[0] == uhf_ov_data.get_vector_dim()

    slices = uhf_ov_data.get_slices()
    shapes = uhf_ov_data.get_shapes()
    mbe = Spin_MBE()
    for spin_block in E1_spin:
        sub_vec = vector[slices[spin_block]]
        mbe.singles[spin_block] = sub_vec.reshape(shapes[spin_block])

    for spin_block in E2_spin:
        sub_vec = vector[slices[spin_block]]
        mbe.doubles[spin_block] = sub_vec.reshape(shapes[spin_block])

    return mbe


def turn_NDArray_to_Spin_MBE_using_uhf_data(
    vector: NDArray,
    uhf_data: Intermediates,
) -> Spin_MBE:
    uhf_ov_data = turn_UHF_Data_to_UHF_ov_data(uhf_data)
    return turn_NDArray_to_Spin_MBE(vector, uhf_ov_data)

from chem.ccsd.containers import E1_spin, E2_spin, Spin_MBE
from chem.hf.electronic_structure import hf
from chem.hf.intermediates_builders import extract_intermediates


def test_spin_MBE():
    """ Geometry from CCCBDB: HF/STO-3G """
    geometry = """
    0 1
    O1	0.0000   0.0000   0.1272
    H2	0.0000   0.7581  -0.5086
    H3	0.0000  -0.7581  -0.5086
    symmetry c1
    """
    hf_result = hf(geometry=geometry, basis='sto-3g')
    intermediates = extract_intermediates(hf_result.wfn)
    response = Spin_MBE()
    dims, slices, shapes = response.find_dims_slices_shapes(
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

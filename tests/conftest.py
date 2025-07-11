import pytest

from chem.hf.containers import ResultHF
from chem.hf.electronic_structure import hf


@pytest.fixture(scope='module')
def water_sto3g() -> ResultHF:
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


@pytest.fixture(scope='module')
def water_ccpVDZ() -> ResultHF:
    """ Geometry from CCCBDB: HF/cc-pVDZ """
    geometry = """
    0 1
    O  0.0  0.0000000  0.1157190
    H  0.0  0.7487850 -0.4628770
    H  0.0 -0.7487850 -0.4628770
    symmetry c1
    """
    hf_result = hf(geometry=geometry, basis='cc-pVDZ')
    return hf_result

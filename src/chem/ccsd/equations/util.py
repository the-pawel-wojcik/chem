from typing import TypedDict
from chem.ccsd.containers import UHF_CCSD_Data
from chem.hf.intermediates_builders import Intermediates


class GeneratorsInput(TypedDict):
    uhf_scf_data: Intermediates
    uhf_ccsd_data: UHF_CCSD_Data

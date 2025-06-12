from typing import TypedDict
from chem.ccs.containers import UHF_CCS_Data
from chem.hf.intermediates_builders import Intermediates


class UHF_CCS_InputPair(TypedDict):
    uhf_scf_data: Intermediates
    uhf_ccs_data: UHF_CCS_Data

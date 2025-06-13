from typing import TypedDict
from chem.ccs.containers import UHF_CCS_Data, UHF_CCS_Lambda_Data
from chem.hf.intermediates_builders import Intermediates


class UHF_CCS_InputPair(TypedDict):
    uhf_scf_data: Intermediates
    uhf_ccs_data: UHF_CCS_Data


class UHF_CCS_InputTriple(TypedDict):
    uhf_data: Intermediates
    uhf_ccs_data: UHF_CCS_Data
    uhf_ccs_lambda_data: UHF_CCS_Lambda_Data

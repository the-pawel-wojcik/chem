from typing import TypedDict
from chem.ccsd.containers import GHF_CCSD_Data
from chem.hf.ghf_data import GHF_Data


class GHF_Generators_Input(TypedDict):
    ghf_data: GHF_Data
    ghf_ccsd_data: GHF_CCSD_Data

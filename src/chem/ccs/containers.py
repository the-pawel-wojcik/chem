from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass
class UHF_CCS_Lambda_Data:
    l1_aa: NDArray
    l1_bb: NDArray


@dataclass
class UHF_CCS_Data:
    t1_aa: NDArray
    t1_bb: NDArray

    lmbda: UHF_CCS_Lambda_Data | None = None

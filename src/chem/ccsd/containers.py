from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass
class UHF_CCSD_Lambda_Data:
    l1_aa: NDArray
    l1_bb: NDArray
    l2_aaaa: NDArray
    l2_abab: NDArray
    l2_bbbb: NDArray


@dataclass
class UHF_CCSD_Data:
    t1_aa: NDArray
    t1_bb: NDArray
    t2_aaaa: NDArray
    t2_abab: NDArray
    t2_bbbb: NDArray
    t2_abba: NDArray
    t2_baab: NDArray
    t2_baba: NDArray

    lmbda: UHF_CCSD_Lambda_Data | None = None

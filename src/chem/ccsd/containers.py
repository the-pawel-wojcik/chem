from dataclasses import dataclass, field
from enum import StrEnum, auto
import numpy as np
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


class E1_spin(StrEnum):
    aa = auto()
    bb = auto()


class E2_spin(StrEnum):
    aaaa = auto()
    abab = auto()
    abba = auto()
    baab = auto()
    baba = auto()
    bbbb = auto()


@dataclass
class Spin_MBE():
    """ MBE stands for many body expansion. """
    singles: dict[E1_spin, NDArray] = field(default_factory=dict)
    doubles: dict[E2_spin, NDArray] = field(default_factory=dict)

    def pretty_print_mbe(self) -> None:
        with np.printoptions(precision=3, suppress=True):

            print("Singles:")
            for key, value in self.singles.items():
                print(f' {key} shape = {value.shape}')
                print(value)

            print("Doubles:")
            for key, value in self.doubles.items():
                print(f' {key} shape = {value.shape}')
            print("")

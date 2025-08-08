from __future__ import annotations
from dataclasses import dataclass
import itertools
from typing import Callable

from chem.meta.coordinates import Descartes, CARTESIAN
import numpy as np


@dataclass
class Polarizability:
    data: dict[Descartes, dict[Descartes, float]]

    @classmethod
    def from_builder(
        cls,
        builder: Callable[[Descartes, Descartes], float]
    ) -> Polarizability:
        pol = {
            first: {
                second: builder(first, second)
                for second in CARTESIAN
            } for first in CARTESIAN
        }
        return cls(data=pol)

    def __sub__(self, other) -> Polarizability:
        if not isinstance(other, Polarizability):
            msg = f"Don't know how to add to {type(other)}."
            raise ValueError(msg)

        return Polarizability.from_builder(
            builder=lambda first, second: (
                self.data[first][second]
                -
                other.data[first][second]
            ),
        )

    def __add__(self, other) -> Polarizability:
        if not isinstance(other, Polarizability):
            msg = f"Don't know how to add to {type(other)}."
            raise ValueError(msg)

        return Polarizability.from_builder(
            builder=lambda first, second: (
                self.data[first][second]
                +
                other.data[first][second]
            ),
        )

    def __str__(self) -> str:
        pretty = ""
        fmt = ' z12.6f'
        for left, right in itertools.product(CARTESIAN, repeat=2):
            pretty += f'{left}{right}: {self.data[left][right]:{fmt}}\n'
        pretty = pretty[:-1]  # remove the trailing new line
        return pretty

    def isclose(self, other: Polarizability, atol: float = 1e-3) -> bool:
        """
        atol : The absolute tolerance parameter. See docstring of `np.isclose`.
        """
        if not isinstance(other, Polarizability):
            raise ValueError("Only Polarizability comaprison supported.")
        for left, right_dict in self.data.items():
            for right, pol in right_dict.items():
                if not np.isclose(other.data[left][right], pol, atol=atol):
                    return False
        return True

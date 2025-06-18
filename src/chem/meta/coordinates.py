from enum import StrEnum, auto


class Descartes(StrEnum):
    x = auto()
    y = auto()
    z = auto()


CARTESIAN = [Descartes.x, Descartes.y, Descartes.z]

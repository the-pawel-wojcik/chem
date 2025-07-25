import math

from chem.hf.containers import ResultHF
from chem.hf.ghf_data import GHF_Data, wfn_to_GHF_Data
from chem.meta.coordinates import CARTESIAN, Descartes
import numpy as np


def test_dipole_elements_HF_water_sto3g(water_sto3g: ResultHF) -> None:
    ghf_data: GHF_Data = wfn_to_GHF_Data(water_sto3g.wfn)
    o = ghf_data.o
    v = ghf_data.v

    PSI4_DIPOLE = {
        Descartes.x: np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.070510953, 0.0, 0.0, 0.0],
            [0.0, 0.070510953, 0.0, 0.0],
        ]),
        Descartes.y: np.array([
            [0.0, 0.0, -0.055109142700219, 0.0],
            [0.0, 0.0, .0, -0.055109142700219],
            [0.0, 0.0, -0.122193314873649, 0.0],
            [0.0, 0.0, 0.0, -0.122193314873649],
            [-0.810982420713625, 0.0, 0.0, 0.0],
            [0.0, -0.810982420713625, 0.0, 0.0],
            [0.0, 0.0, 0.693483744632154, 0.0],
            [0.0, 0.0, 0.0, 0.693483744632154],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]),
        Descartes.z: np.array([
            [-0.045086293393440, 0.0, 0.0, 0.0],
            [0.0, -0.045086293393440, 0.0, 0.0],
            [-0.111816110709066, 0.0, 0.0, 0.0],
            [0.0, -0.111816110709066, 0.0, 0.0],
            [0.0, 0.0, -0.679589465357873, 0.0],
            [0.0, 0.0, 0.0, -0.679589465357873],
            [0.545551238783468, 0.0, 0.0, 0.0],
            [0.0, 0.545551238783468, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]),
    }

    for direction in Descartes:
        assert np.allclose(
            ghf_data.mu[direction][o, v], PSI4_DIPOLE[direction], atol=1e-8
        )


def test_dipole_HF_water_sto3g(water_sto3g: ResultHF) -> None:
    mol = water_sto3g.molecule
    ghf_data: GHF_Data = wfn_to_GHF_Data(water_sto3g.wfn)

    cccdbd_dipole_au = {
        Descartes.x: 0.0,
        Descartes.y: 0.0,
        Descartes.z: -0.672372,
    }

    psi4_dipole_au_electronic = {
        Descartes.x: 0.0,
        Descartes.y: 0.0,
        Descartes.z: 0.3858840,
    }

    psi4_dipole_au_nuclear = {
        Descartes.x: 0.0,
        Descartes.y: 0.0,
        Descartes.z: -1.0583371,
    }

    psi4_dipole_au_total = {
        direction: (
            psi4_dipole_au_electronic[direction]
            +
            psi4_dipole_au_nuclear[direction]
        ) for direction in Descartes
    }

    o = ghf_data.o
    my_dipole_electronic = {
        direction: float(sum(ghf_data.mu[direction].diagonal()[o]))
        for direction in Descartes
    }

    geo = mol.geometry().np
    my_dipole_nuclear = {
        Descartes.x: float(
            sum(mol.charge(i) * atom[0] for i, atom in enumerate(geo))
        ),
        Descartes.y: float(
            sum(mol.charge(i) * atom[1] for i, atom in enumerate(geo))
        ),
        Descartes.z: float(
            sum(mol.charge(i) * atom[2] for i, atom in enumerate(geo))
        ),
    }

    my_dipole_total = {
        direction: (
            my_dipole_electronic[direction]
            +
            my_dipole_nuclear[direction]
        ) for direction in Descartes
    }

    for direction in Descartes:
        assert np.isclose(
            my_dipole_total[direction],
            cccdbd_dipole_au[direction],
            atol=1e-4,
        )
        assert np.isclose(
            my_dipole_electronic[direction],
            psi4_dipole_au_electronic[direction],
            atol=1e-7,  # all digits printed in the Psi4 output match
        )
        assert np.isclose(
            my_dipole_nuclear[direction],
            psi4_dipole_au_nuclear[direction],
            atol=1e-12,  # as good as you wish, both come from Psi4
        )
        assert np.isclose(
            cccdbd_dipole_au[direction],
            psi4_dipole_au_total[direction],
            atol=1e-4,  # CCCBDB differs a little
        )


def test_dipole_HF_water_ccpVDZ(water_ccpVDZ: ResultHF) -> None:
    mol = water_ccpVDZ.molecule
    ghf_data: GHF_Data = wfn_to_GHF_Data(water_ccpVDZ.wfn)
    psi_dipole_electronic = {
        Descartes.x: 0.0,
        Descartes.y: 0.0,
        Descartes.z: 0.1588552,
    }

    psi_dipole_nulcear = {
        Descartes.x: 0.0,
        Descartes.y: 0.0,
        Descartes.z: -0.9631122,
    }

    o = ghf_data.o
    my_dipole_electronic = {
        direction: float(sum(ghf_data.mu[direction].diagonal()[o]))
        for direction in Descartes
    }

    geo = mol.geometry().np
    my_dipole_nuclear = {
        Descartes.x: float(
            sum(mol.charge(i) * atom[0] for i, atom in enumerate(geo))
        ),
        Descartes.y: float(
            sum(mol.charge(i) * atom[1] for i, atom in enumerate(geo))
        ),
        Descartes.z: float(
            sum(mol.charge(i) * atom[2] for i, atom in enumerate(geo))
        ),
    }

    for coord in CARTESIAN:
        assert math.isclose(
            my_dipole_electronic[coord],
            psi_dipole_electronic[coord],
            abs_tol=1e-7,  # Matches all digits printed by Psi4
        )

        assert math.isclose(
            my_dipole_nuclear[coord],
            psi_dipole_nulcear[coord],
            abs_tol=1e-7,  # Matches all digits printed by Psi4
        )

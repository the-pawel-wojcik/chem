import math

from chem.hf.containers import ResultHF
from chem.meta.coordinates import CARTESIAN, Descartes
from chem.hf.electronic_structure import scf
from chem.hf.intermediates_builders import Intermediates, extract_intermediates
import numpy as np


def test_dipole_HF_water_sto3g(water_sto3g: ResultHF) -> None:
    mol = water_sto3g.molecule
    uhf_data: Intermediates = extract_intermediates(water_sto3g.wfn)

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

    oa = uhf_data.oa
    ob = uhf_data.ob
    my_dipole_electronic = {
        Descartes.x: float(
            sum(uhf_data.mua_x.diagonal()[oa])
            +
            sum(uhf_data.mub_x.diagonal()[ob])
        ),
        Descartes.y: float(
            sum(uhf_data.mua_y.diagonal()[oa])
            +
            sum(uhf_data.mub_y.diagonal()[ob])
        ),
        Descartes.z: float(
            sum(uhf_data.mua_z.diagonal()[oa])
            +
            sum(uhf_data.mub_z.diagonal()[ob])
        ),
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
            atol=1e-7,  # all digits printed in the psi4 output match
        )
        assert np.isclose(
            my_dipole_nuclear[direction],
            psi4_dipole_au_nuclear[direction],
            atol=1e-12,  # as good as you wish, both come from psi4
        )
        assert np.isclose(
            cccdbd_dipole_au[direction],
            psi4_dipole_au_total[direction],
            atol=1e-4,  # CCCBDB differs a little
        )


def test_dipole_HF_water_ccpVDZ():
    mol, _, wfn = scf()
    uhf_data: Intermediates = extract_intermediates(wfn)
    psi_dipole_electronic = {
        Descartes.x: 0.0,
        Descartes.y: 0.0,
        Descartes.z: 0.1588593,
    }

    psi_dipole_nulcear = {
        Descartes.x: 0.0,
        Descartes.y: 0.0,
        Descartes.z: -0.9631188,
    }
    #
    # psi_dipole_total = {
    #     Descartes.x: 0.0000000,
    #     Descartes.y: 0.0000000,
    #     Descartes.z: -0.8042596,
    # }

    oa = uhf_data.oa
    ob = uhf_data.ob
    my_dipole_electronic = {
        Descartes.x: float(
            sum(uhf_data.mua_x.diagonal()[oa])
            +
            sum(uhf_data.mub_x.diagonal()[ob])
        ),
        Descartes.y: float(
            sum(uhf_data.mua_y.diagonal()[oa])
            +
            sum(uhf_data.mub_y.diagonal()[ob])
        ),
        Descartes.z: float(
            sum(uhf_data.mua_z.diagonal()[oa])
            +
            sum(uhf_data.mub_z.diagonal()[ob])
        ),
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
            abs_tol=1e-6,
        )

        assert math.isclose(
            my_dipole_nuclear[coord],
            psi_dipole_nulcear[coord],
            abs_tol=1e-6,
        )

import math
from chem.meta.coordinates import CARTESIAN, Descartes
from chem.hf.electronic_structure import scf
from chem.hf.intermediates_builders import Intermediates, extract_intermediates


def test_dipole_HF_water_ccpVDZ():
    mol, _, wfn = scf()
    uhf_data: Intermediates = extract_intermediates(wfn)
    psi_dipole_electronic = {
        Descartes.x: -0.0000000,
        Descartes.y: 0.0000000,
        Descartes.z: 0.1588593,
    }

    psi_dipole_nulcear = {
        Descartes.x: 0.0000000,
        Descartes.y: 0.0000000,
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

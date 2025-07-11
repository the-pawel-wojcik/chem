from chem.hf.ghf_data import wfn_to_GHF_Data, GHF_Data
from chem.hf.electronic_structure import ResultHF
from chem.meta.coordinates import Descartes
from numpy import einsum
import numpy as np


def get_hf_energy(ghf_data: GHF_Data) -> float:
    f = ghf_data.f
    g = ghf_data.g
    o = ghf_data.o
    energy = 1.00 * einsum('ii', f[o, o])
    energy += -0.50 * einsum('jiji', g[o, o, o, o])
    return float(energy)


def test_constructor(water_sto3g: ResultHF):
    wfn_to_GHF_Data(water_sto3g.wfn)


def test_energies(water_sto3g: ResultHF):
    ghf_data = wfn_to_GHF_Data(water_sto3g.wfn)
    assert ghf_data.f.shape == (14, 14)
    fock_diagonal = np.array([-20.25157699, -20.25157699, -1.25754837,
    -1.25754837, -0.59385451, -0.59385451, -0.45972972, -0.45972972,
    -0.39261692, -0.39261692, 0.58179264, 0.58179264, 0.69267285, 0.69267285])
    assert np.allclose(fock_diagonal, ghf_data.f.diagonal(), atol=1e-7)
    energy = get_hf_energy(ghf_data)
    assert np.isclose(-83.87226577852897, energy, atol=1e-7)
    nre = water_sto3g.molecule.nuclear_repulsion_energy()
    assert np.isclose(nre, 8.906479, atol=1e-6)  # value from CCCBDB
    assert np.isclose(energy + nre, -74.965901, atol=1e-6)  # value from CCCBDB


def test_dipoles_shapes(water_sto3g: ResultHF):
    ghf_data = wfn_to_GHF_Data(water_sto3g.wfn)
    for direction in Descartes:
        mu_component =  ghf_data.mu[direction]
        assert mu_component.shape == (14, 14)


def test_dipole_values(water_sto3g: ResultHF) -> None:
    ghf_data = wfn_to_GHF_Data(water_sto3g.wfn)
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

    mol = water_sto3g.molecule
    geo = water_sto3g.molecule.geometry().np
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
        )
        for direction in Descartes
    }

    for direction in Descartes:
        assert np.isclose(
            my_dipole_total[direction],
            cccdbd_dipole_au[direction],
            atol=1e-4,  # CCCBDB does not match too well
        )
        assert np.isclose(
            my_dipole_electronic[direction],
            psi4_dipole_au_electronic[direction],
            atol=1e-7,  # all digits printed by psi4
        )
        assert np.isclose(
            my_dipole_nuclear[direction],
            psi4_dipole_au_nuclear[direction],
            atol=1e-12,  # as good as you wish, both come from psi4
        )
        assert np.isclose(
            cccdbd_dipole_au[direction],
            psi4_dipole_au_total[direction],
            atol=1e-4,  # CCCBDB does not match well
        )

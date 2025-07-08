from dataclasses import dataclass

from chem.meta.coordinates import Descartes
import numpy as np
from numpy.typing import NDArray
from psi4.core import Wavefunction, Matrix, MintsHelper


@dataclass
class GHF_Data:
    mu: dict[Descartes, NDArray]
    identity_singles: NDArray
    nmo: int
    no: int
    nv: int
    v: slice
    o: slice
    f: NDArray
    g: NDArray


def _tei_to_spinorbitals(tei: NDArray, order: NDArray) -> NDArray:
    """ Take the electron repulsion integrals `tei` aka two-electron integrals
    in the chemists notation:
        (s,p|q,r) = ∫∫ φs(r1) φp(r1) (1/r12) φq(r2) φr(r2) dr1 dr2                                        
    and double the size of this matrix by taking it from the basis of orbitals
    to spinorbitals. 

    Assumes that the orbitals come from RHF calculations so the up and down
    orbitals are exactly the same.

    The `tei` is expressed in RHF spatial MOs. The `order` matrix reorderes the
    spin-orbitals into the energy order after the spin-orbitals were created by
    stacking the up MOs first and down MOs second.
    """
    assert len(tei.shape) == 4
    nmo = 2 * tei.shape[0]
    for dim in tei.shape:
        assert nmo//2 == dim
    assert len(order) == nmo

    tei_ghf = np.zeros(shape=(nmo, nmo, nmo, nmo))
    up = slice(0, nmo//2, None)
    down = slice(nmo//2, nmo, None)

    # there are 16 spin-combinations, yet only two of them are non-zero
    tei_ghf[up, up, up, up] = tei
    # tei_ghf[up, up, up, down] = 0
    # tei_ghf[up, up, down, up] = 0
    tei_ghf[up, up, down, down] = tei
    # tei_ghf[up, down, up, up] = 0
    # tei_ghf[up, down, up, down] = 0
    # tei_ghf[up, down, down, up] = 0
    # tei_ghf[up, down, down, down] = 0
    # tei_ghf[down, up, up, up] = 0
    # tei_ghf[down, up, up, down] = 0
    # tei_ghf[down, up, down, up] = 0
    # tei_ghf[down, up, down, down] = 0
    tei_ghf[down, down, up, up] = tei
    # tei_ghf[down, down, up, down] = 0
    # tei_ghf[down, down, down, up] = 0
    tei_ghf[down, down, down, down] = tei

    # reorder from (all up, all down) order to the (increasing energy) order
    # try in 1D, then extend to 2D and you will see
    tei_ghf = np.asarray(
        [
            [
                [ 
                    row[order] for row in wall[order] 
                ] for wall in cube[order]
            ] for cube in tei_ghf[order]
        ]
    )
    return tei_ghf


def wfn_to_GHF_Data(wfn: Wavefunction) -> GHF_Data:
    """ wfn is the result of RHF calculations. Here the spin up and spin down
    orbitals will be combined to form the spin-orbitals for GHF. """
    nmo = 2 * wfn.nmo()  # number of molecular orbitals
    no = 2 * wfn.nalpha()  # number of occupied molecular orbitals
    nv = nmo - no  # number of virtual molecular orbitals
    occ = slice(None, no)
    vir = slice(no, None)

    # orbital energies
    orbital_energies = np.concatenate((
        wfn.epsilon_a().to_array(),
        wfn.epsilon_b().to_array(),
    ))
    order = np.argsort(orbital_energies)
    # orbitals = np.hstack((mos_up, mos_up))[:, order]  # not needed
    fock = np.diag(orbital_energies[order])

    # molecular orbitals up and down would be the same for RHF, so I use only
    # one of them
    mos_up: Matrix = wfn.Ca()

    # use Psi4's MintsHelper to generate ERIs and dipole integrals
    mints = MintsHelper(wfn.basisset())

    # build the electron integrals in the chemists' notation
    tei = np.asarray(mints.mo_eri(mos_up, mos_up, mos_up, mos_up))
    tei_ghf = _tei_to_spinorbitals(tei, order)
    tei_ghf = tei_ghf.transpose(0, 2, 1, 3) - tei_ghf.transpose(0, 2, 3, 1)
    # antisymmetrized integrals in physicists' notation
    # np.set_printoptions(precision=3, suppress=True)
    # print(f'{orbitals.shape=}')
    # for column in range(14):
    #     print(f'{orbitals[:, column]}')

    identity_singles = np.eye(nmo)

    # dipole integrals
    mu = mints.ao_dipole()

    mu_AO = dict()
    mu_AO[Descartes.x] = np.asarray(mu[0])
    mu_AO[Descartes.y] = np.asarray(mu[1])
    mu_AO[Descartes.z] = np.asarray(mu[2])

    # transform the dipole integrals to the MO basis
    mos_up = np.asarray(mos_up)

    mu_MO = {
        coordinate:
        mos_up.T @ component @ mos_up 
        for coordinate, component in mu_AO.items()
    }

    mu_ghf = {
        coordinate:
        np.block([
            [component, component * 0.0],
            [component * 0.0, component],
        ])
        for coordinate, component in mu_MO.items()
    }

    # reorder
    mu_ghf = {
        coordinate: np.asarray([
            row[order] for row in component[order]
        ])
        for coordinate, component in mu_ghf.items()
    }

    return GHF_Data(
        mu=mu_ghf,
        identity_singles=identity_singles,
        f=fock,
        o=occ,
        v=vir,
        nmo=nmo,
        no=no,
        nv=nv,
        g=tei_ghf,
    )

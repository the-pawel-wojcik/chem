import itertools

from chem.ccsd.ghf_ccsd import GHF_CCSD, GHF_CCSD_Config
from chem.hf.electronic_structure import ResultHF
from chem.hf.ghf_data import GHF_Data, wfn_to_GHF_Data
import numpy as np
from numpy.typing import NDArray
from psi4_ccsd_sto3_water_data import PSI4_tIjAb, PSI4_tijab
import pytest


@pytest.fixture(scope='session')
def ghf_data(water_sto3g: ResultHF) -> GHF_Data:
    return wfn_to_GHF_Data(water_sto3g.wfn)


def test_ccsd_print_leading_t_amplitudes(
    ghf_data: GHF_Data,
) -> None:
    ccsd = GHF_CCSD(
        ghf_data,
        GHF_CCSD_Config(
            energy_convergence=1e-10,
            residuals_convergence=1e-10,
            t_amp_print_threshold=1e-2,
        )
    )
    ccsd.solve_cc_equations()
    ccsd.print_leading_t_amplitudes()

    top_t1 = ccsd._find_leading_t1_amplitudes()
    assert len(top_t1) == 2

    assert top_t1[0]['v'] == 0
    assert top_t1[0]['o'] == 6
    assert np.isclose(top_t1[0]['amp'], 0.014, atol=1e-3)

    assert top_t1[1]['v'] == 1
    assert top_t1[1]['o'] == 7
    assert np.isclose(top_t1[1]['amp'], 0.014, atol=1e-3)

    top_t2 = ccsd._find_leading_t2_amplitudes()
    assert len(top_t2) == 88

    assert top_t2[62]['vl'] == 2
    assert top_t2[62]['vr'] == 3
    assert top_t2[62]['ol'] == 4
    assert top_t2[62]['or'] == 5
    assert np.isclose(top_t2[62]['amp'], -0.085, atol=1e-3)

    assert top_t2[63]['vl'] == 2
    assert top_t2[63]['vr'] == 3
    assert top_t2[63]['ol'] == 5
    assert top_t2[63]['or'] == 4
    assert np.isclose(top_t2[63]['amp'], 0.085, atol=1e-3)

    assert top_t2[82]['vl'] == 3
    assert top_t2[82]['vr'] == 2
    assert top_t2[82]['ol'] == 4
    assert top_t2[82]['or'] == 5
    assert np.isclose(top_t2[82]['amp'], 0.085, atol=1e-3)

    assert top_t2[83]['vl'] == 3
    assert top_t2[83]['vr'] == 2
    assert top_t2[83]['ol'] == 5
    assert top_t2[83]['or'] == 4
    assert np.isclose(top_t2[83]['amp'], -0.085, atol=1e-3)


def print_ghf_doubles(doubles: NDArray, ghf_data: GHF_Data) -> None:
    assert len(doubles.shape) == 4
    no = ghf_data.no
    nv = ghf_data.nv
    assert doubles.shape == (nv, nv, no, no)

    pad = ' '
    fmt = ' z6.3f'
    print(r'[')  # ]
    for a, cube in enumerate(doubles):
        print(f'{pad}[ {a=}')  # ]
        for b, wall in enumerate(cube):
            print(f'{pad*2}[ {b=}')  # ]
            print(f'{pad*3}[  j=', end='')  # ]
            for j, _ in enumerate(wall[0]):
                print(f'{j:^6d}', end='')
            print(']')
            for i, row in enumerate(wall):
                print(f'{pad*3}[ {i=}', end='')  # ]
                for value in row:
                    print(f'{value:{fmt}}', end='')
                print('],')
            print(f'{pad*2}],')
        print(f'{pad}],')
    print(']')


def turn_psi4_IjAb_rhf_to_ghf(
    t2IjAb: NDArray,
    ghf_data: GHF_Data,
) -> NDArray:
    nv = ghf_data.nv // 2  # RHF is only half the dimension of GHF
    no = ghf_data.no // 2
    assert t2IjAb.shape == (no, no, nv, nv)
    ghf = np.zeros(shape=[2*nv, 2*nv, 2*no, 2*no])
    for a, b, i, j in itertools.product(
        range(0, nv), range(0, nv), range(0, no), range(0, no)
    ):
        value = t2IjAb[i, j, a, b]
        # each 2-electron matrix element in RHF
        # corresponds to four spin cases in GHF

        # the commented out terms are zero in psi4
        # ghf[2*a, 2*b, 2*i, 2*j] = value
        ghf[2*a+1, 2*b, 2*i+1, 2*j] = value
        ghf[2*a, 2*b+1, 2*i, 2*j+1] = value
        # ghf[2*a+1, 2*b+1, 2*i+1, 2*j+1] = value

        # new terms added
        ghf[2*a+1, 2*b, 2*i, 2*j+1] = -value
        ghf[2*a, 2*b+1, 2*i+1, 2*j] = -value
    return ghf


def turn_psi4_ijab_rhf_to_ghf(
    t2ijab: NDArray,
    ghf_data: GHF_Data,
) -> NDArray:
    nv = ghf_data.nv // 2  # RHF is only half the dimension of GHF
    no = ghf_data.no // 2
    assert t2ijab.shape == (no, no, nv, nv)
    ghf = np.zeros(shape=[2*nv, 2*nv, 2*no, 2*no])
    for a, b, i, j in itertools.product(
        range(0, nv), range(0, nv), range(0, no), range(0, no)
    ):
        lowercase = t2ijab[i, j, a, b]
        ghf[2*a, 2*b, 2*i, 2*j] = lowercase
        ghf[2*a+1, 2*b+1, 2*i+1, 2*j+1] = lowercase
    return ghf


@pytest.mark.skip
def test_t2_amplitudes_vs_Psi4(
    ghf_data: GHF_Data,
) -> None:
    """ This one is unfinished. Learn how to fully convert the t2s from Psi4's
    format to complete this example. """
    ccsd = GHF_CCSD(
        ghf_data,
        GHF_CCSD_Config(
            verbose=0,
            use_diis=False,
            max_iterations=100,
            energy_convergence=1e-9,
            residuals_convergence=1e-9,
            shift_1e=0.0,
            shift_2e=0.0,
            t_amp_print_threshold=1e-2,
        )
    )
    ccsd.solve_cc_equations()
    t2_paweł = ccsd.data.t2
    t2_psi_IjAb = turn_psi4_IjAb_rhf_to_ghf(
        np.array(PSI4_tIjAb),
        ghf_data,
    )
    t2_psi_ijab = turn_psi4_ijab_rhf_to_ghf(
        np.array(PSI4_tijab),
        ghf_data,
    )
    print()
    print("Paweł's t2")
    print_ghf_doubles(t2_paweł, ghf_data)
    print("Psi's t2_IjAb converted to GHF")
    print_ghf_doubles(t2_psi_IjAb, ghf_data)
    print("Psi's t2_ijab converted to GHF")
    print_ghf_doubles(t2_psi_ijab, ghf_data)
    print("Psi's t2 converted to GHF")
    print_ghf_doubles(t2_psi_IjAb + t2_psi_ijab, ghf_data)
    # assert t2_paweł.shape == t2_psi.shape
    # for cube_pw, cube_psi in zip(t2_paweł, t2_psi):
    #     assert np.allclose(cube_pw, cube_psi, atol=1e-8)


def test_ccsd_print_leading_lambda_amplitudes(
    ghf_data: GHF_Data,
) -> None:
    ccsd = GHF_CCSD(
        ghf_data,
        GHF_CCSD_Config(
            energy_convergence=1e-10,
            residuals_convergence=1e-10,
            t_amp_print_threshold=1e-2,
        )
    )
    ccsd.solve_lambda_equations()
    ccsd.print_leading_lambda_amplitudes()

    top_l1 = ccsd._find_leading_l1_amplitudes()
    assert len(top_l1) == 2

    assert top_l1[0]['o'] == 6
    assert top_l1[0]['v'] == 0
    assert np.isclose(top_l1[0]['amp'], 0.01224, atol=1e-5)

    assert top_l1[1]['o'] == 7
    assert top_l1[1]['v'] == 1
    assert np.isclose(top_l1[1]['amp'], 0.01224, atol=1e-5)

    top_l2 = ccsd._find_leading_l2_amplitudes()
    assert len(top_l2) == 80

    assert top_l2[26]['vl'] == 2
    assert top_l2[26]['vr'] == 3
    assert top_l2[26]['ol'] == 4
    assert top_l2[26]['or'] == 5
    assert np.isclose(top_l2[26]['amp'], -0.08330, atol=1e-5)

    assert top_l2[27]['vl'] == 3
    assert top_l2[27]['vr'] == 2
    assert top_l2[27]['ol'] == 4
    assert top_l2[27]['or'] == 5
    assert np.isclose(top_l2[27]['amp'], 0.08330, atol=1e-5)

    assert top_l2[40]['vl'] == 2
    assert top_l2[40]['vr'] == 3
    assert top_l2[40]['ol'] == 5
    assert top_l2[40]['or'] == 4
    assert np.isclose(top_l2[40]['amp'], 0.08330, atol=1e-5)

    assert top_l2[41]['vl'] == 3
    assert top_l2[41]['vr'] == 2
    assert top_l2[41]['ol'] == 5
    assert top_l2[41]['or'] == 4
    assert np.isclose(top_l2[41]['amp'], -0.08330, atol=1e-5)

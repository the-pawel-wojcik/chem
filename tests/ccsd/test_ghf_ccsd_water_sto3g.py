import pytest
from chem.ccsd.ghf_ccsd import GHF_CCSD, GHF_CCSD_Config
from chem.hf.electronic_structure import ResultHF
from chem.hf.ghf_data import GHF_Data, wfn_to_GHF_Data
from chem.meta.coordinates import Descartes
import numpy as np


PSI4_NRE_H2O_STO3G = 8.9064754830195323
PSI4_CCSD_ENERGY_H2O_STO3G = -75.020284329414
PSI4_CCSD_DIPOLE_ELECTRONIC = {
    Descartes.x: 0.0,
    Descartes.y: 0.0,
    Descartes.z: 0.4376790,
}
PSI4_CCSD_DIPOLE_NUCLEAR = {
    Descartes.x: 0.0,
    Descartes.y: 0.0,
    Descartes.z: -1.0583371,
}
PSI4_CCSD_DIPOLE_TOTAL = {
    Descartes.x: 0.0,
    Descartes.y: 0.0,
    Descartes.z: -0.6206582,
}


@pytest.fixture(scope='session')
def nuclear_repulsion_energy(water_sto3g: ResultHF) -> float:
    nuclear_repulsion_energy = water_sto3g.molecule.nuclear_repulsion_energy()
    return nuclear_repulsion_energy


@pytest.mark.skip
def test_nuclear_repulsion_energy(nuclear_repulsion_energy: float) -> None:
    assert np.isclose(nuclear_repulsion_energy, PSI4_NRE_H2O_STO3G, 1e-10)


@pytest.fixture(scope='session')
def ghf_data(water_sto3g: ResultHF) -> GHF_Data:
    return wfn_to_GHF_Data(water_sto3g.wfn)


@pytest.mark.skip
@pytest.mark.parametrize(
    argnames='ghf_ccsd_config',
    argvalues= [
        pytest.param(GHF_CCSD_Config(verbose=1, use_diis=False), id='no_diis'),
        # pytest.param(GHF_CCSD_Config(verbose=1), id='default'),
    ]
)
def test_ccsd_energy(
    ghf_data: GHF_Data,
    nuclear_repulsion_energy: float,
    ghf_ccsd_config: GHF_CCSD_Config,
) -> None:
    ccsd = GHF_CCSD(ghf_data, ghf_ccsd_config)
    ccsd.solve_cc_equations()
    ghf_ccsd_energy = ccsd.get_energy()
    assert np.isclose(
        ghf_ccsd_energy,
        PSI4_CCSD_ENERGY_H2O_STO3G - PSI4_NRE_H2O_STO3G,
        atol=1e-10
    )
    ghf_ccsd_total_energy = ghf_ccsd_energy + nuclear_repulsion_energy
    assert np.isclose(
        ghf_ccsd_total_energy,
        PSI4_CCSD_ENERGY_H2O_STO3G,
        atol=1e-10
    )


# def test_ccsd_diis_energy(
#     ghf_data: GHF_Data,
#     nuclear_repulsion_energy: float
# ):
#     ccsd = GHF_CCSD(ghf_data, GHF_CCSD_Config(verbose=1))
#     ccsd.solve_cc_equations()
#     ghf_ccsd_energy = ccsd.get_energy()
#     ghf_ccsd_total_energy = ghf_ccsd_energy + nuclear_repulsion_energy
#     assert np.isclose(ghf_ccsd_energy, -83.9266502349831, atol=1e-5)
#     assert np.isclose(ghf_ccsd_total_energy, -75.02028564818042, atol=1e-5)


@pytest.mark.skip
def test_lambda_solver(ghf_data: GHF_Data) -> None:
    ccsd = GHF_CCSD(ghf_data, config=GHF_CCSD_Config(verbose=True))
    ccsd.solve_lambda_equations()


def test_dipole_moment(ghf_data: GHF_Data) -> None:
    ccsd = GHF_CCSD(
        ghf_data=ghf_data,
        config=GHF_CCSD_Config(
            verbose=0,
            energy_convergence=1e-12,
            residuals_convergence=1e-12,
        ),
    )
    electronic_edm = ccsd._get_electronic_electric_dipole_moment()
    for key, val in electronic_edm.items():
        assert np.isclose(PSI4_CCSD_DIPOLE_ELECTRONIC[key], val, atol=1e-2)
    # TODO: This is not a consisent result. Figure out what's wrong

    eEDM_via_opdm = ccsd._get_electronic_electric_dipole_moment_via_opdm()
    for key, val in eEDM_via_opdm.items():
        assert np.isclose(PSI4_CCSD_DIPOLE_ELECTRONIC[key], val, atol=1e-2)

    # The two different ways of fiding the eEDM coincide
    for dir in Descartes:
        assert np.isclose(eEDM_via_opdm[dir], electronic_edm[dir], atol=1e-12)

    print()
    print('Electronic part of the electric dipole moment')
    print('       Psi      OPDM      <mu>')
    fmt=' 8.6f'
    for dir in Descartes:
        print(
            f'{dir}:'
            f' {PSI4_CCSD_DIPOLE_ELECTRONIC[dir]:{fmt}}'
            f' {eEDM_via_opdm[dir]:{fmt}}'
            f' {electronic_edm[dir]:{fmt}}'
        )
    print('Total electric dipole moment')
    print('       Psi      OPDM      <mu>')
    for dir in Descartes:
        psi = PSI4_CCSD_DIPOLE_ELECTRONIC[dir] + PSI4_CCSD_DIPOLE_NUCLEAR[dir]
        via_opdm = eEDM_via_opdm[dir] + PSI4_CCSD_DIPOLE_NUCLEAR[dir]
        via_exp = electronic_edm[dir] + PSI4_CCSD_DIPOLE_NUCLEAR[dir]

        print(
            f'{dir}:'
            f' {psi:{fmt}}'
            f' {via_opdm:{fmt}}'
            f' {via_exp:{fmt}}'
        )

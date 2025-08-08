from chem.meta.polarizability import Polarizability


def test_polarizability_str() -> None:
    pol = Polarizability.from_builder(builder=lambda left, right: -0.0)
    str_pol = str(pol)
    str_pol_goal = """xx:     0.000000
xy:     0.000000
xz:     0.000000
yx:     0.000000
yy:     0.000000
yz:     0.000000
zx:     0.000000
zy:     0.000000
zz:     0.000000"""
    assert str_pol == str_pol_goal

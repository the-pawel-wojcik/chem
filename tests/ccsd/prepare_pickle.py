import pickle

from chem.hf.electronic_structure import hf
from chem.hf.intermediates_builders import extract_intermediates
from chem.ccsd.uhf_ccsd import UHF_CCSD


def solve_and_save_ccsd():
    """ Geometry from CCCBDB: HF/STO-3G """
    geometry = """
    0 1
    O1	0.0000   0.0000   0.1272
    H2	0.0000   0.7581  -0.5086
    H3	0.0000  -0.7581  -0.5086
    symmetry c1
    """
    hf_result = hf(geometry=geometry, basis='sto-3g')
    intermediates = extract_intermediates(hf_result.wfn)
    ccsd = UHF_CCSD(intermediates)
    ccsd.verbose = 1
    ccsd.solve_cc_equations()
    with open('water_CCSD_STO-3G@HF_STO-3G.pkl','wb') as file:
        pickle.dump(ccsd, file)


if __name__ == "__main__":
    solve_and_save_ccsd()

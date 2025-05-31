import pickle

from chem.ccsd.uhf_ccsd import UHF_CCSD


def test_uhf_ccsd_lambda() -> None:
    with open('uhf_ccsd.pkl','rb') as bak_file:
       ccsd: UHF_CCSD = pickle.load(bak_file)

    ccsd.verbose = 1
    ccsd.solve_lambda_equations()

if __name__ == "__main__":
    test_uhf_ccsd_lambda()

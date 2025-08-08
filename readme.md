# Computational Chemistry
Collection of electronic structure programs. UHF-CCSD and GHF-CCSD are tested
and work. UHF-CCS is available too, but it was not tested.

## Dependencies
This package depends on [psi4](https://psicode.org/), this dependency is not
listed in the `pyproject.toml` as `psi4` is unavailable from pypi.

## Installation
1. First, make sure that `psi4` is available in your environment, e.g., calling
```bash
python -c 'import psi4'
```
should not report any errors. If you see
```
$ python -c 'import psi4'
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import psi4
ModuleNotFoundError: No module named 'psi4'
```
That means that `psi4` is not available in your environment and this package
would not work.

2. I recommend installing this package with `pip` (even for a `conda`
   distribution)
```bash
git clone git@github.com:the-pawel-wojcik/chem.git
cd chem
python -m pip install -e .
```

## Example use
The directory `examples` contains a complete GHF-CCSD calculation for water in
STO-3G basis set. This example includes all possible options listings and is
very verbose. Run it with
```bash
cd examples
python ghf_ccsd_water_sto3g.py
```

## Tests
The `tests` directory contains many tests of the package. The tests are using
`pytest`. To run the tests do
```bash
python -m pip install pytest
cd tests
pytest -v
```

from dataclasses import dataclass
from psi4.core import Molecule, Wavefunction


@dataclass
class ResultHF:
    molecule: Molecule
    hf_energy: float
    wfn: Wavefunction

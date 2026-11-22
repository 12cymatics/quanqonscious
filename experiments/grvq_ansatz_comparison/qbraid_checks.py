"""Compare returned qBraid distributions with the same optimized objectives.

This reader never submits a job. Provider responses must already exist in
qbraid_results.json. Exact expected states and circuit files remain separate
from the provider's returned distributions.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path


def correlation(probabilities, bits):
    mask = sum(1 << bit for bit in bits)
    return sum((-1 if (int(key, 2) & mask).bit_count() % 2 else 1)*value
               for key, value in probabilities.items())


def compare_energies(directory):
    directory = Path(directory)
    manifest = json.loads((directory/"qbraid_manifest.json").read_text())
    path = directory/"qbraid_results.json"
    report = json.loads(path.read_text())
    result = []
    for item in manifest:
        name, n = item["model"], item["nqubits"]
        distributions = {circuit["basis"]: circuit["simulation"]["probabilities"]
                         for circuit in report["circuits"] if circuit["model"] == name}
        pz, px = distributions["Z"*n], distributions["X"*n]
        bonds = tuple((i, (i+1) % n) for i in range(n))
        if name == "Q4_harmonic":
            rows = json.loads((directory/(name+"_matrices.json")).read_text())["A_rows"]
            potentials = [next(Fraction(v)-n for j, v in row if i == j)
                          for i, row in enumerate(rows)]
            energy = n-sum(correlation(px, (i,)) for i in range(n))
            energy += sum(float(potentials[int(key, 2)])*value for key, value in pz.items())
        elif name.startswith("TFIM") and name.endswith("_h1"):
            energy = -sum(correlation(pz, bond) for bond in bonds)
            energy -= sum(correlation(px, (i,)) for i in range(n))
        elif name == "XXZ4_delta1":
            py = distributions["Y"*n]
            energy = sum(correlation(distribution, bond)
                         for distribution in (px, py, pz) for bond in bonds)
        else:
            raise ValueError("No complete measured-observable decomposition for "+name)
        expected = float(Fraction(item["energy_exact"]))
        error = abs(energy-expected)
        if error > 1e-10:
            raise ArithmeticError(f"qBraid observable mismatch for {name}: {error}")
        result.append({"model": name, "family": item["family"],
                       "measured_basis_energy": energy, "local_exact_energy_projected": expected,
                       "absolute_energy_error": error,
                       "measurement": "complete required Pauli groups from ideal distributions; no finite-shot sampling"})
    report["energy_comparisons"] = result
    report["max_absolute_energy_error"] = max(item["absolute_energy_error"] for item in result)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare_energies(args.directory), indent=2))

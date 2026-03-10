sutraws_tgcr_lean_full

## Hybrid Sutra Platform

Run the 29 Vedic sutras in serial, concurrent, and parallel modes using the
hybrid quantum-classical simulator entrypoint:

```bash
python hybrid_sutra_platform.py 1.618 --mode hybrid --output runs/hsqcp_report.json
```

Launch the one-click web interface that opens the full hybrid sutra engine,
complete with system and industry alignment tags:

```bash
python web_server.py
```

To use a different port, set `PORT`:

```bash
PORT=3001 python web_server.py
```

The web console also lists the full 29-sutra inventory sourced directly from
the repository so you can verify the complete algorithm set used for serial,
concurrent, and parallel execution.

Product and revenue details are documented in
`docs/hybrid_sutra_platform.md`.


## Exact Rational Runtime Usage

Run the hybrid 29-sutra engine with exact rational input:

```bash
python hybrid_sutra_platform.py 1618/1000 --mode hybrid --output runs/hsqcp_report.json
```

Run deterministic benchmark seeds with signature-vault persistence:

```bash
python hybrid_sutra_platform.py 1 --mode hybrid --benchmark-seeds 1,1618/1000,2,31415926535/10000000000 --vault-dir runs/vault --run-label baseline
```

Audit persisted signature-vault integrity:

```bash
python hybrid_sutra_platform.py 1 --audit-vault runs/vault
```

Static contract validation:

```bash
python scripts/validate_hsqcp_contracts.py
```


Export benchmark matrix and reproducibility manifest in one run:

```bash
python hybrid_sutra_platform.py 1 --mode hybrid --benchmark-seeds 1,1618/1000,2,31415926535/10000000000 --benchmark-matrix-output runs/benchmark_matrix.json --manifest-output runs/repro_manifest.json
```

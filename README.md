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

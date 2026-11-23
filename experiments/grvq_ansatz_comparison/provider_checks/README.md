# qBraid verification of the completed feedback sweep

This verification reads the existing **36 trajectories, 72 frames and 288 four-qubit circuit exports**. It does not rerun or modify the field evolution, quantum optimizer or exact feedback trajectory. The numerical dataset is pinned to commit `f2f3bc798294e1fe56b3dacbecb0f494a6d6d8bd`; its `latest_hybrid_results/results.json` SHA-256 is:

```text
2c1989a6ebc91c735161f530d9566434223bb19257ff78c9f365a19afb35c2d1
```

## Execution accounting

The complete verification accounts for **146 successful simulator responses covering all 288 logical exports**:

| Execution form | Actual programs | Logical four-qubit circuits |
| --- | ---: | ---: |
| Individual four-qubit programs | 75 | 75 |
| Twelve-qubit programs containing three independent circuits | 71 | 213 |
| Total | 146 | 288 |

Each frame has its ZZZZ, XXXX, YYYY and XYZX exports checked: 72 circuits per basis. Failed rate-limited attempts remain in `latest_hybrid_results/provider_attempt_history/` and do not count as successful executions. The requested cooldown was respected before the remaining complete circuits were grouped into twelve-qubit programs.

## Exact batch mapping and probability checks

`qbraid_manifest.json` identifies every original circuit, source QASM hash, expected-probability hash, actual execution and batch slot. Slot `s=0,1,2` maps the original qubit `q` to `q+4s`, giving disjoint registers `[0,1,2,3]`, `[4,5,6,7]` and `[8,9,10,11]`. Every gate and angle string stays in its original order; the verifier independently reconstructs the complete submitted program. No gate couples different registers. The final measurement uses the twelve corresponding classical bits.

Provider bitstrings are ordered `q11..q0`; slot zero occupies the rightmost four bits. For each batch, all **4,096 outcomes** are compared with the exact product of the three saved reference distributions. Across 71 batches this checks **290,816 joint probabilities**. Every returned JSON probability is interpreted as its exact binary64 rational using `Fraction.from_float`.

Each four-qubit marginal is the exact sum of the 256 joint probabilities having the required local outcome. These derived marginals are distinguished from the 75 direct four-qubit responses. No fabricated individual provider response is created for a batch member. Raw batch responses, complete submitted programs and exact signed joint-error receipts are retained in `provider_batches/`, `provider_batch_programs/` and `provider_joint_checks/` under `latest_hybrid_results/`.

Every joint and logical probability discrepancy must be at most the exact fraction `1/10^12`; returned normalization sums must satisfy the same tolerance. The verifier never renormalizes or clamps the probabilities.

## Hamiltonian and feedback checks

For every Z-basis frame, all four Z moments and three neighboring ZZ moments are reconstructed. Current-Hamiltonian energies use the retained rational lambda enclosure certificates, with an independent contraction over all sixteen basis states.

Writing `Lq` for the maximum absolute endpoint of lambda's enclosure, the certified energy-error bound is no larger than

```text
sum_q Lq*abs(delta_Zq) + 1/2*sum_q abs(delta_ZZq)
<= 16/10^12 * (sum_q Lq + 3/2).
```

The report labels this value as an **energy discrepancy upper bound**, not a subtraction of rounded energy displays. Computed feedback `-Zq/1000` and applied feedback are checked separately; disabled-feedback controls apply exact zero. The actual probability L1 discrepancy also supplies the tighter feedback-error bound `L1/1000`.

## Offline verification

From the extracted repository root:

```bash
python -m experiments.grvq_ansatz_comparison.provider_checks.verify_qbraid \
  --results experiments/grvq_ansatz_comparison/latest_hybrid_results
```

This command submits no provider jobs. It verifies the retained responses and hashes, writes the actual verification summary and exact-error receipts, and exits successfully only for the complete declared coverage. Final publication requires `qbraid_verification.json` to report `status: complete`. The summary and `qbraid_results.json` preserve the distinction between logical circuits, actual individual/batch executions and failed attempts. These are ideal simulator checks; they are not QPU measurements.

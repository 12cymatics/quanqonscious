"""Visualize the palindromic dual-lattice alloy.

This utility computes the integer evaluations ``S_k(1)`` for ``k=1..16`` and
forms the palindromic alloy described in the README.  It plots these values and
saves the figure as ``palindromic_alloy.png``.
"""

import argparse
import math
from fractions import Fraction

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it with 'pip install matplotlib'"
    ) from exc

# Integer evaluations of S_k(1)
def compute_S_k_at_1(k: int) -> int:
    d_k = (k % 4) + 2
    return sum(((-1) ** (i * k)) * math.comb(k + d_k, i) for i in range(d_k + 1))

# Generate S_k(1) for k = 1..16
S_values = [compute_S_k_at_1(k) for k in range(1, 17)]

# Lucas numbers for weights
lucas = [2, 1, 3, 4, 7, 11, 18, 29]
sum_lucas = sum(lucas)
weights = [Fraction(L, sum_lucas) for L in lucas]

# Compute palindromic alloy sum
pal_fraction = sum(
    weights[i] * (S_values[i] + S_values[15 - i])
    for i in range(8)
)
pal_sum = float(pal_fraction)

print(
    f"Palindromic alloy Λ_pal = {pal_fraction} ".lstrip()
    + f"(≈ {pal_sum:.2f})"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--no-show",
    action="store_true",
    help="Do not display the plot window",
)
parser.add_argument(
    "--output",
    default="palindromic_alloy.png",
    help="Where to save the plot",
)

args = parser.parse_args()

# Plot S_k(1) values and mirrored pairs
fig, ax = plt.subplots(figsize=(10, 6))
indices = list(range(1, 17))
ax.bar(indices, S_values, color='skyblue')
ax.set_xlabel('k')
ax.set_ylabel('S_k(1)')
ax.set_title('Main-sutra polynomial evaluations at z=1')

# Highlight mirrored pairs with lines
for i in range(8):
    ax.plot([i + 1, 16 - i], [S_values[i], S_values[15 - i]], 'r--')

plt.tight_layout()
plt.savefig(args.output)
if not args.no_show:
    plt.show()

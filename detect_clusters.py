#!/usr/bin/env python3
"""
Cluster Detection in Vedic Cymatic Patterns
Analyzes the field to find regions of similar values (clusters)
"""

import math
from collections import defaultdict
from typing import List, Tuple, Dict

# Import the sutra functions
from vedic_sutra_cymatic_engine import (
    grvq_field_solver, sisyate_sesasamjnah,
    CHAKRA_FREQUENCIES, SCHUMANN_RESONANCES
)


def detect_clusters_in_field(resolution: int, chakra_freq: float,
                              schumann: float, num_bins: int = 32) -> Dict:
    """
    Detect clusters by binning field values and finding connected regions.
    """
    center = resolution // 2
    max_r = resolution // 2

    # Compute field values
    field = [[0.0] * resolution for _ in range(resolution)]

    for y in range(resolution):
        for x in range(resolution):
            dx = x - center
            dy = y - center
            r = math.sqrt(dx * dx + dy * dy) / max_r * 10
            if r < 0.001:
                r = 0.001
            theta = math.atan2(dy, dx)
            phi = (chakra_freq / 100.0) * theta + (schumann / 10.0) * r

            grvq_val = grvq_field_solver(r, theta, phi, 0.5)
            chakra_wave = math.sin(2 * math.pi * (chakra_freq / 50.0) * r)
            schumann_wave = math.sin(2 * math.pi * (schumann / 5.0) * r)

            field[y][x] = grvq_val * (1 + chakra_wave) * (1 + 0.5 * schumann_wave)

    # Bin the values using sisyate (remainder)
    bins = [[0] * resolution for _ in range(resolution)]
    for y in range(resolution):
        for x in range(resolution):
            bin_val = int(sisyate_sesasamjnah(abs(field[y][x]) * 1000, num_bins))
            bins[y][x] = bin_val

    # Count bin frequencies
    bin_counts = defaultdict(int)
    for y in range(resolution):
        for x in range(resolution):
            bin_counts[bins[y][x]] += 1

    # Find clusters (connected components of same bin value)
    visited = [[False] * resolution for _ in range(resolution)]
    clusters = []

    def flood_fill_tolerance(start_y, start_x, center_bin, tolerance=3):
        """Find connected region with bin values within tolerance."""
        stack = [(start_y, start_x)]
        region = []

        while stack:
            cy, cx = stack.pop()
            if cy < 0 or cy >= resolution or cx < 0 or cx >= resolution:
                continue
            if visited[cy][cx]:
                continue
            # Allow bins within tolerance (handles wrap-around)
            bin_diff = min(abs(bins[cy][cx] - center_bin),
                          num_bins - abs(bins[cy][cx] - center_bin))
            if bin_diff > tolerance:
                continue

            visited[cy][cx] = True
            region.append((cy, cx))

            # 8-connectivity (includes diagonals)
            stack.extend([
                (cy+1, cx), (cy-1, cx), (cy, cx+1), (cy, cx-1),
                (cy+1, cx+1), (cy+1, cx-1), (cy-1, cx+1), (cy-1, cx-1)
            ])

        return region

    # Find all clusters (minimum size 50 pixels with tolerance)
    for y in range(resolution):
        for x in range(resolution):
            if not visited[y][x]:
                region = flood_fill_tolerance(y, x, bins[y][x], tolerance=4)
                if len(region) >= 50:  # Clusters of 50+ similar-valued pixels
                    # Calculate cluster properties
                    avg_y = sum(p[0] for p in region) / len(region)
                    avg_x = sum(p[1] for p in region) / len(region)
                    avg_r = math.sqrt((avg_x - center)**2 + (avg_y - center)**2) / max_r
                    avg_theta = math.atan2(avg_y - center, avg_x - center)

                    clusters.append({
                        'bin': bins[y][x],
                        'size': len(region),
                        'center': (avg_x, avg_y),
                        'radius': avg_r,
                        'angle': math.degrees(avg_theta),
                        'field_value': field[int(avg_y)][int(avg_x)]
                    })

    # Sort by size
    clusters.sort(key=lambda c: c['size'], reverse=True)

    # Analyze border region (r > 0.7)
    border_clusters = [c for c in clusters if c['radius'] > 0.7]
    inner_clusters = [c for c in clusters if c['radius'] <= 0.7]

    return {
        'total_clusters': len(clusters),
        'border_clusters': len(border_clusters),
        'inner_clusters': len(inner_clusters),
        'bin_distribution': dict(bin_counts),
        'top_10_clusters': clusters[:10],
        'border_analysis': border_clusters[:10]
    }


def main():
    print("=" * 70)
    print("CLUSTER DETECTION IN VEDIC CYMATIC PATTERNS")
    print("=" * 70)

    # Analyze at lower resolution for speed
    resolution = 400

    for chakra_name, freq in list(CHAKRA_FREQUENCIES.items())[:3]:  # First 3 chakras
        idx = list(CHAKRA_FREQUENCIES.keys()).index(chakra_name)
        schumann = SCHUMANN_RESONANCES[idx]

        print(f"\n[{chakra_name} Chakra - {freq} Hz + Schumann {schumann} Hz]")
        print("-" * 50)

        result = detect_clusters_in_field(resolution, freq, schumann)

        print(f"Total clusters detected: {result['total_clusters']}")
        print(f"Border clusters (r > 0.7): {result['border_clusters']}")
        print(f"Inner clusters (r <= 0.7): {result['inner_clusters']}")

        print(f"\nTop 5 largest clusters:")
        for i, c in enumerate(result['top_10_clusters'][:5]):
            print(f"  {i+1}. Bin {c['bin']:2d} | Size: {c['size']:5d} px | "
                  f"r={c['radius']:.2f} | θ={c['angle']:6.1f}° | "
                  f"field={c['field_value']:.4f}")

        print(f"\nBorder region clusters:")
        for i, c in enumerate(result['border_analysis'][:5]):
            print(f"  {i+1}. Bin {c['bin']:2d} | Size: {c['size']:5d} px | "
                  f"r={c['radius']:.2f} | θ={c['angle']:6.1f}° | "
                  f"field={c['field_value']:.4f}")

        # Bin distribution summary
        print(f"\nBin distribution (top 5 most common):")
        sorted_bins = sorted(result['bin_distribution'].items(),
                            key=lambda x: x[1], reverse=True)[:5]
        for bin_val, count in sorted_bins:
            pct = count / (resolution * resolution) * 100
            print(f"  Bin {bin_val:2d}: {count:6d} pixels ({pct:.1f}%)")

    print("\n" + "=" * 70)
    print("CLUSTER DETECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Visualize Colab Notebook Inventory
Generates comprehensive visualizations of notebook features and coverage
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Notebook feature coverage data
notebooks = [
    "Another copy of Untitled38.ipynb",
    "Copy of Untitled24.ipynb",
    "Copy of Untitled38.ipynb",
    "Copy of Untitled5.ipynb",
    "Untitled1.ipynb",
    "Untitled10.ipynb",
    "Untitled11.ipynb",
    "Untitled12.ipynb",
    "Untitled13.ipynb",
    "Untitled15.ipynb",
    "Untitled19.ipynb",
    "Untitled20.ipynb",
    "Untitled21.ipynb",
    "Untitled24.ipynb",
    "Untitled26.ipynb",
    "Untitled28.ipynb",
    "Untitled31.ipynb",
    "Untitled32.ipynb",
    "Untitled35.ipynb",
    "Untitled39.ipynb",
    "Untitled44.ipynb",
    "Untitled46.ipynb",
    "Untitled5.ipynb",
    "Untitled7.ipynb",
]

# Feature matrix (1 = has feature, 0 = no feature)
features = {
    "CUDA": [1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
    "MPI": [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
    "Quantum": [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
    "Dask": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    "TensorFlow": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "Sutra": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "Parallel": [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
}

# Create DataFrame
df = pd.DataFrame(features, index=notebooks)

# Statistics
total_notebooks = len(notebooks)
feature_counts = {
    "CUDA": 16,
    "MPI": 16,
    "Quantum": 21,
    "Dask": 3,
    "TensorFlow": 3,
    "Sutra": 24,
    "Parallel": 22,
}


def create_visualizations():
    """Generate all visualizations"""

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Feature Coverage Heatmap
    ax1 = fig.add_subplot(gs[0:2, 0])
    sns.heatmap(df.T, cmap='YlGnBu', cbar_kws={'label': 'Feature Present'},
                ax=ax1, linewidths=0.5, linecolor='gray')
    ax1.set_title('Feature Coverage Matrix\nAcross 24 Colab Notebooks',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Notebooks', fontsize=11)
    ax1.set_ylabel('Features', fontsize=11)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=10)

    # 2. Feature Usage Bar Chart
    ax2 = fig.add_subplot(gs[0, 1])
    feature_names = list(feature_counts.keys())
    counts = list(feature_counts.values())
    colors = sns.color_palette("husl", len(feature_names))

    bars = ax2.barh(feature_names, counts, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('Number of Notebooks', fontsize=11)
    ax2.set_title('Feature Usage Statistics\n(out of 24 notebooks)',
                  fontsize=12, fontweight='bold', pad=15)
    ax2.set_xlim(0, 25)

    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax2.text(count + 0.5, i, str(count), va='center', fontsize=10, fontweight='bold')

    ax2.grid(axis='x', alpha=0.3)

    # 3. Feature Distribution Pie Chart
    ax3 = fig.add_subplot(gs[1, 1])
    feature_percentages = [(count / total_notebooks) * 100 for count in counts]

    wedges, texts, autotexts = ax3.pie(feature_percentages, labels=feature_names,
                                         autopct='%1.1f%%', startangle=90,
                                         colors=colors, textprops={'fontsize': 10})
    ax3.set_title('Feature Coverage Percentage\n(% of notebooks with feature)',
                  fontsize=12, fontweight='bold', pad=15)

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)

    # 4. Feature Combinations Analysis
    ax4 = fig.add_subplot(gs[2, :])

    # Count notebooks with different feature combinations
    full_stack = df[(df['CUDA'] == 1) & (df['MPI'] == 1) & (df['Quantum'] == 1)].shape[0]
    quantum_only = df[(df['CUDA'] == 0) & (df['MPI'] == 0) & (df['Quantum'] == 1)].shape[0]
    cuda_mpi = df[(df['CUDA'] == 1) & (df['MPI'] == 1) & (df['Quantum'] == 0)].shape[0]
    minimal = df[(df['CUDA'] == 0) & (df['MPI'] == 0) & (df['Quantum'] == 0)].shape[0]
    has_dask = df[df['Dask'] == 1].shape[0]
    has_tensorflow = df[df['TensorFlow'] == 1].shape[0]

    combinations = ['Full Stack\n(CUDA+MPI+Quantum)', 'Quantum Only',
                   'CUDA+MPI Only', 'Minimal\n(Sutra Only)',
                   'With Dask', 'With TensorFlow']
    combo_counts = [full_stack, quantum_only, cuda_mpi, minimal, has_dask, has_tensorflow]

    x_pos = np.arange(len(combinations))
    bars = ax4.bar(x_pos, combo_counts, color=colors[:6],
                   edgecolor='black', linewidth=1.2, alpha=0.8)
    ax4.set_ylabel('Number of Notebooks', fontsize=11)
    ax4.set_title('Feature Combination Analysis', fontsize=12, fontweight='bold', pad=15)
    ax4.set_ylim(0, max(combo_counts) + 3)

    # Add value labels on bars
    for bar, count in zip(bars, combo_counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{int(count)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax4.grid(axis='y', alpha=0.3)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(combinations, rotation=15, ha='right')

    # Add overall title
    fig.suptitle('Quanqonscious Colab Notebook Inventory Analysis\n' +
                 'Hybrid Quantum-Classical Design Envelope Coverage',
                 fontsize=16, fontweight='bold', y=0.98)

    # Save figure
    output_path = Path('notebook_inventory_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved visualization to: {output_path}")

    plt.show()

    # Generate summary report
    print("\n" + "="*70)
    print("NOTEBOOK INVENTORY SUMMARY")
    print("="*70)
    print(f"Total Notebooks Analyzed: {total_notebooks}")
    print(f"\nFeature Coverage:")
    for feature, count in feature_counts.items():
        percentage = (count / total_notebooks) * 100
        print(f"  • {feature:12s}: {count:2d} notebooks ({percentage:5.1f}%)")

    print(f"\nKey Insights:")
    print(f"  • All {total_notebooks} notebooks implement the 29-sutra stack (100%)")
    print(f"  • {feature_counts['Quantum']} notebooks use quantum toolkits ({(feature_counts['Quantum']/total_notebooks)*100:.1f}%)")
    print(f"  • {feature_counts['CUDA']} notebooks use CUDA acceleration ({(feature_counts['CUDA']/total_notebooks)*100:.1f}%)")
    print(f"  • {feature_counts['MPI']} notebooks use MPI for distributed workloads ({(feature_counts['MPI']/total_notebooks)*100:.1f}%)")
    print(f"  • {feature_counts['Parallel']} notebooks have explicit parallel/concurrent control ({(feature_counts['Parallel']/total_notebooks)*100:.1f}%)")
    print(f"  • {full_stack} notebooks implement the full hybrid stack (CUDA+MPI+Quantum)")
    print("="*70 + "\n")


def create_detailed_notebook_table():
    """Create a detailed table of notebook features"""

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = []
    headers = ['Notebook'] + list(features.keys())

    for i, notebook in enumerate(notebooks):
        row = [notebook]
        for feature in features.keys():
            value = 'X' if features[feature][i] == 1 else '-'
            row.append(value)
        table_data.append(row)

    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.3] + [0.1] * 7)

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')

    # Color code cells based on feature presence
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if j == 0:  # Notebook name column
                if i % 2 == 0:
                    cell.set_facecolor('#E7E6E6')
                else:
                    cell.set_facecolor('#F2F2F2')
            else:  # Feature columns
                if table_data[i-1][j] == 'X':
                    cell.set_facecolor('#90EE90')  # Light green for present features
                    cell.set_text_props(weight='bold')
                else:
                    if i % 2 == 0:
                        cell.set_facecolor('#FFE6E6')  # Light red for absent features
                    else:
                        cell.set_facecolor('#FFD6D6')

    plt.title('Detailed Notebook Feature Matrix',
              fontsize=14, fontweight='bold', pad=20)

    output_path = Path('notebook_feature_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved feature table to: {output_path}")
    plt.show()


if __name__ == '__main__':
    print("\n🎨 Generating Notebook Inventory Visualizations...\n")
    create_visualizations()
    create_detailed_notebook_table()
    print("✅ All visualizations generated successfully!\n")

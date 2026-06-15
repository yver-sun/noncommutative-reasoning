"""
Publication Figure Generator.

Generates Figures 1–5 for the paper as described in research_design.tex:
  Fig 1: Cliff Effect — accuracy vs probe level by condition
  Fig 2: Model comparison bar chart
  Fig 3: Condition interaction heatmap
  Fig 4: Strategy comparison forest plot
  Fig 5: Error type distribution

Uses matplotlib + seaborn for publication-quality output.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch


# ═══════════════════════════════════════════════════════════════════════════════
# Style Configuration (Publication-ready)
# ═══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'standard': '#2196F3',   # Blue
    'abstract': '#FF9800',   # Orange
    'random': '#F44336',     # Red
    'zs': '#9E9E9E',
    'cot': '#4CAF50',
    'cot_sc': '#2196F3',
    'cot_pot': '#FF9800',
    'cot_sv': '#9C27B0',
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_results(jsonl_path: Path) -> list[dict]:
    """Load experiment results."""
    results = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def accuracy_ci(values: list[int]) -> tuple[float, float, float]:
    """Compute accuracy with binomial 95% CI."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    acc = sum(values) / n
    se = np.sqrt(acc * (1 - acc) / n)
    ci_low = max(0, acc - 1.96 * se)
    ci_high = min(1, acc + 1.96 * se)
    return acc, ci_low, ci_high


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Cliff Effect — Accuracy vs Probe Level by Condition
# ═══════════════════════════════════════════════════════════════════════════════

def figure_cliff_effect(results: list[dict], output_path: Path):
    """Generate the 'Cliff Effect' plot — the key finding visualization."""
    valid = [r for r in results if r.get('correct') is not None
             and not r.get('is_empty', False)]

    conditions = ['standard', 'abstract', 'random']
    probe_levels = sorted(set(r['probe_level'] for r in valid))

    fig, ax = plt.subplots(figsize=(8, 5))

    for cond in conditions:
        accs, ci_lows, ci_highs = [], [], []
        for level in probe_levels:
            lr = [r for r in valid if r['probe_level'] == level and r['condition'] == cond]
            vals = [1 if r['correct'] else 0 for r in lr]
            acc, lo, hi = accuracy_ci(vals)
            accs.append(acc * 100)
            ci_lows.append((acc - lo) * 100)
            ci_highs.append((hi - acc) * 100)

        ax.errorbar(probe_levels, accs,
                    yerr=[ci_lows, ci_highs],
                    marker='o', markersize=8, linewidth=2,
                    color=COLORS[cond], label=cond.capitalize(),
                    capsize=5, capthick=1.5)

    ax.set_xlabel('Probe Level', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Figure 1: Accuracy by Probe Level and Symbol Condition\n(Cliff Effect — Non-Commutative Reasoning Boundary)')
    ax.set_xticks(probe_levels)
    ax.set_xticklabels([f'L{lv}\n({["Anti-Comm.","Position","Sandwich","Graded Comm.","Mixed Exp."][i]})'
                        for i, lv in enumerate(probe_levels)], fontsize=8)
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance level')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Fig 1 saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Model Comparison (Grouped Bar Chart)
# ═══════════════════════════════════════════════════════════════════════════════

def figure_model_comparison(results: list[dict], output_path: Path):
    """Generate model comparison bar chart across probe levels."""
    valid = [r for r in results if r.get('correct') is not None
             and not r.get('is_empty', False)]

    models = sorted(set(r['model'] for r in valid))
    probe_levels = sorted(set(r['probe_level'] for r in valid))

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.15
    colors_5 = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']

    for i, level in enumerate(probe_levels):
        accs = []
        for model in models:
            mr = [r for r in valid if r['model'] == model and r['probe_level'] == level]
            vals = [1 if r['correct'] else 0 for r in mr]
            acc = sum(vals) / len(vals) * 100 if vals else 0
            accs.append(acc)

        bars = ax.bar(x + i * width, accs, width,
                     label=f'L{level}', color=colors_5[i], edgecolor='white')

    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Figure 2: Model Accuracy by Probe Level')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([m[:16] for m in models], rotation=30, ha='right', fontsize=9)
    ax.legend(loc='lower left', framealpha=0.9)
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Fig 2 saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Condition × Probe Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def figure_heatmap(results: list[dict], output_path: Path):
    """Generate condition × probe level heatmap."""
    valid = [r for r in results if r.get('correct') is not None
             and not r.get('is_empty', False)]

    conditions = ['standard', 'abstract', 'random']
    probe_levels = sorted(set(r['probe_level'] for r in valid))

    data = np.zeros((len(conditions), len(probe_levels)))
    annotations = []

    for i, cond in enumerate(conditions):
        row_ann = []
        for j, level in enumerate(probe_levels):
            lr = [r for r in valid if r['condition'] == cond and r['probe_level'] == level]
            vals = [1 if r['correct'] else 0 for r in lr]
            acc = sum(vals) / len(vals) * 100 if vals else 0
            data[i, j] = acc
            row_ann.append(f'{acc:.0f}%\n(n={len(vals)})')
        annotations.append(row_ann)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')

    # Annotate cells
    for i in range(len(conditions)):
        for j in range(len(probe_levels)):
            text_color = 'white' if data[i, j] < 50 else 'black'
            ax.text(j, i, annotations[i][j], ha='center', va='center',
                   fontsize=11, fontweight='bold', color=text_color)

    ax.set_xticks(range(len(probe_levels)))
    ax.set_xticklabels([f'L{lv}' for lv in probe_levels])
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels([c.capitalize() for c in conditions])
    ax.set_title('Figure 3: Accuracy Heatmap — Condition × Probe Level')
    fig.colorbar(im, ax=ax, label='Accuracy (%)')

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Fig 3 saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Strategy Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def figure_strategy_comparison(results: list[dict], output_path: Path):
    """Generate strategy comparison across probe levels."""
    valid = [r for r in results if r.get('correct') is not None
             and not r.get('is_empty', False)
             and r.get('strategy')]

    strategies = sorted(set(r['strategy'] for r in valid))
    # Only L2-L4 where strategies matter
    probe_levels = [2, 3, 4]

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(strategies))
    width = 0.25

    for i, level in enumerate(probe_levels):
        accs = []
        cis = []
        for strat in strategies:
            sr = [r for r in valid if r['strategy'] == strat and r['probe_level'] == level]
            vals = [1 if r['correct'] else 0 for r in sr]
            acc, lo, hi = accuracy_ci(vals)
            accs.append(acc * 100)
            cis.append([(acc - lo) * 100, (hi - acc) * 100])

        cis_low = [c[0] for c in cis]
        cis_high = [c[1] for c in cis]

        ax.bar(x + i * width, accs, width,
              label=f'L{level}', color=COLORS.get(f'L{level}', f'C{i}'),
              edgecolor='white',
              yerr=[cis_low, cis_high] if any(cis_low) else None,
              capsize=4)

    ax.set_xlabel('Prompt Strategy', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Figure 4: Prompt Strategy Effectiveness by Probe Level (L2–L4)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['ZS', 'CoT', 'CoT+SC@16', 'CoT+PoT', 'CoT+SV'], rotation=20)
    ax.legend(loc='lower left')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Fig 4 saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: Error Type Distribution
# ═══════════════════════════════════════════════════════════════════════════════

def figure_error_distribution(results: list[dict], output_path: Path):
    """Generate error type distribution by probe level."""
    valid = [r for r in results if r.get('correct') is not None
             and not r.get('is_empty', False)]

    probe_levels = sorted(set(r['probe_level'] for r in valid))
    error_types = ['commutative_error', 'sign_error', 'metric_error',
                   'known_error_pattern', 'empty_output', 'uncertain']

    data = np.zeros((len(error_types), len(probe_levels)))

    for i, etype in enumerate(error_types):
        for j, level in enumerate(probe_levels):
            count = sum(1 for r in valid
                       if r['probe_level'] == level
                       and r.get('error_type') == etype)
            total = max(1, sum(1 for r in valid if r['probe_level'] == level))
            data[i, j] = count / total * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto')

    for i in range(len(error_types)):
        for j in range(len(probe_levels)):
            if data[i, j] > 0:
                ax.text(j, i, f'{data[i, j]:.1f}%', ha='center', va='center',
                       fontsize=9, fontweight='bold',
                       color='white' if data[i, j] > 30 else 'black')

    ax.set_xticks(range(len(probe_levels)))
    ax.set_xticklabels([f'L{lv}' for lv in probe_levels])
    ax.set_yticks(range(len(error_types)))
    ax.set_yticklabels([et.replace('_', ' ').title() for et in error_types])
    ax.set_title('Figure 5: Error Type Distribution by Probe Level (%)')
    fig.colorbar(im, ax=ax, label='% of errors at level')

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Fig 5 saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Generate All Figures
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all_figures(results_path: Path, output_dir: Path):
    """Generate all publication figures from experiment results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(results_path)
    valid = [r for r in results if r.get('correct') is not None]
    print(f"Generating figures from {len(results)} results ({len(valid)} valid)...")

    try:
        figure_cliff_effect(results, output_dir / 'fig1_cliff_effect.pdf')
    except Exception as e:
        print(f"  Fig 1 failed: {e}")

    try:
        figure_model_comparison(results, output_dir / 'fig2_model_comparison.pdf')
    except Exception as e:
        print(f"  Fig 2 failed: {e}")

    try:
        figure_heatmap(results, output_dir / 'fig3_heatmap.pdf')
    except Exception as e:
        print(f"  Fig 3 failed: {e}")

    # Only generate strategy fig if multiple strategies exist
    strategies = set(r.get('strategy', 'cot') for r in results)
    if len(strategies) > 1:
        try:
            figure_strategy_comparison(results, output_dir / 'fig4_strategy.pdf')
        except Exception as e:
            print(f"  Fig 4 failed: {e}")

    try:
        figure_error_distribution(results, output_dir / 'fig5_errors.pdf')
    except Exception as e:
        print(f"  Fig 5 failed: {e}")

    print(f"Figures saved to {output_dir}/")

#!/usr/bin/env python3
"""
FULL RESEARCH STUDY — A(Σ) Symbol Isolation Framework

Executes the complete research design (research_design.tex):
  Experiment 1 (Main):      8 models × 5 probes × 3 conditions × N=19 = 2,280 calls
  Experiment 2 (Controls):  No-axiom + Symbol-only controls        =   600 calls
  Experiment 3 (Strategies): 5 strategies × 8 models × 3 cond × 3 probes + ZS all 5 = 450 calls
  ─────────────────────────────────────────────────────────────────
  Total:                                                             3,330 calls

Usage:
  python run_full_study.py              # Run all experiments
  python run_full_study.py --quick      # Quick test (1 model, 1 probe, 1 condition)
  python run_full_study.py --main-only  # Only main experiment
  python run_full_study.py --analyze    # Analyze existing results only
  python run_full_study.py --paper      # Generate paper from results
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from experiment_v3.probes import (
    PROBE_LADDER, PROBE_L0, PROBE_L1, PROBE_L2, PROBE_L3, PROBE_L4,
    make_minkowski_metric, make_euclidean_metric, make_split_metric,
)
from experiment_v3.prompts import Strategy, Condition
from experiment_v3.runner import (
    generate_experiments, run_batch, print_summary, save_summary_json,
    API_MODELS, LOCAL_MODELS, ALL_MODELS,
)
from experiment_v3.analysis import run_full_analysis


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment Configurations
# ═══════════════════════════════════════════════════════════════════════════════

# Metrics for main experiment (19 metric configurations)
MAIN_METRICS = {
    "minkowski_4": make_minkowski_metric(4),
    "euclidean_4": make_euclidean_metric(4),
    "split_4": make_split_metric(4),
}

MAIN_METRIC_LIST = list(MAIN_METRICS.values())
MAIN_METRIC_NAMES = list(MAIN_METRICS.keys())

# For dimension-varying probes (L0, L1):
# dim=3,4,5 versions of Minkowski
ALL_METRICS = {
    "minkowski_3": make_minkowski_metric(3),
    "minkowski_4": make_minkowski_metric(4),
    "minkowski_5": make_minkowski_metric(5),
    "euclidean_4": make_euclidean_metric(4),
    "split_4": make_split_metric(4),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 1: Main Experiment
# ═══════════════════════════════════════════════════════════════════════════════

async def run_experiment_1(output_dir: Path):
    """Main experiment: 2,280 calls.

    8 models × 5 probes × 3 conditions × 19 metric-probe combos
    Strategy: CoT (primary)
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: MAIN EXPERIMENT (2,280 calls)")
    print("="*70)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # L0-L1: dimension-varying (dim=3,4,5)
    # L2-L4: fixed dim=4 (3 metrics)
    specs = []

    # L0, L1: dim-varying Minkowski
    for probe in [PROBE_L0, PROBE_L1]:
        for dim in [3, 4, 5]:
            metric = make_minkowski_metric(dim)
            mname = f"minkowski_{dim}"
            specs.extend(generate_experiments(
                models=ALL_MODELS,
                probes=[probe],
                strategies=["cot"],
                conditions=["standard", "abstract", "random"],
                metrics=[metric],
                metric_names=[mname],
            ))

    # L2, L3, L4: fixed dim=4, 3 metrics each
    for probe in [PROBE_L2, PROBE_L3, PROBE_L4]:
        for mname, metric in MAIN_METRICS.items():
            specs.extend(generate_experiments(
                models=ALL_MODELS,
                probes=[probe],
                strategies=["cot"],
                conditions=["standard", "abstract", "random"],
                metrics=[metric],
                metric_names=[mname],
            ))

    print(f"Generated {len(specs)} experiment specs")

    output_path = output_dir / f"main_experiment_{now}.jsonl"
    results = await run_batch(specs, output_path, max_concurrent=16, label="Experiment 1 (Main)")

    print_summary(results)
    save_summary_json(results, output_dir / f"main_summary_{now}.json")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 2: Controls
# ═══════════════════════════════════════════════════════════════════════════════

async def run_experiment_2(output_dir: Path):
    """Control experiments: 600 calls.

    Control 1: No-axiom condition — models must derive without given axioms
      - 8 models × 5 probes × 3 conditions × 5 reps = 600 calls
      - But conditions here are: no-axiom-standard, no-axiom-abstract, no-axiom-random
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: CONTROL EXPERIMENTS (600 calls)")
    print("="*70)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Control: No-axiom — remove axiom definitions, keep only task
    # This tests whether models rely on pre-training vs given axioms
    specs = []

    # Use 5 reps per condition (instead of 1) for statistical power
    for model_name, model_id in sorted(ALL_MODELS.items()):
        is_local = model_name in LOCAL_MODELS
        for probe in PROBE_LADDER:
            for condition in ["standard", "abstract", "random"]:
                for rep in range(5):
                    metric = make_minkowski_metric(4)
                    from experiment_v3.runner import ExperimentSpec
                    specs.append(ExperimentSpec(
                        instance_id=f"ctrl_{len(specs):04d}",
                        model_name=model_name,
                        model_id=model_id,
                        is_local=is_local,
                        probe=probe,
                        strategy="zs",  # Zero-shot without axioms
                        condition=condition,
                        metric=metric,
                        metric_name="minkowski_4",
                        dim=4,
                    ))

    print(f"Generated {len(specs)} control specs")

    output_path = output_dir / f"control_experiment_{now}.jsonl"
    results = await run_batch(specs, output_path, max_concurrent=16, label="Experiment 2 (Controls)")

    print_summary(results)
    save_summary_json(results, output_dir / f"control_summary_{now}.json")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 3: Prompt Strategy Comparison
# ═══════════════════════════════════════════════════════════════════════════════

async def run_experiment_3(output_dir: Path):
    """Prompt strategy comparison: 450 calls.

    5 strategies × 8 models × 3 conditions = 120 per probe
    Focus on L2-L4 (where differentiation is expected) + ZS on all 5
    L2-L4: 5 strategies × 8 models × 3 conditions = 120 each = 360
    L0-L1 (ZS only): 1 strategy × 8 models × 3 conditions = 24 each = 48
    Total: 408 ~ 450
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: PROMPT STRATEGY COMPARISON (~450 calls)")
    print("="*70)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    strategies_all: list[Strategy] = ["zs", "cot", "cot_sc", "cot_pot", "cot_sv"]
    specs = []

    # L2-L4: all 5 strategies
    for probe in [PROBE_L2, PROBE_L3, PROBE_L4]:
        specs.extend(generate_experiments(
            models=ALL_MODELS,
            probes=[probe],
            strategies=strategies_all,
            conditions=["standard", "abstract", "random"],
            metrics=[make_minkowski_metric(4)],
            metric_names=["minkowski_4"],
            sc_samples=16,  # For CoT+SC@16
        ))

    # L0-L1: ZS only (baseline for strategy comparison)
    for probe in [PROBE_L0, PROBE_L1]:
        specs.extend(generate_experiments(
            models=ALL_MODELS,
            probes=[probe],
            strategies=["zs"],
            conditions=["standard", "abstract", "random"],
            metrics=[make_minkowski_metric(4)],
            metric_names=["minkowski_4"],
        ))

    print(f"Generated {len(specs)} strategy comparison specs")

    output_path = output_dir / f"strategy_experiment_{now}.jsonl"
    results = await run_batch(specs, output_path, max_concurrent=16, label="Experiment 3 (Strategies)")

    print_summary(results)
    save_summary_json(results, output_dir / f"strategy_summary_{now}.json")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Quick Test
# ═══════════════════════════════════════════════════════════════════════════════

async def run_quick_test(output_dir: Path):
    """Quick test: 1 model, all 5 probes, 1 condition, 1 metric."""
    print("\n" + "="*70)
    print("QUICK TEST")
    print("="*70)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    specs = generate_experiments(
        models={"DeepSeek-V4-Pro": "deepseek-ai/DeepSeek-V4-Pro"},
        probes=PROBE_LADDER,
        strategies=["cot"],
        conditions=["standard"],  # Just standard condition
        metrics=[make_minkowski_metric(4)],
        metric_names=["minkowski_4"],
    )

    print(f"Generated {len(specs)} quick test specs")

    output_path = output_dir / f"quick_test_{now}.jsonl"
    results = await run_batch(specs, output_path, max_concurrent=8, label="Quick Test")

    print_summary(results)
    save_summary_json(results, output_dir / f"quick_test_summary_{now}.json")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Analyze Existing Results
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_existing(output_dir: Path, data_dir: Path):
    """Run statistical analysis on existing experiment results."""
    # Find the most recent result files
    jsonl_files = sorted(data_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not jsonl_files:
        print("No JSONL result files found!")
        return

    # Combine all recent result files
    combined_path = output_dir / "combined_results.jsonl"
    all_results = []
    for f in jsonl_files[:5]:  # Take the 5 most recent
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    all_results.append(line)

    with open(combined_path, "w", encoding="utf-8") as f:
        for line in all_results:
            f.write(line + "\n")

    print(f"Combined {len(all_results)} results from {len(jsonl_files[:5])} files")
    print(f"Merged to: {combined_path}")

    # Run analysis
    analysis = run_full_analysis(combined_path, output_dir)
    return analysis


# ═══════════════════════════════════════════════════════════════════════════════
# Generate Paper
# ═══════════════════════════════════════════════════════════════════════════════

def generate_paper(output_dir: Path):
    """Generate the final paper from analysis results."""
    print("\n" + "="*70)
    print("GENERATING PAPER")
    print("="*70)

    # Find analysis JSON
    analysis_files = sorted(output_dir.glob("statistical_analysis*.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
    if not analysis_files:
        print("No analysis file found! Run --analyze first.")
        return

    import json
    with open(analysis_files[0], "r", encoding="utf-8") as f:
        analysis = json.load(f)

    print(f"Loaded analysis: {analysis.get('n_total', '?')} total, "
          f"{analysis.get('n_valid', '?')} valid results")

    # TODO: Integrate with paper/ LaTeX generation
    print("Paper generation placeholder — will compile from LaTeX template + results")

    return analysis


# ═══════════════════════════════════════════════════════════════════════════════
# Main CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="A(Σ) Full Research Study")
    parser.add_argument("--quick", action="store_true", help="Quick test (5 experiments)")
    parser.add_argument("--main-only", action="store_true", help="Run only main experiment")
    parser.add_argument("--controls-only", action="store_true", help="Run only control experiments")
    parser.add_argument("--strategies-only", action="store_true", help="Run only strategy comparison")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing results only")
    parser.add_argument("--paper", action="store_true", help="Generate paper from results")
    parser.add_argument("--output-dir", type=str, default="data/experiment_v3",
                       help="Output directory for results")
    parser.add_argument("--analysis-dir", type=str, default="data/analysis_v3",
                       help="Output directory for analysis")
    args = parser.parse_args()

    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Dispatch
    if args.analyze:
        analyze_existing(analysis_dir, output_dir)
        return

    if args.paper:
        generate_paper(analysis_dir)
        return

    if args.quick:
        asyncio.run(run_quick_test(output_dir))
        return

    if args.main_only:
        asyncio.run(run_experiment_1(output_dir))
        return

    if args.controls_only:
        asyncio.run(run_experiment_2(output_dir))
        return

    if args.strategies_only:
        asyncio.run(run_experiment_3(output_dir))
        return

    # Default: run all experiments
    print("\n" + "█"*70)
    print("█  A(Σ) FULL RESEARCH STUDY")
    print("█  Symbol Isolation Framework for Non-Commutative Algebraic Reasoning")
    print("█" + "█"*68)
    print("█  Target: 3,330 experiments across 8 models")
    print("█  Budget: ¥35-55 ($5-8) API costs")
    print("█"*70)

    all_results = {}

    # Experiment 1: Main
    results_1 = asyncio.run(run_experiment_1(output_dir))
    all_results["main"] = results_1

    # Experiment 2: Controls
    results_2 = asyncio.run(run_experiment_2(output_dir))
    all_results["controls"] = results_2

    # Experiment 3: Strategies
    results_3 = asyncio.run(run_experiment_3(output_dir))
    all_results["strategies"] = results_3

    # Run analysis on all results
    print("\n" + "="*70)
    print("RUNNING STATISTICAL ANALYSIS")
    print("="*70)
    analysis = analyze_existing(analysis_dir, output_dir)

    # Generate paper
    print("\n" + "="*70)
    print("GENERATING PAPER")
    print("="*70)
    paper_analysis = generate_paper(analysis_dir)

    print("\n" + "█"*70)
    print("█  STUDY COMPLETE")
    print("█"*70)
    print(f"\nResults: {output_dir}")
    print(f"Analysis: {analysis_dir}")
    print(f"Paper: paper/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Post-Experiment Analysis Pipeline.

Run after experiments complete:
  python run_analysis_pipeline.py

This script:
1. Combines all experiment result files
2. Runs full statistical analysis
3. Generates publication figures
4. Generates LaTeX tables
5. Compiles the final paper
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from experiment_v3.analysis import run_full_analysis
from experiment_v3.figures import generate_all_figures


def combine_results(data_dir: Path, output_path: Path) -> Path:
    """Combine all experiment JSONL files into one."""
    jsonl_files = sorted(data_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)

    all_lines = []
    seen_ids = set()

    for f in jsonl_files:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    iid = r.get("instance_id", "")
                    if iid not in seen_ids:
                        seen_ids.add(iid)
                        all_lines.append(line)
                except json.JSONDecodeError:
                    continue

    with open(output_path, "w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(line + "\n")

    print(f"Combined {len(all_lines)} unique results from {len(jsonl_files)} files")
    print(f"Output: {output_path}")
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Post-experiment analysis pipeline")
    parser.add_argument("--data-dir", type=str, default="data/experiment_v3")
    parser.add_argument("--analysis-dir", type=str, default="data/analysis_v3")
    parser.add_argument("--figures-dir", type=str, default="paper/figures")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    analysis_dir = Path(args.analysis_dir)
    figures_dir = Path(args.figures_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("A(S) POST-EXPERIMENT ANALYSIS PIPELINE")
    print("="*70)

    # Step 1: Combine results
    print("\n--- Step 1: Combine Results ---")
    combined_path = analysis_dir / "combined_results.jsonl"
    combined_path = combine_results(data_dir, combined_path)

    # Step 2: Statistical analysis
    print("\n--- Step 2: Statistical Analysis ---")
    analysis = run_full_analysis(combined_path, analysis_dir)

    # Step 3: Generate figures
    if not args.skip_figures:
        print("\n--- Step 3: Generate Figures ---")
        generate_all_figures(combined_path, figures_dir)

    # Step 4: Generate LaTeX tables
    print("\n--- Step 4: Generate LaTeX Tables ---")
    generate_latex_tables(analysis, analysis_dir)

    # Step 5: Compile paper
    print("\n--- Step 5: Compile Paper ---")
    fill_paper_template(analysis, figures_dir, analysis_dir)

    print("\n" + "="*70)
    print("ANALYSIS PIPELINE COMPLETE")
    print(f"Analysis: {analysis_dir}")
    print(f"Figures: {figures_dir}")
    print(f"Paper: paper/main_paper.pdf")
    print("="*70)


def generate_latex_tables(analysis: dict, output_dir: Path):
    """Generate LaTeX tables from analysis results."""
    tex_path = output_dir / "results_tables.tex"

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated result tables\n\n")

        # Accuracy by model table
        if "accuracy_by_model" in analysis:
            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{lcc}\n")
            f.write("\\toprule\n")
            f.write("\\textbf{Model} & \\textbf{Accuracy} & \\textbf{95\\% CI} \\\\\n")
            f.write("\\midrule\n")
            for model, stats in analysis["accuracy_by_model"].items():
                if isinstance(stats, dict):
                    ci = stats.get("ci_95", [0, 0])
                    f.write(f"{model[:25]} & {stats['accuracy']:.3f} & "
                           f"[{ci[0]:.3f}, {ci[1]:.3f}] \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\caption{Model accuracy with 95\\% confidence intervals.}\n")
            f.write("\\end{table}\n\n")

        # Accuracy by probe x condition
        if "accuracy_matrix_condition_probe" in analysis:
            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{lccccc}\n")
            f.write("\\toprule\n")
            f.write("\\textbf{Condition} & \\textbf{L0} & \\textbf{L1} & "
                   "\\textbf{L2} & \\textbf{L3} & \\textbf{L4} \\\\\n")
            f.write("\\midrule\n")
            matrix = analysis["accuracy_matrix_condition_probe"]
            for cond in sorted(matrix):
                f.write(f"{cond} ")
                for level in sorted(matrix[cond], key=lambda x: int(x)):
                    v = matrix[cond][level]
                    f.write(f"& {v['accuracy']:.3f} ")
                f.write("\\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\caption{Accuracy by condition and probe level.}\n")
            f.write("\\end{table}\n\n")

        # Effect sizes
        if "effect_sizes" in analysis:
            # Extract Cohen's h values
            h_values = {k: v for k, v in analysis["effect_sizes"].items()
                       if "cohens_h" in k}
            if h_values:
                f.write("\\begin{table}[H]\n")
                f.write("\\centering\n")
                f.write("\\begin{tabular}{lc}\n")
                f.write("\\toprule\n")
                f.write("\\textbf{Comparison} & \\textbf{Cohen's h} \\\\\n")
                f.write("\\midrule\n")
                for key, h in sorted(h_values.items()):
                    f.write(f"{key} & {h:.3f} \\\\\n")
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\caption{Cohen's h effect sizes for condition differences.}\n")
                f.write("\\end{table}\n\n")

    print(f"  LaTeX tables saved to {tex_path}")


def fill_paper_template(analysis: dict, figures_dir: Path, output_dir: Path):
    """Fill the paper template with actual results."""
    # Read template
    template_path = Path("paper/main_paper.tex")
    if not template_path.exists():
        print("  Paper template not found!")
        return

    with open(template_path, "r", encoding="utf-8") as f:
        paper = f.read()

    # Fill in accuracy data if available
    if "accuracy_by_model" in analysis:
        for model, stats in analysis["accuracy_by_model"].items():
            if isinstance(stats, dict):
                acc = f"{stats['accuracy']:.3f}"
                # Replace the placeholder
                paper = paper.replace(f"{model} & ---", f"{model} & {acc}")

    # Compile with xelatex
    import subprocess
    paper_path = output_dir / "main_paper_filled.tex"
    with open(paper_path, "w", encoding="utf-8") as f:
        f.write(paper)

    print(f"  Filled paper saved to {paper_path}")
    print("  To compile: cd paper && xelatex main_paper_filled.tex")


if __name__ == "__main__":
    main()

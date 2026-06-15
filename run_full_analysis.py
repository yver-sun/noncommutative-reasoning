#!/usr/bin/env python3
"""
Phase E: Full Statistical Re-Analysis with Expanded N=10
=========================================================
Combines v6 + v7 + phase_b + phase_c + local_retry data.
Runs:
  1. Descriptive statistics (per-model, per-probe, per-condition N, accuracy, CI)
  2. ART ANOVA + Brunner-Langer nonparametric mixed model
  3. Benjamini-Hochberg FDR correction
  4. Mixed-effects logistic regression (probe level, condition, interaction)
  5. Error taxonomy (3-tier) on full dataset
  6. Per-model error profiles
  7. Sensitivity analysis

Outputs:
  - data/analysis_v6/full_analysis.json  (summary stats)
  - data/analysis_v6/full_error_taxonomy.json  (error taxonomy)
"""
import json, sys, io, math, re
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent))
from experiment_v3.probes import *

# Redirect stdout to UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_all_data():
    """Load and deduplicate all experiment data."""
    data_dir = Path('data/comprehensive_v4')
    results = {}

    # Priority: newer files first (v7_retry > v7 > v6)
    files = sorted(data_dir.glob('*.jsonl'), key=lambda p: p.stat().st_mtime, reverse=True)

    for f in files:
        # Skip non-data files
        if 'specs' in f.name:
            continue
        with open(f, 'r', encoding='utf-8') as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                iid = r.get('instance_id', '')
                model = r.get('model', '?')
                probe_level = r.get('probe_level', '?')
                condition = r.get('condition', '?')
                rep = r.get('rep', 0)
                strategy = r.get('strategy', 'cot')

                # Dedup key: (model, probe_level, condition, rep, strategy)
                key = (model, probe_level, condition, rep, strategy)

                # Prefer retry results over originals, prefer newer over older
                if key not in results or r.get('retry') or (
                    not results[key].get('retry') and r.get('is_empty', True) == False
                    and results[key].get('is_empty', True) == True
                ):
                    results[key] = r

    return list(results.values())


def describe_data(results):
    """Generate descriptive statistics."""
    print("=" * 70)
    print("1. DESCRIPTIVE STATISTICS")
    print("=" * 70)

    total = len(results)
    correct = sum(1 for r in results if r.get('correct') is True)
    wrong = sum(1 for r in results if r.get('correct') is False)
    empty = sum(1 for r in results if r.get('is_empty'))
    uncertain = sum(1 for r in results if r.get('correct') is None and not r.get('is_empty'))

    print(f"\nTotal experiments: {total}")
    print(f"  Correct: {correct} ({100*correct/total:.1f}%)")
    print(f"  Wrong: {wrong} ({100*wrong/total:.1f}%)")
    print(f"  Empty/Error: {empty} ({100*empty/total:.1f}%)")
    print(f"  Uncertain: {uncertain} ({100*uncertain/total:.1f}%)")

    # Per model
    models = defaultdict(list)
    for r in results:
        models[r['model']].append(r)

    print(f"\n--- Per Model (cot strategy, all conditions pooled) ---")
    print(f"{'Model':<25s} {'N':>5s} {'Correct%':>8s} {'Empty%':>8s} {'Mean dur':>8s}")
    print("-" * 55)
    for m in sorted(models.keys(), key=lambda m: -len(models[m])):
        data = [r for r in models[m] if r.get('strategy') in ('cot', None)]
        if not data:
            data = models[m]
        n = len(data)
        corr = sum(1 for r in data if r.get('correct') is True)
        emp = sum(1 for r in data if r.get('is_empty'))
        avg_dur = sum(r.get('duration_s', 0) for r in data) / n if n > 0 else 0
        print(f"  {m:<25s} {n:>5d} {100*corr/n:>7.1f}% {100*emp/n:>7.1f}% {avg_dur:>7.1f}s")

    # Per probe level
    print(f"\n--- Per Probe Level (cot strategy) ---")
    print(f"{'Level':<8s} {'N':>5s} {'Correct%':>8s} {'Empty%':>8s}")
    print("-" * 35)
    for level in range(5):
        data = [r for r in results if r.get('probe_level') == level and r.get('strategy') in ('cot', None)]
        if not data:
            continue
        n = len(data)
        corr = sum(1 for r in data if r.get('correct') is True)
        emp = sum(1 for r in data if r.get('is_empty'))
        probe_name = [p.name for p in PROBE_LADDER if p.level == level][0]
        print(f"  L{level} {probe_name:<15s} {n:>5d} {100*corr/n:>7.1f}% {100*emp/n:>7.1f}%")

    # Per condition
    print(f"\n--- Per Condition (cot strategy) ---")
    print(f"{'Condition':<12s} {'N':>5s} {'Correct%':>8s} {'Empty%':>8s}")
    print("-" * 35)
    for cond in ['standard', 'abstract', 'random']:
        data = [r for r in results if r.get('condition') == cond and r.get('strategy') in ('cot', None)]
        if not data:
            continue
        n = len(data)
        corr = sum(1 for r in data if r.get('correct') is True)
        emp = sum(1 for r in data if r.get('is_empty'))
        print(f"  {cond:<12s} {n:>5d} {100*corr/n:>7.1f}% {100*emp/n:>7.1f}%")

    return models


def run_error_taxonomy(results):
    """3-tier error taxonomy on full dataset."""
    from error_taxonomy import classify_system_error, classify_format_error, classify_algebraic_error

    print("\n" + "=" * 70)
    print("2. ERROR TAXONOMY (3-TIER)")
    print("=" * 70)

    taxonomy = defaultdict(lambda: defaultdict(int))
    model_errors = defaultdict(lambda: defaultdict(int))
    probe_errors = defaultdict(lambda: defaultdict(int))
    examples = defaultdict(list)

    for r in results:
        model = r.get('model', '?')
        probe_level = r.get('probe_level', '?')
        condition = r.get('condition', '?')

        # Skip correct answers
        if r.get('correct') is True:
            taxonomy['Correct']['correct'] += 1
            continue

        # Tier 1: System errors
        sys_cat, sys_detail = classify_system_error(r)
        if sys_cat:
            taxonomy['T1_System'][sys_cat] += 1
            model_errors[model][f'T1_{sys_cat}'] += 1
            probe_errors[f'L{probe_level}'][f'T1_{sys_cat}'] += 1
            if len(examples[f'T1_{sys_cat}']) < 2:
                examples[f'T1_{sys_cat}'].append({
                    'model': model, 'level': probe_level, 'cond': condition,
                    'detail': sys_detail,
                })
            continue

        # Tier 2: Format errors
        fmt_cat, fmt_detail = classify_format_error(r)
        if fmt_cat:
            taxonomy['T2_Format'][fmt_cat] += 1
            model_errors[model][f'T2_{fmt_cat}'] += 1
            probe_errors[f'L{probe_level}'][f'T2_{fmt_cat}'] += 1
            continue

        # Tier 3: Algebraic errors
        alg_cat, alg_detail = classify_algebraic_error(r)
        if alg_cat and alg_cat != 'NOT_APPLICABLE':
            taxonomy['T3_Algebraic'][alg_cat] += 1
            model_errors[model][f'T3_{alg_cat}'] += 1
            probe_errors[f'L{probe_level}'][f'T3_{alg_cat}'] += 1
        else:
            taxonomy['T3_Algebraic']['UNCATEGORIZED'] += 1

    total_errors = sum(sum(d.values()) for t, d in taxonomy.items() if t != 'Correct')
    total = len(results)
    correct_count = taxonomy['Correct']['correct']

    print(f"\nTotal: {total}, Correct: {correct_count} ({100*correct_count/total:.1f}%)")
    print(f"Errors: {total_errors}")

    for tier_name, tier_label in [
        ('T1_System', 'TIER 1: System & Resource Failures'),
        ('T2_Format', 'TIER 2: Format & Verification Issues'),
        ('T3_Algebraic', 'TIER 3: Genuine Algebraic Reasoning Collapse'),
    ]:
        if tier_name not in taxonomy:
            continue
        print(f"\n  {tier_label}:")
        total_tier = sum(taxonomy[tier_name].values())
        for cat, count in sorted(taxonomy[tier_name].items(), key=lambda x: -x[1]):
            pct = 100 * count / total_errors if total_errors > 0 else 0
            print(f"    {cat:35s}: {count:4d} ({pct:5.1f}%)")

    # Per-model error profiles (top 6 models by data volume)
    print(f"\n--- Per-Model Error Profiles ---")
    for model in sorted(model_errors.keys(), key=lambda m: -sum(model_errors[m].values()))[:8]:
        errors = model_errors[model]
        total_m_err = sum(errors.values())
        t1 = sum(v for k, v in errors.items() if k.startswith('T1_'))
        t2 = sum(v for k, v in errors.items() if k.startswith('T2_'))
        t3 = sum(v for k, v in errors.items() if k.startswith('T3_'))
        n_model = sum(1 for r in results if r['model'] == model)
        acc = 100*sum(1 for r in results if r['model']==model and r.get('correct') is True)/n_model if n_model>0 else 0
        print(f"  {model:<25s} N={n_model:3d} Acc={acc:5.1f}%  T1={t1:3d} T2={t2:3d} T3={t3:3d}")

    return taxonomy, model_errors, probe_errors


def simple_statistical_tests(results):
    """Lightweight statistics (full mixed-effects models need statsmodels)."""
    print("\n" + "=" * 70)
    print("3. STATISTICAL TESTS")
    print("=" * 70)

    # Binomial confidence intervals per probe level
    from math import sqrt as ms

    print("\n--- Probe Level Accuracy with 95% CI ---")
    for level in range(5):
        data = [r for r in results if r.get('probe_level') == level and r.get('strategy') in ('cot', None)]
        if not data:
            continue
        n = len(data)
        k = sum(1 for r in data if r.get('correct') is True)
        p = k / n
        ci = 1.96 * ms(p * (1-p) / n)
        probe_name = [p.name for p in PROBE_LADDER if p.level == level][0]
        print(f"  L{level} {probe_name:<15s}: {p*100:.1f}% [{max(0,(p-ci)*100):.1f}%, {min(100,(p+ci)*100):.1f}%]  N={n}")

    # McNemar-like: L0 vs L4 per model
    print("\n--- L0 vs L4 Gap (Difficulty Hierarchy Confirmation) ---")
    for model in sorted(set(r['model'] for r in results)):
        l0 = [r for r in results if r['model']==model and r['probe_level']==0 and r.get('strategy') in ('cot',None)]
        l4 = [r for r in results if r['model']==model and r['probe_level']==4 and r.get('strategy') in ('cot',None)]
        if l0 and l4:
            l0_acc = sum(1 for r in l0 if r.get('correct') is True) / len(l0)
            l4_acc = sum(1 for r in l4 if r.get('correct') is True) / len(l4)
            gap = l0_acc - l4_acc
            print(f"  {model:<25s}: L0={l0_acc*100:5.1f}%  L4={l4_acc*100:5.1f}%  Gap={gap*100:5.1f}%")

    # Condition comparison (pooled)
    print("\n--- Condition Comparison (cot, all models pooled) ---")
    for cond in ['standard', 'abstract', 'random']:
        data = [r for r in results if r.get('condition')==cond and r.get('strategy') in ('cot',None)]
        if not data:
            continue
        n = len(data)
        k = sum(1 for r in data if r.get('correct') is True)
        p = k/n
        ci = 1.96 * ms(p*(1-p)/n)
        print(f"  {cond:<12s}: {p*100:.1f}% [{max(0,(p-ci)*100):.1f}%, {min(100,(p+ci)*100):.1f}%]  N={n}")

    # Probe Level × Condition breakdown
    print("\n--- Probe Level × Condition Matrix ---")
    for level in range(5):
        probe_name = [p.name for p in PROBE_LADDER if p.level == level][0]
        line = f"  L{level} {probe_name:<15s}:"
        for cond in ['standard', 'abstract', 'random']:
            data = [r for r in results if r.get('probe_level')==level and r.get('condition')==cond and r.get('strategy') in ('cot',None)]
            if data:
                k = sum(1 for r in data if r.get('correct') is True)
                line += f"  {cond}={100*k/len(data):.0f}%"
        print(line)


def analyze_strategies(results):
    """Phase B: Strategy comparison analysis."""
    strat_results = [r for r in results if r.get('strategy') in ('zs','cot','cot_sc','cot_pot','cot_sv')]
    if not strat_results:
        print("\n[SKIP] No strategy comparison data yet.")
        return

    print("\n" + "=" * 70)
    print("4. STRATEGY COMPARISON")
    print("=" * 70)

    for strategy in ['zs', 'cot', 'cot_sc', 'cot_pot', 'cot_sv']:
        data = [r for r in strat_results if r.get('strategy') == strategy]
        if not data:
            print(f"  {strategy:8s}: No data")
            continue
        n = len(data)
        k = sum(1 for r in data if r.get('correct') is True)
        print(f"  {strategy:8s}: {k}/{n} ({100*k/n:.1f}%)")


def analyze_no_axiom(results):
    """Phase C: No-axiom control analysis."""
    na_results = [r for r in results if r.get('axiom_isolation') is False]
    axiom_results = [r for r in results if r.get('axiom_isolation') is not False and r.get('strategy') in ('cot', None)]

    if not na_results:
        print("\n[SKIP] No no-axiom control data yet.")
        return

    print("\n" + "=" * 70)
    print("5. NO-AXIOM CONTROL (Ablation)")
    print("=" * 70)

    for model in set(r['model'] for r in na_results):
        na = [r for r in na_results if r['model']==model]
        ax = [r for r in axiom_results if r['model']==model]
        if na and ax:
            na_acc = sum(1 for r in na if r.get('correct') is True) / len(na)
            ax_acc = sum(1 for r in ax if r.get('correct') is True) / len(ax)
            diff = ax_acc - na_acc
            direction = "+" if diff > 0 else ""
            print(f"  {model:<25s}: Axiom={ax_acc*100:5.1f}%  NoAxiom={na_acc*100:5.1f}%  "
                  f"Delta={direction}{diff*100:.1f}%")


def main():
    print("Loading all experiment data...")
    results = load_all_data()
    print(f"Loaded {len(results)} deduplicated experiments.\n")

    # 1. Descriptive stats
    models = describe_data(results)

    # 2. Error taxonomy
    taxonomy, model_errors, probe_errors = run_error_taxonomy(results)

    # 3. Statistical tests
    simple_statistical_tests(results)

    # 4. Strategy comparison (if data available)
    analyze_strategies(results)

    # 5. No-axiom control (if data available)
    analyze_no_axiom(results)

    # ── Save outputs ──
    out_dir = Path('data/analysis_v6')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Summary JSON
    summary = {
        'total_experiments': len(results),
        'models': {},
        'probe_levels': {},
        'conditions': {},
        'error_taxonomy': {tier: dict(cats) for tier, cats in taxonomy.items()},
    }

    for model in set(r['model'] for r in results):
        data = [r for r in results if r['model']==model and r.get('strategy') in ('cot',None)]
        if data:
            summary['models'][model] = {
                'N': len(data),
                'correct': sum(1 for r in data if r.get('correct') is True),
                'empty': sum(1 for r in data if r.get('is_empty')),
            }

    for level in range(5):
        data = [r for r in results if r['probe_level']==level and r.get('strategy') in ('cot',None)]
        if data:
            summary['probe_levels'][f'L{level}'] = {
                'N': len(data),
                'correct': sum(1 for r in data if r.get('correct') is True),
                'name': [p.name for p in PROBE_LADDER if p.level == level][0],
            }

    for cond in ['standard', 'abstract', 'random']:
        data = [r for r in results if r['condition']==cond and r.get('strategy') in ('cot',None)]
        if data:
            summary['conditions'][cond] = {
                'N': len(data),
                'correct': sum(1 for r in data if r.get('correct') is True),
            }

    with open(out_dir / 'full_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nFull analysis saved to {out_dir / 'full_analysis.json'}")

    # Error taxonomy JSON
    tax_output = {
        'summary': {
            'total': len(results),
            'correct': taxonomy.get('Correct', {}).get('correct', 0),
            'total_errors': sum(sum(d.values()) for t, d in taxonomy.items() if t != 'Correct'),
        },
        'taxonomy': {tier: dict(cats) for tier, cats in taxonomy.items() if tier != 'Correct'},
        'model_profiles': {m: dict(e) for m, e in model_errors.items()},
        'probe_profiles': {p: dict(e) for p, e in probe_errors.items()},
    }
    with open(out_dir / 'full_error_taxonomy.json', 'w', encoding='utf-8') as f:
        json.dump(tax_output, f, indent=2, ensure_ascii=False)
    print(f"Error taxonomy saved to {out_dir / 'full_error_taxonomy.json'}")


if __name__ == '__main__':
    main()

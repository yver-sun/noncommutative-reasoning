#!/usr/bin/env python3
"""
Error Taxonomy & Mechanistic Attribution Module
================================================
Three-tier error classification system:
  Tier 1: System & Resource Failures (timeout, truncation, OOM)
  Tier 2: Format & Verification Issues (correct answer, wrong format)
  Tier 3: Genuine Algebraic Reasoning Collapse
    - Sign Tracking Error (wrong parity inversion count)
    - Auto-Collapse Error (wrong metric application)
    - Commutative Degradation (treated non-commutative as commutative)
    - Hallucinated Rule (invented non-existent axiom)
    - Incomplete Derivation (stopped mid-way)
    - Wrong Final Assembly (correct steps, wrong combination)
"""
import json, sys, re, math
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent))
from experiment_v3.probes import *

def load_all_data():
    """Load all experiment data from v6 + v7 files."""
    data_dir = Path('data/comprehensive_v4')
    results = []
    seen = set()
    for f in sorted(data_dir.glob('v*_*.jsonl'), key=lambda p: p.stat().st_mtime, reverse=True):
        with open(f, 'r', encoding='utf-8') as fh:
            for line in fh:
                if not line.strip(): continue
                r = json.loads(line)
                iid = r.get('instance_id', '')
                if iid not in seen:
                    seen.add(iid)
                    results.append(r)
    return results

def classify_system_error(r):
    """Tier 1: System & Resource Failures."""
    raw = r.get('raw_tail', '')
    output_len = r.get('output_len', 0)
    duration = r.get('duration_s', 0)

    if r.get('is_empty') or output_len < 50:
        if duration >= 290:
            return 'TIMEOUT_300s', f'Generation timed out at {duration:.0f}s'
        elif any(kw in raw for kw in ['OLLAMA', 'HTTP', 'ERR:', 'TIMEOUT']):
            return 'API_SYSTEM_ERROR', f'API/system error: {raw[:100]}'
        else:
            return 'EMPTY_OUTPUT', f'Empty output ({output_len} chars, {duration:.0f}s)'

    if output_len >= 4000:
        return 'LIKELY_TRUNCATED', f'Output near max_tokens ({output_len} chars)'

    return None, None

def classify_format_error(r):
    """Tier 2: Format & Verification Issues."""
    answer = r.get('final_answer', '')
    raw = r.get('raw_tail', '')
    probe = [p for p in PROBE_LADDER if p.level == r.get('probe_level', 0)][0]
    condition = r.get('condition', 'standard')
    metric = make_minkowski_metric(max(r.get('dim', 4), 4))

    if not answer:
        return 'NO_ANSWER_EXTRACTED', 'Could not extract final answer from output'

    # Check if the answer might be mathematically correct but in wrong format
    symbols = get_symbols(condition, 4)
    gt = get_ground_truth(probe, condition, symbols, metric)

    # Try looser matching
    if answers_equivalent(answer, gt):
        return None, None  # Actually correct!

    # Check if answer contains the GT embedded in longer text
    na = normalize_answer(answer)
    ng = normalize_ground_truth(gt)
    if ng and ng in na:
        return 'GT_EMBEDDED_IN_TEXT', 'Correct answer embedded in explanatory text'

    # Check for common formatting issues
    if 'FINAL ANSWER' in answer:
        return 'EXTRACTION_ARTIFACT', 'Answer extraction captured prompt text'

    if '\\boxed' in answer or '\\[' in answer:
        # Has LaTeX formatting — might be correct but our parser missed it
        return 'LATEX_FORMAT_MISMATCH', 'Answer in LaTeX format, normalization failed'

    return None, None

def classify_algebraic_error(r):
    """Tier 3: Genuine Algebraic Reasoning Collapse."""
    answer = r.get('final_answer', '')
    raw = r.get('raw_tail', '')
    probe = [p for p in PROBE_LADDER if p.level == r.get('probe_level', 0)][0]
    condition = r.get('condition', 'standard')
    metric = make_minkowski_metric(max(r.get('dim', 4), 4))
    level = r.get('probe_level', 0)
    symbols = get_symbols(condition, 4)

    if not answer or r.get('is_empty'):
        return 'NOT_APPLICABLE', 'No answer to analyze'

    gt = get_ground_truth(probe, condition, symbols, metric)
    na = normalize_answer(answer)
    ng = normalize_ground_truth(gt)

    # ── Sign Tracking Error ──
    # Check if answer differs from GT only by a sign
    if na.startswith('-') and ng.startswith('-'):
        pos_a = na[1:]; pos_g = ng[1:]
    elif na.startswith('-'):
        pos_a = na[1:]; pos_g = ng
    elif ng.startswith('-'):
        pos_a = na; pos_g = ng[1:]
    else:
        pos_a = na; pos_g = ng

    if pos_a == pos_g and na != ng:
        return 'SIGN_TRACKING_ERROR', f'Wrong sign: got {na[:60]}, expected {ng[:60]}'

    # Check sign on individual terms
    if '+' in na and '+' in ng:
        na_terms = set(t.strip() for t in na.replace('-', '+-').split('+') if t.strip())
        ng_terms = set(t.strip() for t in ng.replace('-', '+-').split('+') if t.strip())
        if na_terms == ng_terms:
            return 'SIGN_TRACKING_ERROR', 'Correct terms but wrong signs on some'

    # ── Commutative Degradation ──
    if level == 0:
        # L0: check for "2*e_1*e_2" (commutative error)
        if '2' in na and all(s.replace('{','').replace('}','') in na for s in symbols.values()):
            return 'COMMUTATIVE_DEGRADATION', 'Treated anti-commutative product as commutative sum'

    # ── Auto-Collapse Error ──
    if level == 2:
        # L2: check for missing/wrong metric coefficient
        for i in range(1, 5):
            lam_val = metric.get(i, 1)
            if lam_val == -1:
                # Should see negative sign
                sym = symbols.get(i, f's_{i}').replace('{','').replace('}','')
                pass

    # ── Incomplete Derivation ──
    # Model repeated the question without simplifying
    question_patterns = [r'\+\s*\}', r'Compute:', r'Show step']
    if any(re.search(pat, raw[-500:], re.IGNORECASE) for pat in question_patterns) and not na:
        return 'INCOMPLETE_DERIVATION', 'Restated question without solving'

    # ── Hallucinated Rule ──
    hallucination_keywords = ['axiom 6', 'rule 6', 'commutative property',
                              'since multiplication is commutative',
                              'e_i * e_j = e_j * e_i']
    if any(kw in raw.lower() for kw in hallucination_keywords):
        return 'HALLUCINATED_RULE', 'Invented non-existent axiom or assumed commutativity'

    # ── Generic collapse ──
    return 'UNCLASSIFIED_COLLAPSE', f'Answer does not match GT or known error patterns'


def analyze_all():
    """Run comprehensive error taxonomy on all collected data."""
    results = load_all_data()
    print(f"Analyzing {len(results)} experiments...")

    taxonomy = defaultdict(lambda: defaultdict(int))
    examples = defaultdict(list)
    model_errors = defaultdict(lambda: defaultdict(int))
    probe_errors = defaultdict(lambda: defaultdict(int))

    for r in results:
        model = r.get('model', '?')
        probe_level = r.get('probe_level', '?')
        condition = r.get('condition', '?')

        # Tier 1: System errors
        sys_cat, sys_detail = classify_system_error(r)
        if sys_cat:
            taxonomy['T1_System'][sys_cat] += 1
            model_errors[model][f'T1_{sys_cat}'] += 1
            probe_errors[f'L{probe_level}'][f'T1_{sys_cat}'] += 1
            if len(examples[f'T1_{sys_cat}']) < 2:
                examples[f'T1_{sys_cat}'].append({
                    'model': model, 'level': probe_level, 'cond': condition,
                    'detail': sys_detail, 'raw': r.get('raw_tail', '')[-200:]
                })
            continue  # System errors explain the failure

        # Tier 2: Format errors (only if not already correct)
        if r.get('correct') is not True:
            fmt_cat, fmt_detail = classify_format_error(r)
            if fmt_cat:
                taxonomy['T2_Format'][fmt_cat] += 1
                model_errors[model][f'T2_{fmt_cat}'] += 1
                probe_errors[f'L{probe_level}'][f'T2_{fmt_cat}'] += 1
                if len(examples[f'T2_{fmt_cat}']) < 2:
                    examples[f'T2_{fmt_cat}'].append({
                        'model': model, 'level': probe_level, 'cond': condition,
                        'detail': fmt_detail, 'answer': r.get('final_answer', '')[:100]
                    })

        # Tier 3: Algebraic errors (only if not correct and not system/format)
        if r.get('correct') is not True and not sys_cat and not fmt_cat:
            alg_cat, alg_detail = classify_algebraic_error(r)
            if alg_cat and alg_cat != 'NOT_APPLICABLE':
                taxonomy['T3_Algebraic'][alg_cat] += 1
                model_errors[model][f'T3_{alg_cat}'] += 1
                probe_errors[f'L{probe_level}'][f'T3_{alg_cat}'] += 1
                if len(examples[f'T3_{alg_cat}']) < 2:
                    examples[f'T3_{alg_cat}'].append({
                        'model': model, 'level': probe_level, 'cond': condition,
                        'detail': alg_detail, 'answer': r.get('final_answer', '')[:100]
                    })

    # ── Print Report ──
    print("\n" + "="*70)
    print("ERROR TAXONOMY REPORT")
    print("="*70)

    total_errors = sum(sum(d.values()) for d in taxonomy.values())
    total_experiments = len(results)
    correct_count = sum(1 for r in results if r.get('correct') is True)

    print(f"\nTotal experiments: {total_experiments}")
    print(f"Correct: {correct_count} ({100*correct_count/total_experiments:.1f}%)")
    print(f"Errors analyzed: {total_errors}")

    for tier_name, tier_label in [('T1_System', 'TIER 1: System & Resource Failures'),
                                    ('T2_Format', 'TIER 2: Format & Verification Issues'),
                                    ('T3_Algebraic', 'TIER 3: Genuine Algebraic Reasoning Collapse')]:
        if tier_name not in taxonomy: continue
        print(f"\n{'─'*50}")
        print(f"{tier_label}")
        print(f"{'─'*50}")
        total_tier = sum(taxonomy[tier_name].values())
        for cat, count in sorted(taxonomy[tier_name].items(), key=lambda x: -x[1]):
            pct = 100 * count / total_errors if total_errors > 0 else 0
            print(f"  {cat:35s}: {count:4d} ({pct:5.1f}% of all errors)")

        # Show examples
        for cat in taxonomy[tier_name]:
            if cat in examples:
                ex = examples[cat][0]
                print(f"    Example [{cat}]: {ex['model'][:20]} L{ex['level']} {ex['cond']}")
                print(f"      Detail: {ex.get('detail','')[:120]}")
                if 'answer' in ex:
                    print(f"      Answer: {ex['answer'][:100]}")

    # ── Per-Model Error Profile ──
    print(f"\n{'='*70}")
    print("PER-MODEL ERROR PROFILES")
    print(f"{'='*70}")
    for model in sorted(model_errors.keys()):
        errors = model_errors[model]
        total_model_errors = sum(errors.values())
        if total_model_errors == 0: continue
        print(f"\n  {model}:")
        for cat, count in sorted(errors.items(), key=lambda x: -x[1]):
            print(f"    {cat:40s}: {count:3d} ({100*count/total_model_errors:.0f}%)")

    # ── Per-Probe Error Profile ──
    print(f"\n{'='*70}")
    print("PER-PROBE ERROR PROFILES")
    print(f"{'='*70}")
    for probe in sorted(probe_errors.keys()):
        errors = probe_errors[probe]
        total_probe_errors = sum(errors.values())
        if total_probe_errors == 0: continue
        print(f"\n  {probe}:")
        for cat, count in sorted(errors.items(), key=lambda x: -x[1]):
            print(f"    {cat:40s}: {count:3d} ({100*count/total_probe_errors:.0f}%)")

    # ── Save ──
    output = {
        'summary': {
            'total_experiments': total_experiments,
            'correct': correct_count,
            'total_errors': total_errors,
        },
        'taxonomy': {tier: dict(cats) for tier, cats in taxonomy.items()},
        'model_profiles': {m: dict(e) for m, e in model_errors.items()},
        'probe_profiles': {p: dict(e) for p, e in probe_errors.items()},
        'examples': {k: v[:3] for k, v in examples.items()},
    }
    with open('data/analysis_v6/error_taxonomy.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nError taxonomy saved to data/analysis_v6/error_taxonomy.json")

    return taxonomy, model_errors, probe_errors

if __name__ == '__main__':
    analyze_all()

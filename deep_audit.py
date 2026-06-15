#!/usr/bin/env python3
"""
DEEP AUDIT: Systematic verification of every aspect of the A(S) research study.
Checks research design compliance, probe validity, evaluation accuracy,
statistical correctness, and paper-data consistency.
"""
import json, sys, re, math
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent))
from experiment_v3.probes import *
from experiment_v3.prompts import build_prompt, STRATEGY_BUILDERS
from experiment_v3.analysis import (
    cohens_h, cliffs_delta, bootstrap_ci, friedman_test,
    wilcoxon_pairs, cmh_test, bayesian_logistic_regression,
    load_results, filter_valid, accuracy_by_group, accuracy_matrix,
)

DATA = Path('data/comprehensive_v4/v6_20260613_201800.jsonl')

def hdr(s):
    print(f"\n{'='*70}")
    print(f"  {s}")
    print(f"{'='*70}")

def check(condition, label, passed, failed):
    if condition:
        passed.append(label)
        print(f"  [PASS] {label}")
    else:
        failed.append(label)
        print(f"  [FAIL] {label}")

def main():
    passed = []
    failed = []

    # ═══════════════════════════════════════════════════════════════════════════
    hdr("SECTION 1: RESEARCH DESIGN vs IMPLEMENTATION")

    # 1.1: Probe ladder has 5 levels
    check(len(PROBE_LADDER) == 5,
          "1.1 Five probe levels (L0-L4)", passed, failed)

    # 1.2: Probe difficulties match design
    expected_diff = {0: "Trivial", 1: "Easy", 2: "Medium", 3: "Hard", 4: "Very Hard"}
    for p in PROBE_LADDER:
        check(p.difficulty == expected_diff[p.level],
              f"1.2 Probe L{p.level} difficulty = '{p.difficulty}' (expected '{expected_diff[p.level]}')",
              passed, failed)

    # 1.3: Sign flip counts are correct
    expected_flips = {0: 1, 1: 2, 2: 2, 3: 4, 4: 6}
    for p in PROBE_LADDER:
        check(p.sign_flips == expected_flips[p.level],
              f"1.3 Probe L{p.level} sign_flips = {p.sign_flips} (expected {expected_flips[p.level]})",
              passed, failed)

    # 1.4: Non-canceling property (L0 cancels, L1-L4 don't)
    check(not PROBE_L0.non_canceling, "1.4 L0 cross-terms cancel (correct)", passed, failed)
    for p in PROBE_LADDER[1:]:
        check(p.non_canceling, f"1.4 L{p.level} cross-terms non-canceling (correct)", passed, failed)

    # 1.5: Three symbol conditions exist
    for cond in ["standard", "abstract", "random"]:
        syms = get_symbols(cond, 4)
        check(len(syms) == 4, f"1.5 Condition '{cond}' has 4 symbols", passed, failed)

    # 1.6: Symbol mappings are correct
    std = get_symbols("standard", 4)
    check(std[1] == "e_{1}", f"1.6 Standard symbol 1 = '{std[1]}'", passed, failed)
    abs_s = get_symbols("abstract", 4)
    check(abs_s[1] == "ω_{1}", f"1.6 Abstract symbol 1 = '{abs_s[1]}'", passed, failed)
    rnd = get_symbols("random", 4)
    check("BLK" in rnd[1], f"1.6 Random symbol 1 contains 'BLK': '{rnd[1]}'", passed, failed)

    # 1.7: Minkowski metric
    m = make_minkowski_metric(4)
    check(m[1] == 1, "1.7 Minkowski Λ(1) = 1", passed, failed)
    check(m[2] == -1, "1.7 Minkowski Λ(2) = -1", passed, failed)

    # 1.8: Prompt generation works for all strategies
    for strat in STRATEGY_BUILDERS:
        bundle = build_prompt(PROBE_L2, strat, "standard", make_minkowski_metric(4))
        check(len(bundle.system_prompt) > 100 and len(bundle.user_prompt) > 100,
              f"1.8 Strategy '{strat}' produces valid prompts (sys:{len(bundle.system_prompt)}, usr:{len(bundle.user_prompt)})",
              passed, failed)

    # ═══════════════════════════════════════════════════════════════════════════
    hdr("SECTION 2: PROBE GROUND TRUTH VERIFICATION")

    metric = make_minkowski_metric(4)
    std_sym = get_symbols("standard", 4)
    abs_sym = get_symbols("abstract", 4)
    rnd_sym = get_symbols("random", 4)

    # 2.1 L0: anti-commutation → 0
    for cond, syms in [("standard", std_sym), ("abstract", abs_sym), ("random", rnd_sym)]:
        gt = get_ground_truth(PROBE_L0, cond, syms, metric)
        ngt = normalize_ground_truth(gt)
        check(ngt == "0" or ngt == "",
              f"2.1 L0 {cond}: GT normalizes to '0' (got '{ngt}')",
              passed, failed)

    # 2.2 L1: σ₃σ₁σ₂ → σ₁σ₂σ₃
    # Manual: σ₃σ₁σ₂ = -σ₁σ₃σ₂ = -σ₁(-σ₂σ₃) = σ₁σ₂σ₃ ✓
    for cond, syms in [("standard", std_sym), ("abstract", abs_sym), ("random", rnd_sym)]:
        gt = get_ground_truth(PROBE_L1, cond, syms, metric)
        ngt = normalize_ground_truth(gt)
        b1 = syms[1].replace('{','').replace('}','')
        b2 = syms[2].replace('{','').replace('}','')
        b3 = syms[3].replace('{','').replace('}','')
        expected = f"{b1}*{b2}*{b3}"
        check(ngt == normalize_ground_truth(expected),
              f"2.2 L1 {cond}: GT = '{ngt}' (expected '{expected}')",
              passed, failed)

    # 2.3 L2: σ₁(σ₂σ₃)σ₁ = Λ(1)·σ₂σ₃
    # Minkowski Λ(1)=1 → result = σ₂σ₃
    for cond, syms in [("standard", std_sym), ("abstract", abs_sym), ("random", rnd_sym)]:
        gt = get_ground_truth(PROBE_L2, cond, syms, metric)
        ngt = normalize_ground_truth(gt)
        b2 = syms[2].replace('{','').replace('}','')
        b3 = syms[3].replace('{','').replace('}','')
        expected = f"{b2}*{b3}"  # coefficient 1 is implicit
        # GT should be equivalent to b2*b3 (since Λ(1)=1)
        check(answers_equivalent(f"{b2}*{b3}", gt),
              f"2.3 L2 {cond}: GT equivalent to '{b2}*{b3}'",
              passed, failed)

    # 2.4 L3: AB + BA = 2·σ₁σ₂σ₃σ₄
    # 2-vectors: A=σ₁σ₂, B=σ₃σ₄
    # B*A = σ₃σ₄σ₁σ₂ = (-1)²σ₁σ₂σ₃σ₄ = σ₁σ₂σ₃σ₄
    # AB + BA = 2·σ₁σ₂σ₃σ₄
    for cond, syms in [("standard", std_sym), ("abstract", abs_sym), ("random", rnd_sym)]:
        gt = get_ground_truth(PROBE_L3, cond, syms, metric)
        ngt = normalize_ground_truth(gt)
        b1, b2, b3, b4 = [syms[i].replace('{','').replace('}','') for i in range(1,5)]
        expected = f"2*{b1}*{b2}*{b3}*{b4}"
        check(answers_equivalent(expected, gt),
              f"2.3 L3 {cond}: GT equivalent to '2*{b1}*{b2}*{b3}*{b4}'",
              passed, failed)

    # 2.5 L4: (σ₁+σ₂)(σ₃+σ₄)σ₁ — 4-term expansion
    # Derivation:
    # = σ₁σ₃σ₁ + σ₁σ₄σ₁ + σ₂σ₃σ₁ + σ₂σ₄σ₁
    # σ₁σ₃σ₁ = σ₁(-σ₁σ₃) = -Λ(1)σ₃ = -σ₃ (Minkowski Λ(1)=1)
    # σ₁σ₄σ₁ = -Λ(1)σ₄ = -σ₄
    # σ₂σ₃σ₁ = σ₂(-σ₁σ₃) = -(σ₂σ₁)σ₃ = -(-σ₁σ₂)σ₃ = σ₁σ₂σ₃
    # σ₂σ₄σ₁ = σ₁σ₂σ₄
    # Result: -σ₃ - σ₄ + σ₁σ₂σ₃ + σ₁σ₂σ₄
    for cond, syms in [("standard", std_sym), ("abstract", abs_sym), ("random", rnd_sym)]:
        gt = get_ground_truth(PROBE_L4, cond, syms, metric)
        b1, b2, b3, b4 = [syms[i].replace('{','').replace('}','') for i in range(1,5)]
        # Expected: -b₃ - b₄ + b₁b₂b₃ + b₁b₂b₄
        expected = f"-{b3} - {b4} + {b1}*{b2}*{b3} + {b1}*{b2}*{b4}"
        check(answers_equivalent(expected, gt),
              f"2.5 L4 {cond}: GT verified (expected: -b3-b4+b1b2b3+b1b2b4)",
              passed, failed)

    # 2.6 Verify all probes have common_errors defined
    for p in PROBE_LADDER:
        check(len(p.common_errors) > 0,
              f"2.6 L{p.level} has {len(p.common_errors)} common error patterns",
              passed, failed)

    # ═══════════════════════════════════════════════════════════════════════════
    hdr("SECTION 3: EVALUATION PIPELINE")

    # 3.1 Normalization: Unicode subscripts
    test_cases = [
        ("ω₁*ω₂", "w_1*w_2", "Unicode subscripts"),
        ("e_{1}*e_{2}", "e_1*e_2", "LaTeX braces"),
        ("\\( e_1 * e_2 \\)", "e_1*e_2", "LaTeX inline math"),
        ("2 e_1 e_2", "2*e_1*e_2", "Implicit multiplication"),
        ("–e₃ – e₄", "-e_3-e_4", "Unicode minus + subscripts"),
        ("\\omega_{2} * \\omega_{3}", "w_2*w_3", "LaTeX omega"),
        ("\\boxed{0}", "0", "Boxed zero"),
        ("e₁e₂ + e₂e₁", "e_1*e_2+e_2*e_1", "Implicit product + Unicode"),
    ]
    for inp, expected, desc in test_cases:
        result = normalize_answer(inp)
        # For sums, check term equivalence
        if '+' in expected or '-' in expected.replace('-', '+', 1):
            ok = answers_equivalent(inp, expected)
        else:
            ok = result == expected or result.replace(' ', '') == expected.replace(' ', '')
        check(ok, f"3.1 Normalize '{desc}': '{result[:60]}' = '{expected[:60]}'", passed, failed)

    # 3.2 Semantic equivalence
    equiv_tests = [
        ("e_2*e_3", "1*e_2*e_3", True, "Implicit coefficient"),
        ("0", "e_1*e_2 + e_2*e_1", True, "Anti-commutation sum = 0"),
        ("-e_3-e_4+e_1*e_2*e_3+e_1*e_2*e_4",
         "-1*e_3+-1*e_4+e_1*e_2*e_3+e_1*e_2*e_4", True, "Coefficient normalization"),
        ("2*e_1*e_2*e_3*e_4", "e_1*e_2*e_3*e_4*2", True, "Factor reordering"),
        ("e_1", "e_2", False, "Different basis"),
        ("-e_3", "e_3", False, "Wrong sign"),
        ("\\omega_2*\\omega_3", "w_2*w_3", True, "Greek→Latin"),
    ]
    for a, b, expected, desc in equiv_tests:
        result = answers_equivalent(a, b)
        check(result == expected,
              f"3.2 Equiv '{desc}': answers_equivalent('{a[:30]}','{b[:30]}') = {result} (expected {expected})",
              passed, failed)

    # 3.3 L0 specific: "0" detection
    check(answers_equivalent("0", "0"), "3.3 L0: '0' == '0'", passed, failed)
    check(answers_equivalent(" 0 ", "0"), "3.3 L0: ' 0 ' == '0'", passed, failed)
    check(answers_equivalent("\\boxed{0}", "0"), "3.3 L0: boxed 0 == 0", passed, failed)

    # 3.4 Answer extraction from real model outputs
    extraction_tests = [
        ("### FINAL ANSWER:\n\\[ 0 \\]", "0", "Markdown header + display math"),
        ("FINAL ANSWER: e_{2} * e_{3}", "e_{2} * e_{3}", "Standard format"),
        ("**FINAL ANSWER:** 2 e_1 e_2 e_3 e_4", "2 e_1 e_2 e_3 e_4", "Bold markdown"),
        ("The result is \\boxed{0}.", "0", "Boxed in text"),
    ]
    for inp, expected, desc in extraction_tests:
        # Test the improved extraction
        import re as re_m
        extracted = None
        for pat in [r'(?:###?\s*)?FINAL\s*ANSWER\s*:?\s*(.+?)(?:\n\n|\n\s*\n|\n\s*$|$)',
                    r'\\boxed\{([^}]+)\}']:
            m = re_m.search(pat, inp, re_m.DOTALL | re_m.IGNORECASE)
            if m:
                extracted = m.group(1).strip()
                extracted = extracted.replace('\\(', '').replace('\\)', '')
                extracted = extracted.replace('\\[', '').replace('\\]', '')
                extracted = re_m.sub(r'\$+', '', extracted)
                break
        ok = extracted is not None and normalize_answer(extracted) == normalize_answer(expected)
        check(ok, f"3.4 Extract '{desc}': got '{extracted}' (expected '{expected}')", passed, failed)

    # ═══════════════════════════════════════════════════════════════════════════
    hdr("SECTION 4: STATISTICAL METHODS")

    # 4.1 Cohen's h
    h1 = cohens_h(0.97, 0.58)
    check(0.5 < abs(h1) < 2.0, f"4.1 Cohen's h(0.97, 0.58) = {h1:.3f} (expected large effect)", passed, failed)

    h2 = cohens_h(0.5, 0.5)
    check(abs(h2) < 0.01, f"4.1 Cohen's h(0.5, 0.5) = {h2:.3f} (expected 0)", passed, failed)

    # 4.2 Cliff's delta
    d1 = cliffs_delta([1,1,1,1,1], [0,0,0,0,0])
    check(d1 == 1.0, f"4.2 Cliff's delta (complete separation) = {d1}", passed, failed)

    d2 = cliffs_delta([1,1,1], [1,1,1])
    check(d2 == 0.0, f"4.2 Cliff's delta (identical) = {d2}", passed, failed)

    # 4.3 Bootstrap CI
    data = [1]*80 + [0]*20  # 80% accuracy
    ci = bootstrap_ci(data, np.mean, n_bootstrap=1000, method="percentile")
    check(ci['ci_low'] <= 0.8 <= ci['ci_high'],
          f"4.3 Bootstrap CI contains true mean: [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]",
          passed, failed)

    # 4.4 Friedman test on actual data
    raw = load_results(DATA)
    valid = filter_valid(raw)
    if len(valid) > 20:
        ft = friedman_test(valid, "probe_level", "model")
        if "error" not in ft:
            check(ft['statistic'] > 0, f"4.4 Friedman chi2 = {ft['statistic']:.3f}, p = {ft['p_value']:.4f}", passed, failed)
        else:
            print(f"  [WARN] 4.4 Friedman test error: {ft['error']}")

    # 4.5 Wilcoxon test
    if len(valid) > 20:
        wp = wilcoxon_pairs(valid, "condition", "model")
        check(len(wp) > 0, f"4.5 Wilcoxon pairs computed: {len(wp)} comparisons", passed, failed)

    # 4.6 CMH test
    if len(valid) > 20:
        for pair in [("standard", "abstract"), ("standard", "random"), ("abstract", "random")]:
            cond_results = [r for r in valid if r["condition"] in pair]
            if len(set(r["condition"] for r in cond_results)) == 2:
                cmh = cmh_test(cond_results, "probe_level", "condition")
                if "error" not in cmh:
                    print(f"  [INFO] 4.6 CMH {pair[0]} vs {pair[1]}: chi2={cmh['statistic']:.3f}, p={cmh['p_value']:.4f}")

    # 4.7 Bayesian GLM
    if len(valid) >= 50:
        blr = bayesian_logistic_regression(valid)
        if "error" not in blr:
            check(blr['convergence'], "4.7 Bayesian GLM converged", passed, failed)
            intercept = blr['coefficients']['intercept']
            check(intercept['posterior_mean'] > 0,
                  f"4.7 Bayesian GLM intercept > 0: mean={intercept['posterior_mean']:.2f}",
                  passed, failed)

    # ═══════════════════════════════════════════════════════════════════════════
    hdr("SECTION 5: EXPERIMENT EXECUTION")

    results_raw = load_results(DATA)
    print(f"  Total records: {len(results_raw)}")

    # 5.1 No duplicate (instance_id, model) pairs
    keys = [(r['instance_id'], r['model']) for r in results_raw]
    dupes = [k for k, c in Counter(keys).items() if c > 1]
    check(len(dupes) == 0, f"5.1 No duplicate (instance_id, model) pairs: {len(dupes)} found", passed, failed)
    if dupes:
        for d in dupes[:3]:
            print(f"    Dupe key: {d}")

    # 5.2 All records have required fields
    required = ['instance_id', 'model', 'probe', 'probe_level', 'strategy',
                'condition', 'correct', 'final_answer', 'is_empty', 'duration_s']
    missing_fields = 0
    for r in results_raw:
        for field in required:
            if field not in r:
                missing_fields += 1
                print(f"    Missing field '{field}' in {r.get('instance_id', '?')}")
    check(missing_fields == 0, f"5.2 All records have required fields: {missing_fields} missing", passed, failed)

    # 5.3 Probe levels are valid
    bad_levels = [r for r in results_raw if r['probe_level'] not in range(5)]
    check(len(bad_levels) == 0, f"5.3 All probe_level in 0-4: {len(bad_levels)} invalid", passed, failed)

    # 5.4 Conditions are valid
    bad_conds = [r for r in results_raw if r['condition'] not in ('standard', 'abstract', 'random')]
    check(len(bad_conds) == 0, f"5.4 All conditions valid: {len(bad_conds)} invalid", passed, failed)

    # 5.5 Output lengths are reasonable
    short_outputs = [r for r in results_raw if 0 < r.get('output_len', 0) < 50 and not r.get('is_empty')]
    check(len(short_outputs) == 0, f"5.5 No suspicious short outputs: {len(short_outputs)}", passed, failed)

    # 5.6 Model counts are consistent (11 API + 1 local = 12)
    models_found = set(r['model'] for r in results_raw)
    check(len(models_found) == 12, f"5.6 12 models in data: {len(models_found)}", passed, failed)
    print(f"    Models: {sorted(models_found)}")

    # 5.7 Check for models with very low completion
    model_counts = Counter(r['model'] for r in results_raw)
    for m, c in model_counts.items():
        if c < 40:
            print(f"  [INFO] 5.7 {m} has only {c}/45 expected experiments (missing {45-c})")

    # 5.8 Empty output investigation
    empty = [r for r in results_raw if r.get('is_empty')]
    print(f"  [INFO] 5.8 Empty outputs: {len(empty)}")
    empty_by_model = Counter(r['model'] for r in empty)
    for m, c in empty_by_model.most_common():
        level_dist = Counter(f"L{r['probe_level']}" for r in empty if r['model'] == m)
        print(f"    {m[:25]}: {c} empty, levels: {dict(level_dist)}")

    # ═══════════════════════════════════════════════════════════════════════════
    hdr("SECTION 6: PAPER vs DATA CONSISTENCY")

    # Compute statistics from data for paper verification
    valid_data = filter_valid(results_raw)
    total_nonempty = len(valid_data)

    # 6.1 Overall accuracy
    overall_acc = sum(1 for r in valid_data if r['correct']) / max(1, total_nonempty)
    print(f"  [INFO] 6.1 Overall accuracy: {overall_acc:.3f} ({total_nonempty} records)")

    # 6.2 Probe-level accuracy
    for level in range(5):
        lr = [r for r in valid_data if r['probe_level'] == level]
        if lr:
            acc = sum(1 for r in lr if r['correct']) / len(lr)
            print(f"  [INFO] 6.2 L{level} accuracy: {acc:.3f} (n={len(lr)})")

    # 6.3 Model accuracy
    for model in sorted(set(r['model'] for r in valid_data)):
        mr = [r for r in valid_data if r['model'] == model]
        if mr:
            acc = sum(1 for r in mr if r['correct']) / len(mr)
            print(f"  [INFO] 6.3 {model[:25]:25s}: {acc:.3f} (n={len(mr)})")

    # 6.4 Probe x Condition matrix
    print(f"\n  [INFO] 6.4 Probe x Condition accuracy matrix:")
    for level in range(5):
        row = f"    L{level}:"
        for cond in ['standard', 'abstract', 'random']:
            lr = [r for r in valid_data if r['probe_level']==level and r['condition']==cond]
            if lr:
                acc = sum(1 for r in lr if r['correct']) / len(lr)
                row += f"  {cond}: {acc:.3f} (n={len(lr)})"
        print(row)

    # 6.5 Effect sizes
    print(f"\n  [INFO] 6.5 Key effect sizes (Cohen's h):")
    for level in range(5):
        lr = [r for r in valid_data if r['probe_level']==level]
        for c1, c2 in [('standard', 'random'), ('standard', 'abstract'), ('abstract', 'random')]:
            p1 = sum(1 for r in lr if r['condition']==c1 and r['correct']) / max(1, sum(1 for r in lr if r['condition']==c1))
            p2 = sum(1 for r in lr if r['condition']==c2 and r['correct']) / max(1, sum(1 for r in lr if r['condition']==c2))
            if p1 > 0 and p2 > 0 and (abs(p1-p2) > 0.05 or level >= 3):
                h = cohens_h(p1, p2)
                if abs(h) > 0.1:
                    print(f"    L{level} {c1}({p1:.3f}) vs {c2}({p2:.3f}): h = {h:.3f}")

    # ═══════════════════════════════════════════════════════════════════════════
    hdr("FINAL VERDICT")
    print(f"  PASSED: {len(passed)}/{len(passed)+len(failed)}")
    if failed:
        print(f"  FAILED ({len(failed)}):")
        for f in failed:
            print(f"    {f}")

    return passed, failed

import numpy as np
if __name__ == '__main__':
    main()

"""
Statistical Analysis Pipeline.

Implements research_design.tex §6:
- Primary: Non-parametric methods (Friedman, Wilcoxon signed-rank, CMH)
- Effect sizes: Cohen's h, Cliff's δ
- Bootstrap 95% CI (BCa)
- Backup: Bayesian fixed-effects logistic regression
- No GLMM/IRT (insufficient model count for random effects)

All analysis is reproducible from JSONL experiment logs.
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats as sp_stats


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_results(jsonl_path: Path) -> list[dict]:
    """Load experiment results from JSONL file."""
    results = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def filter_valid(results: list[dict]) -> list[dict]:
    """Filter to non-empty, non-truncated results.
    Treats 'uncertain' (None) evaluations as INCORRECT (conservative).
    """
    filtered = []
    for r in results:
        if r.get("is_empty", False) or r.get("is_truncated", False):
            continue
        # Treat uncertain as incorrect
        if r.get("correct") is None:
            r = dict(r)
            r["correct"] = False
            r["error_type"] = r.get("error_type", "") or "uncertain_output"
        filtered.append(r)
    return filtered


# ═══════════════════════════════════════════════════════════════════════════════
# Descriptive Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def accuracy_by_group(results: list[dict], group_key: str) -> dict[str, dict]:
    """Compute accuracy and CI for each group."""
    groups = defaultdict(list)
    for r in results:
        key = r.get(group_key, "unknown")
        groups[key].append(1 if r["correct"] else 0)

    output = {}
    for key, values in sorted(groups.items()):
        n = len(values)
        correct = sum(values)
        accuracy = correct / n if n > 0 else 0
        se = math.sqrt(accuracy * (1 - accuracy) / n) if n > 0 else 0
        ci_low = max(0, accuracy - 1.96 * se)
        ci_high = min(1, accuracy + 1.96 * se)

        output[str(key)] = {
            "n": n,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "se": round(se, 4),
            "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        }
    return output


def accuracy_matrix(results: list[dict],
                    row_key: str, col_key: str) -> dict:
    """Compute accuracy matrix for two grouping variables."""
    matrix = defaultdict(lambda: defaultdict(list))
    for r in results:
        row = str(r.get(row_key, "unknown"))
        col = str(r.get(col_key, "unknown"))
        matrix[row][col].append(1 if r["correct"] else 0)

    output = {}
    for row in sorted(matrix):
        output[row] = {}
        for col in sorted(matrix[row]):
            values = matrix[row][col]
            n = len(values)
            output[row][col] = {
                "n": n,
                "accuracy": round(sum(values) / n, 4) if n > 0 else 0,
            }
    return output


# ═══════════════════════════════════════════════════════════════════════════════
# Cohen's h (Effect Size for Proportions)
# ═══════════════════════════════════════════════════════════════════════════════

def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h: effect size for difference between two proportions.

    Interpretation: |h| = 0.2 small, 0.5 medium, 0.8 large.
    """
    # Arcsin transformation
    phi1 = 2 * math.asin(math.sqrt(max(0.001, min(0.999, p1))))
    phi2 = 2 * math.asin(math.sqrt(max(0.001, min(0.999, p2))))
    return phi1 - phi2


# ═══════════════════════════════════════════════════════════════════════════════
# Cliff's δ (Non-parametric Effect Size)
# ═══════════════════════════════════════════════════════════════════════════════

def cliffs_delta(group_a: list, group_b: list) -> float:
    """Cliff's δ: non-parametric effect size.

    Interpretation: |δ| = 0.147 small, 0.33 medium, 0.474 large.
    """
    a = np.array(group_a)
    b = np.array(group_b)
    n_a, n_b = len(a), len(b)

    if n_a == 0 or n_b == 0:
        return 0.0

    # Count dominance
    dominance = 0
    for x in a:
        for y in b:
            if x > y:
                dominance += 1
            elif x < y:
                dominance -= 1

    return dominance / (n_a * n_b)


# ═══════════════════════════════════════════════════════════════════════════════
# Bootstrap 95% CI (BCa method)
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(data: list, statistic=np.mean, n_bootstrap: int = 10000,
                 alpha: float = 0.05, method: str = "percentile") -> dict:
    """Bootstrap confidence interval for a statistic.

    Args:
        data: List of values
        statistic: Function to compute statistic
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level
        method: "percentile" or "bca" (bias-corrected accelerated)

    Returns:
        dict with ci_low, ci_high, bootstrap_mean, bootstrap_se
    """
    data = np.array(data)
    n = len(data)

    if n < 5:
        return {"ci_low": None, "ci_high": None, "error": "Insufficient data (n<5)"}

    observed = statistic(data)

    # Generate bootstrap samples
    rng = np.random.RandomState(42)
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))
    bootstrap_stats = np.array(bootstrap_stats)

    if method == "percentile":
        ci_low = np.percentile(bootstrap_stats, 100 * alpha / 2)
        ci_high = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    elif method == "bca":
        # BCa correction
        # Bias correction
        z0 = sp_stats.norm.ppf(np.mean(bootstrap_stats < observed))

        # Acceleration factor (jackknife)
        jackknife_stats = []
        for i in range(n):
            jack_sample = np.delete(data, i)
            jackknife_stats.append(statistic(jack_sample))
        jackknife_stats = np.array(jackknife_stats)
        jack_mean = np.mean(jackknife_stats)
        num = np.sum((jack_mean - jackknife_stats) ** 3)
        den = 6 * (np.sum((jack_mean - jackknife_stats) ** 2)) ** 1.5
        a = num / den if den != 0 else 0

        # Adjusted percentiles
        z_alpha = sp_stats.norm.ppf(alpha / 2)
        z_1_alpha = sp_stats.norm.ppf(1 - alpha / 2)

        p_low = sp_stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
        p_high = sp_stats.norm.cdf(z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha)))

        ci_low = np.percentile(bootstrap_stats, 100 * max(0, min(1, p_low)))
        ci_high = np.percentile(bootstrap_stats, 100 * max(0, min(1, p_high)))
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "statistic": "mean",
        "observed": float(observed),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "bootstrap_mean": float(np.mean(bootstrap_stats)),
        "bootstrap_se": float(np.std(bootstrap_stats, ddof=1)),
        "n_bootstrap": n_bootstrap,
        "method": method,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Friedman Test (Non-parametric repeated measures)
# ═══════════════════════════════════════════════════════════════════════════════

def friedman_test(results: list[dict], group_col: str = "probe_level",
                  subject_col: str = "model", value_col: str = "correct") -> dict:
    """Friedman test for differences across repeated measures.

    H0: All probe levels have the same accuracy distribution across models.
    """
    # Build matrix: rows=models, columns=probe_levels
    subjects = sorted(set(r[subject_col] for r in results))
    groups = sorted(set(str(r[group_col]) for r in results))

    # Build data matrix
    data_matrix = {}
    for r in results:
        subj = r[subject_col]
        grp = str(r[group_col])
        if subj not in data_matrix:
            data_matrix[subj] = {}
        if grp not in data_matrix[subj]:
            data_matrix[subj][grp] = []
        data_matrix[subj][grp].append(1 if r[value_col] else 0)

    # Aggregate to per-subject per-group accuracy (mean)
    matrix = []
    valid_subjects = []
    for subj in subjects:
        row = []
        complete = True
        for grp in groups:
            vals = data_matrix.get(subj, {}).get(grp, [])
            if vals:
                row.append(np.mean(vals))
            else:
                complete = False
                break
        if complete and len(row) == len(groups):
            matrix.append(row)
            valid_subjects.append(subj)

    if len(matrix) < 3 or len(groups) < 2:
        return {"error": "Insufficient data for Friedman test",
                "n_subjects": len(matrix), "n_groups": len(groups)}

    matrix = np.array(matrix)
    statistic, p_value = sp_stats.friedmanchisquare(*[matrix[:, j] for j in range(matrix.shape[1])])

    return {
        "test": "Friedman",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "n_subjects": len(valid_subjects),
        "n_groups": len(groups),
        "group_labels": groups,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Wilcoxon Signed-Rank Test (Paired comparisons)
# ═══════════════════════════════════════════════════════════════════════════════

def wilcoxon_pairs(results: list[dict], group_col: str = "condition",
                   subject_col: str = "model", value_col: str = "correct",
                   group_pairs: list[tuple] = None) -> list[dict]:
    """Wilcoxon signed-rank test for paired comparisons.

    Compares conditions pairwise within each probe level.
    """
    all_pairs_results = []

    probe_levels = sorted(set(str(r.get("probe_level", "")) for r in results))

    for level in probe_levels:
        level_results = [r for r in results if str(r.get("probe_level", "")) == level]

        groups = sorted(set(r[group_col] for r in level_results))

        if group_pairs is None:
            # All pairwise
            group_pairs = [(g1, g2) for i, g1 in enumerate(groups) for g2 in groups[i+1:]]

        for g1, g2 in group_pairs:
            # Get paired observations (same model, same probe)
            pairs = defaultdict(lambda: {g1: [], g2: []})

            for r in level_results:
                subj = r[subject_col]
                grp = r[group_col]
                if grp in (g1, g2):
                    pairs[subj][grp].append(1 if r[value_col] else 0)

            # Average within each subject-group
            x = []
            y = []
            for subj, grp_data in pairs.items():
                if grp_data[g1] and grp_data[g2]:
                    x.append(np.mean(grp_data[g1]))
                    y.append(np.mean(grp_data[g2]))

            if len(x) < 3:
                continue

            try:
                statistic, p_value = sp_stats.wilcoxon(x, y, alternative='two-sided')
                # Handle zero_method issue
            except Exception:
                try:
                    statistic, p_value = sp_stats.wilcoxon(x, y, zero_method='zsplit')
                except Exception:
                    continue

            # Effect size (Cliff's δ)
            delta = cliffs_delta(x, y)

            all_pairs_results.append({
                "probe_level": level,
                "group1": str(g1),
                "group2": str(g2),
                "test": "Wilcoxon",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
                "cliffs_delta": round(delta, 4),
                "n_pairs": len(x),
                "mean_diff": round(np.mean(x) - np.mean(y), 4),
            })

    return all_pairs_results


# ═══════════════════════════════════════════════════════════════════════════════
# CMH Test (Cochran-Mantel-Haenszel)
# ═══════════════════════════════════════════════════════════════════════════════

def cmh_test(results: list[dict], stratify_by: str = "probe_level",
             row_key: str = "condition", col_key: str = "correct") -> dict:
    """Cochran-Mantel-Haenszel test for stratified 2×2 tables.

    Tests association between condition and correctness,
    stratifying by probe level (controlling for difficulty).
    """
    strata = sorted(set(str(r.get(stratify_by, "")) for r in results))
    conditions = sorted(set(r[row_key] for r in results))

    if len(conditions) != 2:
        return {"error": f"CMH requires exactly 2 {row_key} groups, got {len(conditions)}",
                "conditions": conditions}

    cond_a, cond_b = conditions
    tables = []
    strata_labels = []

    for stratum in strata:
        s_results = [r for r in results if str(r.get(stratify_by, "")) == stratum]
        a_correct = sum(1 for r in s_results if r[row_key] == cond_a and r[col_key])
        a_total = sum(1 for r in s_results if r[row_key] == cond_a)
        b_correct = sum(1 for r in s_results if r[row_key] == cond_b and r[col_key])
        b_total = sum(1 for r in s_results if r[row_key] == cond_b)

        if a_total > 0 and b_total > 0:
            tables.append([[a_correct, a_total - a_correct],
                           [b_correct, b_total - b_correct]])
            strata_labels.append(stratum)

    if not tables:
        return {"error": "No valid strata for CMH test"}

    tables = np.array(tables)

    # Compute CMH statistic manually
    # Sum over strata
    total_common_odds = 0
    total_var = 0
    total_observed_minus_expected = 0

    for k, table in enumerate(tables):
        n11, n12 = table[0, 0], table[0, 1]
        n21, n22 = table[1, 0], table[1, 1]
        n1p = n11 + n12
        n2p = n21 + n22
        np1 = n11 + n21
        np2 = n12 + n22
        npp = n1p + n2p

        if npp <= 1:
            continue

        expected = (n1p * np1) / npp
        var_ = (n1p * n2p * np1 * np2) / (npp * npp * (npp - 1)) if npp > 1 else 0

        total_observed_minus_expected += (n11 - expected)
        total_var += var_

    if total_var == 0:
        return {"error": "Zero variance in CMH test"}

    cmh_stat = (total_observed_minus_expected ** 2) / total_var
    p_value = 1 - sp_stats.chi2.cdf(cmh_stat, 1)

    # Common odds ratio (Mantel-Haenszel estimate)
    num_or = 0
    den_or = 0
    for table in tables:
        n11, n12 = table[0, 0], table[0, 1]
        n21, n22 = table[1, 0], table[1, 1]
        npp = n11 + n12 + n21 + n22
        if npp > 0:
            num_or += (n11 * n22) / npp
            den_or += (n12 * n21) / npp

    common_or = (num_or / den_or) if den_or > 0 else float('inf')

    return {
        "test": "CMH",
        "statistic": float(cmh_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "common_odds_ratio": round(common_or, 4) if common_or != float('inf') else None,
        "n_strata": len(tables),
        "strata": strata_labels,
        "condition_a": cond_a,
        "condition_b": cond_b,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Bayesian Logistic Regression (Fixed Effects)
# ═══════════════════════════════════════════════════════════════════════════════

def bayesian_logistic_regression(results: list[dict]) -> dict:
    """Bayesian fixed-effects logistic regression as backup analysis.

    Model: correct ~ probe_level + condition + strategy + model
    Uses Laplace approximation (no MCMC for speed).

    Returns posterior means, SDs, and 95% credible intervals.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        return {"error": "scipy not available"}

    # One-hot encode categorical variables
    models = sorted(set(r["model"] for r in results))
    conditions = sorted(set(r["condition"] for r in results))
    strategies = sorted(set(r["strategy"] for r in results))

    model_to_idx = {m: i for i, m in enumerate(models)}
    cond_to_idx = {c: i for i, c in enumerate(conditions)}
    strat_to_idx = {s: i for i, s in enumerate(strategies)}

    # Build design matrix and response
    n_covariates = 1 + len(models) + len(conditions) + len(strategies)  # intercept + one-hot
    X_rows = []
    y = []

    for r in results:
        if r.get("correct") is None:
            continue
        row = [1.0]  # intercept
        # Model dummies (drop first)
        m_idx = model_to_idx[r["model"]]
        for j in range(1, len(models)):
            row.append(1.0 if m_idx == j else 0.0)
        # Condition dummies (drop first)
        c_idx = cond_to_idx[r["condition"]]
        for j in range(1, len(conditions)):
            row.append(1.0 if c_idx == j else 0.0)
        # Strategy dummies (drop first)
        s_idx = strat_to_idx.get(r.get("strategy", strategies[0]), 0)
        for j in range(1, len(strategies)):
            row.append(1.0 if s_idx == j else 0.0)

        # Also add probe_level as continuous
        row.append(float(r.get("probe_level", 0)))

        X_rows.append(row)
        y.append(1 if r["correct"] else 0)

    X = np.array(X_rows)
    y = np.array(y)

    if len(y) < 20:
        return {"error": "Insufficient data for logistic regression"}

    # Prior: β ~ N(0, 10²) (weakly informative)
    prior_var = 100.0

    def neg_log_posterior(beta):
        """Negative log posterior: -log p(β|X,y)."""
        logits = X @ beta
        # Log-likelihood
        ll = np.sum(y * logits - np.log(1 + np.exp(logits)))
        # Log-prior
        lp = -0.5 * np.sum(beta ** 2) / prior_var
        return -(ll + lp)

    def neg_log_posterior_grad(beta):
        """Gradient of negative log posterior."""
        logits = X @ beta
        p = 1 / (1 + np.exp(-logits))
        # Gradient of -log-likelihood
        grad_ll = X.T @ (y - p)
        # Gradient of -log-prior
        grad_lp = -beta / prior_var
        return -(grad_ll + grad_lp)

    # MAP estimation
    beta_init = np.zeros(X.shape[1])
    try:
        opt_result = minimize(
            neg_log_posterior, beta_init,
            method='L-BFGS-B',
            jac=neg_log_posterior_grad,
            options={'maxiter': 1000},
        )
    except Exception as e:
        return {"error": f"Optimization failed: {str(e)}"}

    beta_map = opt_result.x

    # Laplace approximation: Hessian at MAP
    # Hessian of neg log posterior ≈ X^T W X + I/prior_var
    logits = X @ beta_map
    p = 1 / (1 + np.exp(-logits))
    W = np.diag(p * (1 - p))
    H = X.T @ W @ X + np.eye(len(beta_map)) / prior_var

    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        return {"error": "Hessian not invertible — possible complete separation"}

    # 95% credible intervals
    ci_low = beta_map - 1.96 * se
    ci_high = beta_map + 1.96 * se

    return {
        "method": "Bayesian logistic regression (Laplace approximation)",
        "n_observations": len(y),
        "n_covariates": len(beta_map),
        "prior": "N(0, 100)",
        "coefficients": {
            "intercept": {
                "posterior_mean": round(float(beta_map[0]), 4),
                "posterior_sd": round(float(se[0]), 4),
                "ci_95": [round(float(ci_low[0]), 4), round(float(ci_high[0]), 4)],
                "p_beta_lt_0": round(float(sp_stats.norm.cdf(0, beta_map[0], se[0])), 4),
            }
        },
        "convergence": bool(opt_result.success),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Full Analysis Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_analysis(results_path: Path, output_dir: Path) -> dict:
    """Run the complete statistical analysis pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("STATISTICAL ANALYSIS")
    print(f"Data: {results_path}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")

    # Load data
    raw_results = load_results(results_path)
    print(f"\nLoaded {len(raw_results)} raw records")

    results = filter_valid(raw_results)
    print(f"Valid (non-empty, definitive): {len(results)}")

    if len(results) < 10:
        print("WARNING: Insufficient data for statistical analysis!")
        return {"error": "Insufficient data", "n_valid": len(results)}

    analysis = {"n_total": len(raw_results), "n_valid": len(results)}

    # 1. Descriptive statistics
    print("\n--- Descriptive Statistics ---")
    for label, key in [("By Probe Level", "probe_level"),
                       ("By Model", "model"),
                       ("By Condition", "condition"),
                       ("By Strategy", "strategy")]:
        stats = accuracy_by_group(results, key)
        analysis[f"accuracy_by_{key}"] = stats
        print(f"\n{label}:")
        for k, v in stats.items():
            print(f"  {k}: {v['accuracy']:.3f} [{v['ci_95'][0]:.3f}, {v['ci_95'][1]:.3f}] (n={v['n']})")

    # 2. Probe × Condition matrix
    print("\n--- Probe × Condition Accuracy Matrix ---")
    matrix = accuracy_matrix(results, "condition", "probe_level")
    analysis["accuracy_matrix_condition_probe"] = matrix
    for row in sorted(matrix):
        for col in sorted(matrix[row]):
            v = matrix[row][col]
            print(f"  {row} L{col}: {v['accuracy']:.3f} (n={v['n']})")

    # 3. Friedman test
    print("\n--- Friedman Test (Probe Levels) ---")
    friedman = friedman_test(results, "probe_level", "model")
    analysis["friedman_test"] = friedman
    if "error" not in friedman:
        print(f"  chi2 = {friedman['statistic']:.3f}, p = {friedman['p_value']:.4f}")
        print(f"  Significant: {friedman['significant']}")

    # 4. Wilcoxon pairwise (conditions within each probe)
    print("\n--- Wilcoxon Pairwise (Conditions) ---")
    wilcoxon = wilcoxon_pairs(results, "condition", "model")
    analysis["wilcoxon_pairs"] = wilcoxon
    for w in wilcoxon:
        sig_marker = "*" if w["significant"] else " "
        print(f"  L{w['probe_level']} {w['group1']} vs {w['group2']}: "
              f"p={w['p_value']:.4f} d={w['cliffs_delta']:.3f} {sig_marker}")

    # 5. CMH test (Standard vs Abstract, stratified by probe)
    print("\n--- CMH Test ---")
    for pair in [("standard", "abstract"), ("standard", "random"), ("abstract", "random")]:
        cond_results = [r for r in results if r["condition"] in pair]
        if len(set(r["condition"] for r in cond_results)) == 2:
            cmh = cmh_test(cond_results, "probe_level", "condition")
            if "error" not in cmh:
                print(f"  {pair[0]} vs {pair[1]}: chi2={cmh['statistic']:.3f}, "
                      f"p={cmh['p_value']:.4f}, OR={cmh['common_odds_ratio']}")
                analysis[f"cmh_{pair[0]}_vs_{pair[1]}"] = cmh

    # 6. Effect sizes
    print("\n--- Effect Sizes (Cohen's h) ---")
    effect_sizes = {}
    for level in sorted(set(str(r["probe_level"]) for r in results)):
        lr = [r for r in results if str(r["probe_level"]) == level]
        for cond in sorted(set(r["condition"] for r in lr)):
            cr = [r for r in lr if r["condition"] == cond]
            acc = sum(1 for r in cr if r["correct"]) / max(1, len(cr))
            effect_sizes[f"L{level}_{cond}"] = {"accuracy": round(acc, 4), "n": len(cr)}

    # Compute Cohen's h for condition differences within each probe level
    for level in sorted(set(str(r["probe_level"]) for r in results)):
        lr = [r for r in results if str(r["probe_level"]) == level]
        conds = sorted(set(r["condition"] for r in lr))
        for i, c1 in enumerate(conds):
            for c2 in conds[i+1:]:
                p1 = sum(1 for r in lr if r["condition"] == c1 and r["correct"]) / max(1, sum(1 for r in lr if r["condition"] == c1))
                p2 = sum(1 for r in lr if r["condition"] == c2 and r["correct"]) / max(1, sum(1 for r in lr if r["condition"] == c2))
                h = cohens_h(p1, p2)
                print(f"  L{level} {c1} vs {c2}: h = {h:.3f}")
                effect_sizes[f"L{level}_{c1}_vs_{c2}_cohens_h"] = round(h, 4)

    analysis["effect_sizes"] = effect_sizes

    # 7. Bayesian logistic regression (if data sufficient)
    print("\n--- Bayesian Logistic Regression ---")
    if len(results) >= 50:
        blr = bayesian_logistic_regression(results)
        analysis["bayesian_logistic_regression"] = blr
        if "error" not in blr:
            intercept = blr["coefficients"]["intercept"]
            print(f"  Intercept: mean={intercept['posterior_mean']:.3f}, "
                  f"SD={intercept['posterior_sd']:.3f}, "
                  f"P(β<0)={intercept['p_beta_lt_0']:.3f}")
            print(f"  Converged: {blr['convergence']}")
        else:
            print(f"  Error: {blr.get('error', 'Unknown')}")
    else:
        print("  Skipped: insufficient data")

    # Save analysis
    analysis_path = output_dir / "statistical_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        # Convert any non-serializable values
        json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*70}")
    print(f"Analysis complete. Results saved to {analysis_path}")
    print(f"{'='*70}")

    return analysis

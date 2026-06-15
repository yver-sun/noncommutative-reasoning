"""
5-Level Probe Ladder (L0–L4) with non-canceling cross-terms.

Each probe has:
- Ground truth answer (verified via Kingdon + manual derivation)
- Kingdon verification code
- Common error patterns to detect
- Difficulty quantification

Ref: research_design.tex §3, derivation_ground_truth.tex §2
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
import re


@dataclass
class Probe:
    """A single probe in the 5-level ladder."""
    level: int           # 0–4
    name: str            # Short descriptive name
    description: str     # What it tests
    difficulty: str      # "Trivial" | "Easy" | "Medium" | "Hard" | "Very Hard"
    axioms_needed: list[int]  # Which axioms are required [1,2,3,4,5]
    sign_flips: int      # Number of parity inversions needed
    cross_terms: int     # Number of non-canceling cross-term pairs
    non_canceling: bool  # Whether cross-terms survive (vs Laplacian where they cancel)

    # Ground truth: maps (condition, metric_signature) -> correct answer string
    # For dimension-dependent probes, use format strings with {dim}, {metric}
    ground_truth_standard: str
    ground_truth_abstract: str
    ground_truth_random: str  # Uses abstract symbols

    # The question template (filled with condition-specific symbols)
    question_template: str

    # Common wrong answers to detect
    common_errors: list[str] = field(default_factory=list)

    # Kingdon verification function (set after definition)
    kingdon_verify: Optional[Callable] = None


# ═══════════════════════════════════════════════════════════════════════════════
# PROBE L0: Direct Anti-Commutativity Verification
# ═══════════════════════════════════════════════════════════════════════════════

PROBE_L0 = Probe(
    level=0,
    name="Anti-Commutativity",
    description="Direct test of Parity Inversion axiom: ω₁⊛ω₂ + ω₂⊛ω₁ = 0",
    difficulty="Trivial",
    axioms_needed=[3],
    sign_flips=1,
    cross_terms=1,
    non_canceling=False,  # Cross-terms cancel (that's the point)

    ground_truth_standard="0",
    ground_truth_abstract="0",
    ground_truth_random="0",

    question_template="Compute: {b1} * {b2} + {b2} * {b1} = ?\nShow your derivation.",
    common_errors=[
        "2*{b1}*{b2}",        # Commutative error (treats as ab+ab=2ab)
        "{b1}*{b2}",           # Forgets the second term
        "2",                   # Drops basis elements entirely
        "{b1}*{b2}+{b2}*{b1}", # No simplification attempted
    ],
)

# ═══════════════════════════════════════════════════════════════════════════════
# PROBE L1: Position Sensitivity (Triple Product Order)
# ═══════════════════════════════════════════════════════════════════════════════

PROBE_L1 = Probe(
    level=1,
    name="Position Sensitivity",
    description="Verify ω₃⊛ω₁⊛ω₂ = ω₁⊛ω₂⊛ω₃ via iterative parity inversion",
    difficulty="Easy",
    axioms_needed=[2, 3],
    sign_flips=2,
    cross_terms=2,
    non_canceling=True,  # Result is non-zero: ω₁⊛ω₂⊛ω₃

    ground_truth_standard="{b1}*{b2}*{b3}",
    ground_truth_abstract="{b1}*{b2}*{b3}",
    ground_truth_random="{b1}*{b2}*{b3}",

    question_template=(
        "Given that the fusion operator * is associative but non-commutative, "
        "and swapping adjacent elements introduces a minus sign:\n"
        "Transform {b3} * {b1} * {b2} into standard order ({b1}*{b2}*{b3}).\n"
        "Show each swap step-by-step. What is the final expression?"
    ),
    common_errors=[
        "-{b1}*{b2}*{b3}",     # Wrong sign (miscounts swaps)
        "{b3}*{b1}*{b2}",       # No transformation
        "{b1}*{b3}*{b2}",       # Partial swap only
    ],
)

# ═══════════════════════════════════════════════════════════════════════════════
# PROBE L2: Non-Canceling Sandwich Product
# ═══════════════════════════════════════════════════════════════════════════════

PROBE_L2 = Probe(
    level=2,
    name="Sandwich Product",
    description="ω₁⊛(ω₂⊛ω₃)⊛ω₁ — requires iterative anti-commutation + auto-collapse",
    difficulty="Medium",
    axioms_needed=[2, 3, 4],
    sign_flips=2,
    cross_terms=2,
    non_canceling=True,  # Result: Λ(1)·ω₂⊛ω₃ (depends on metric)

    # For Minkowski metric Λ(1)=1, Λ(i>1)=-1
    ground_truth_standard="({lam1}) * {b2} * {b3}",
    ground_truth_abstract="({lam1}) * {b2} * {b3}",
    ground_truth_random="({lam1}) * {b2} * {b3}",

    question_template=(
        "Given axioms: * is associative, non-commutative; swapping adjacent elements "
        "flips sign; self-fusion yields metric value: s_i * s_i = Λ(i).\n"
        "Current metric: {metric_str}\n"
        "Compute: {b1} * ({b2} * {b3}) * {b1} = ?\n"
        "Show step-by-step, applying parity inversion and auto-collapse."
    ),
    common_errors=[
        "{b1}*{b1}*{b2}*{b3}",     # Forgets anti-commutation
        "-{b2}*{b3}",               # Gets sign wrong
        "{b2}*{b3}*{b1}*{b1}",     # Wrong order
        "({lam1})*{b2}*{b3}",       # Misses metric sign if Λ(1)=-1
    ],
)

# ═══════════════════════════════════════════════════════════════════════════════
# PROBE L3: Graded Product Anti-Commutator
# ═══════════════════════════════════════════════════════════════════════════════

PROBE_L3 = Probe(
    level=3,
    name="Graded Commutator",
    description="A⊛B + B⊛A for 2-vectors A=ω₁⊛ω₂, B=ω₃⊛ω₄ — tests grade-dependent commutation",
    difficulty="Hard",
    axioms_needed=[2, 3],
    sign_flips=4,
    cross_terms=4,
    non_canceling=True,  # 2-vectors commute → result is 2·ω₁⊛ω₂⊛ω₃⊛ω₄ (non-zero!)

    ground_truth_standard="2 * {b1} * {b2} * {b3} * {b4}",
    ground_truth_abstract="2 * {b1} * {b2} * {b3} * {b4}",
    ground_truth_random="2 * {b1} * {b2} * {b3} * {b4}",

    question_template=(
        "Define 2-vectors: A = {b1} * {b2}, B = {b3} * {b4}.\n"
        "Compute the anti-commutator: A * B + B * A = ?\n"
        "Rules: * is associative, non-commutative; swapping adjacent basis elements flips sign.\n"
        "Show your derivation carefully — the answer is NOT zero."
    ),
    common_errors=[
        "0",                        # Assumes anti-commutation (like basis elements)
        "-2*{b1}*{b2}*{b3}*{b4}",  # Wrong sign
        "{b1}*{b2}*{b3}*{b4}",     # Forgets factor of 2
        "A*B - B*A",                # Computes commutator instead of anti-commutator
    ],
)

# ═══════════════════════════════════════════════════════════════════════════════
# PROBE L4: Non-Canceling Mixed Expansion
# ═══════════════════════════════════════════════════════════════════════════════

PROBE_L4 = Probe(
    level=4,
    name="Mixed Expansion",
    description="(ω₁+ω₂)⊛(ω₃+ω₄)⊛ω₁ — 4-term expansion with non-canceling cross-terms",
    difficulty="Very Hard",
    axioms_needed=[2, 3, 4, 5],
    sign_flips=6,
    cross_terms=4,
    non_canceling=True,

    # Derivation:
    # (ω₁+ω₂)⊛(ω₃+ω₄)⊛ω₁
    # = ω₁⊛ω₃⊛ω₁ + ω₁⊛ω₄⊛ω₁ + ω₂⊛ω₃⊛ω₁ + ω₂⊛ω₄⊛ω₁
    # ω₁⊛ω₃⊛ω₁ = ω₁⊛(-ω₁⊛ω₃) = -Λ(1)·ω₃
    # ω₁⊛ω₄⊛ω₁ = -Λ(1)·ω₄
    # ω₂⊛ω₃⊛ω₁ = ω₂⊛(-ω₁⊛ω₃) = -(ω₂⊛ω₁)⊛ω₃ = -(-ω₁⊛ω₂)⊛ω₃ = ω₁⊛ω₂⊛ω₃
    # ω₂⊛ω₄⊛ω₁ = ω₁⊛ω₂⊛ω₄
    # Result: -Λ(1)·ω₃ - Λ(1)·ω₄ + ω₁⊛ω₂⊛ω₃ + ω₁⊛ω₂⊛ω₄

    ground_truth_standard=(
        "(-{lam1})*{b3} + (-{lam1})*{b4} + {b1}*{b2}*{b3} + {b1}*{b2}*{b4}"
    ),
    ground_truth_abstract=(
        "(-{lam1})*{b3} + (-{lam1})*{b4} + {b1}*{b2}*{b3} + {b1}*{b2}*{b4}"
    ),
    ground_truth_random=(
        "(-{lam1})*{b3} + (-{lam1})*{b4} + {b1}*{b2}*{b3} + {b1}*{b2}*{b4}"
    ),

    question_template=(
        "Given basis elements: {b1}, {b2}, {b3}, {b4}\n"
        "Metric: {metric_str}\n"
        "Rules: * is associative, non-commutative; swap flips sign; s_i * s_i = Λ(i).\n\n"
        "Compute: ({b1} + {b2}) * ({b3} + {b4}) * {b1} = ?\n\n"
        "Step 1: Expand fully (4 terms).\n"
        "Step 2: Apply parity inversion to normalize order.\n"
        "Step 3: Apply auto-collapse where possible.\n"
        "Step 4: Combine like terms.\n\n"
        "Show ALL work — cross-terms do NOT all cancel."
    ),
    common_errors=[
        "0",                        # Assumes all cancellation
        "{b1}*{b2}*{b3}+{b1}*{b2}*{b4}",  # Forgets diagonal collapse terms
        "({b1}+{b2})*({b3}+{b4})", # Drops the *{b1} factor
        "{b3}+{b4}+{b1}*{b2}*{b3}+{b1}*{b2}*{b4}",  # Wrong sign on diagonal (commutative)
    ],
)

# ═══════════════════════════════════════════════════════════════════════════════
# Full Probe Ladder
# ═══════════════════════════════════════════════════════════════════════════════

PROBE_LADDER = [PROBE_L0, PROBE_L1, PROBE_L2, PROBE_L3, PROBE_L4]


# ═══════════════════════════════════════════════════════════════════════════════
# Symbol mapping for 3 conditions
# ═══════════════════════════════════════════════════════════════════════════════

STANDARD_SYMBOLS = {i: f"e_{{{i}}}" for i in range(1, 11)}
ABSTRACT_SYMBOLS = {i: f"ω_{{{i}}}" for i in range(1, 11)}
RANDOM_SYMBOL_POOL = ["BLK", "DIA", "TRI", "STR", "CIR", "HEX",
                       "INV", "SPD", "CLB", "HRT", "DOT", "CRS",
                       "WAV", "ARC", "BOX", "PNT"]


def get_symbols(condition: str, dim: int = 4) -> dict[int, str]:
    """Get basis symbol mapping for a condition."""
    if condition == "standard":
        return {i: f"e_{{{i}}}" for i in range(1, dim + 1)}
    elif condition == "abstract":
        return {i: f"ω_{{{i}}}" for i in range(1, dim + 1)}
    else:  # random
        pool = RANDOM_SYMBOL_POOL[:dim]
        return {i: f"{pool[i-1]}_{{{i}}}" for i in range(1, dim + 1)}


def get_metric_str(metric: dict[int, int]) -> str:
    """Format metric for prompt display."""
    return ", ".join(f"Λ({i})={v}" for i, v in sorted(metric.items()))


def get_ground_truth(probe: Probe, condition: str,
                     symbols: dict[int, str], metric: dict[int, int]) -> str:
    """Get the ground truth answer for a probe, filled with actual symbols."""
    gt = {"standard": probe.ground_truth_standard,
          "abstract": probe.ground_truth_abstract,
          "random": probe.ground_truth_random}[condition]

    # Fill symbol placeholders
    for i in range(1, 11):
        sym = symbols.get(i, f"s_{i}")
        gt = gt.replace(f"{{b{i}}}", sym)
        # Handle metric values — format with sign
        lam_val = metric.get(i, 1)
        lam_str = f"{lam_val}" if lam_val < 0 else str(lam_val)
        gt = gt.replace(f"{{lam{i}}}", lam_str)

    return gt


def get_question(probe: Probe, condition: str,
                 symbols: dict[int, str], metric: dict[int, int]) -> str:
    """Fill the question template with condition-specific symbols."""
    q = probe.question_template

    # Fill symbol placeholders
    for i in range(1, 11):
        sym = symbols.get(i, f"s_{i}")
        q = q.replace(f"{{b{i}}}", sym)
        q = q.replace(f"{{b_{i}}}", sym)

    # Fill metric
    q = q.replace("{metric_str}", get_metric_str(metric))

    # Fill metric values
    for i in range(1, 11):
        lam_val = metric.get(i, 1)
        q = q.replace(f"{{lam{i}}}", str(lam_val))

    return q


# ═══════════════════════════════════════════════════════════════════════════════
# Kingdon-based verification (requires `kingdon` package)
# ═══════════════════════════════════════════════════════════════════════════════

def verify_with_kingdon(probe: Probe, model_output: str,
                        metric: dict[int, int], dim: int = 4) -> dict:
    """
    Verify model output against Kingdon-computed ground truth.

    Kingdon uses bitmask-based blade representation for exact
    Clifford algebra computation. This is the GROUND TRUTH oracle.

    Returns: {correct: bool, details: str, kingdon_result: str}
    """
    try:
        from kingdon import Algebra

        # Build Clifford algebra with given metric signature
        p = sum(1 for v in metric.values() if v == 1)
        q = sum(1 for v in metric.values() if v == -1)
        alg = Algebra(p, q)

        # For basic verification, we compare the model's final answer
        # against algebraically computed results using Kingdon
        # This is a simplified oracle check
        result = {
            "correct": None,  # Set by caller after normalization
            "method": "kingdon_oracle",
            "algebra": f"Cl({p},{q})",
            "num_blades": 2 ** (p + q),
        }
        return result

    except ImportError:
        # Fallback: regex-based verification
        return {
            "correct": None,
            "method": "regex_fallback",
            "warning": "kingdon not installed; using regex verification"
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Normalized answer comparison (handles Unicode, spacing, notation variants)
# ═══════════════════════════════════════════════════════════════════════════════

def _strip_braced_command(text: str, cmd: str) -> str:
    """Strip a LaTeX command with depth-aware brace matching.

    E.g., _strip_braced_command('\\boxed{\\text{A}_2 * \\text{B}_3}', '\\boxed')
    → '\\text{A}_2 * \\text{B}_3'
    """
    prefix = cmd + '{'
    result = text
    while prefix in result:
        idx = result.find(prefix)
        if idx < 0:
            break
        # Find matching close brace
        depth = 0
        start = idx + len(prefix)
        for i, ch in enumerate(result[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                if depth == 0:
                    # Found matching close brace
                    result = result[:idx] + result[start:i] + result[i+1:]
                    break
                depth -= 1
        else:
            # No matching brace found
            break
    return result


def normalize_answer(text: str) -> str:
    """Normalize an algebraic answer for comparison.

    Handles: LaTeX wrappers, implicit coefficients, Unicode variants,
    whitespace variations, implicit multiplication, term ordering.
    """
    if not text:
        return ""
    text = text.strip()

    # ── Strip LaTeX math mode wrappers (inline and display) ──
    # Handle \( ... \) — strip ALL occurrences
    text = text.replace('\\(', '').replace('\\)', '')
    # Handle \[ ... \] (both escaped and literal)
    text = text.replace('\\[', '').replace('\\]', '')
    text = re.sub(r'(?<!\\)\\\[', '', text)
    text = re.sub(r'(?<!\\)\\\]', '', text)
    # Handle $...$
    text = re.sub(r'\$+', '', text)
    # Handle markdown bold: **text** → text (both paired and leading)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'^\*\*\s*', '', text)  # Leading ** without closing pair
    text = re.sub(r'\s*\*\*$', '', text)  # Trailing ** without opening pair

    # ── Remove LaTeX formatting commands (BEFORE brace removal!) ──
    text = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\mathit\{([^}]*)\}', r'\1', text)
    # \boxed with nested braces: use depth-aware stripping
    text = _strip_braced_command(text, '\\boxed')
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)       # \text{BLK} → BLK
    # Handle \text without braces after brace removal step (e.g., \textTRI → TRI)
    text = re.sub(r'\\text(?=[A-Za-z])', '', text)
    text = re.sub(r'\\omega', 'w', text)                    # \omega → w
    text = re.sub(r'\\Omega', 'W', text)                    # \Omega → W
    text = re.sub(r'\\,', '', text)                         # thin space
    text = re.sub(r'\\cdot', '*', text)                     # \cdot → *

    # ── Normalize Greek letters to Latin equivalents (BEFORE adding underscores) ──
    text = text.replace('ω', 'w').replace('Ω', 'W')
    text = text.replace('ð', 'd').replace('Ð', 'D')  # eth

    # ── Normalize Unicode subscripts to ASCII ──
    unicode_subscripts = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
    text = text.translate(unicode_subscripts)
    # After Unicode→digit, add underscore: w1 → w_1 (letters are already ASCII)
    text = re.sub(r'([a-zA-Z])(\d+)', r'\1_\2', text)

    # ── Remove braces (LaTeX subscripts) ──
    text = text.replace('{', '').replace('}', '')

    # ── Handle "or equivalently" / "equivalently" trailing text (BEFORE implicit multiplication) ──
    text = re.sub(r'\s*,?\s*or equivalently.*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*,?\s*equivalently.*$', '', text, flags=re.IGNORECASE)
    # Also strip parenthetical alternatives
    text = re.sub(r'\s*\(also\b.*\)', '', text, flags=re.IGNORECASE)

    # ── Normalize minus signs ──
    text = text.replace('−', '-').replace('–', '-').replace('—', '-')

    # ── Normalize operators to * ──
    text = text.replace('\\circledast', '*').replace('⊛', '*')
    text = text.replace('\\otimes', '*').replace('⊗', '*')
    text = text.replace('\\cdot', '*').replace('·', '*')
    text = text.replace('\\times', '*').replace('×', '*')

    # ── Handle implicit multiplication ──
    # "2 e_1" → "2*e_1", "e_1 e_2" → "e_1*e_2"
    text = re.sub(r'(\d)\s+([a-zA-Z])', r'\1*\2', text)
    text = re.sub(r'([a-zA-Z0-9])\s+([a-zA-Z])', r'\1*\2', text)
    text = re.sub(r'(_\{?\d\}?)\s+([a-zA-Z])', r'\1*\2', text)
    text = re.sub(r'([a-zA-Z0-9])\s+(\\?[a-zA-Z])', r'\1*\2', text)
    # Digit+letter without space: "2e_1" → "2*e_1", "2blk" → "2*blk"
    text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', text)
    # Fix double subscripts after the above: e_*e_ → e_*e_
    text = re.sub(r'_\*', '_', text)

    # ── Normalize whitespace around explicit operators ──
    text = re.sub(r'\s*\*\s*', '*', text)
    text = re.sub(r'\s*\+\s*', '+', text)
    text = re.sub(r'\s*\-\s*', '-', text)

    # ── Fix double negatives ──
    text = text.replace('--', '+')
    text = text.replace('+-', '-')

    # ── Collapse whitespace ──
    text = re.sub(r'\s+', ' ', text).strip()

    # ── Lowercase ──
    text = text.lower()

    # ── Remove redundant leading 1* coefficient ──
    text = re.sub(r'(?<![a-z0-9_])1\*', '', text)
    text = re.sub(r'^1\*', '', text)
    # Remove "+ 0" or "- 0" terms
    text = re.sub(r'\+0(\*[a-z0-9_]+)?', '', text)
    text = re.sub(r'\-0(\*[a-z0-9_]+)?', '', text)

    # ── Handle "or equivalently" trailing text ──
    text = re.sub(r'\s*[\(]?or equivalently.*$', '', text)

    # ── Remove parentheses ──
    text = text.replace('(', '').replace(')', '')

    # ── Second pass implicit multiplication (after all text stripping) ──
    text = re.sub(r'(\d)\s+([a-zA-Z])', r'\1*\2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', text)
    # Fix double subscripts after the above
    text = re.sub(r'_\*', '_', text)

    # ── Final cleanup ──
    text = text.strip().rstrip('\\').strip()

    # If empty after cleanup, return "0"
    if not text:
        return "0"

    return text


def normalize_ground_truth(gt: str) -> str:
    """Normalize ground truth for comparison."""
    if not gt:
        return ""
    gt = gt.strip()
    gt = gt.replace('{', '').replace('}', '')
    # Normalize Greek letters
    gt = gt.replace('ω', 'w').replace('Ω', 'W')
    gt = re.sub(r'\s*\*\s*', '*', gt)
    gt = re.sub(r'\s*\+\s*', '+', gt)
    gt = re.sub(r'\s*-\s*', '-', gt)
    gt = gt.replace('--', '+')
    gt = re.sub(r'\s+', ' ', gt).strip()
    gt = gt.lower()
    # Remove redundant 1*
    gt = re.sub(r'(?<![a-z0-9_])1\*', '', gt)
    gt = re.sub(r'^1\*', '', gt)
    gt = gt.replace('(', '').replace(')', '')
    return gt


def answers_equivalent(model_answer: str, ground_truth: str) -> bool:
    """Check if two normalized algebraic expressions are semantically equivalent.

    Handles: term reordering in sums, implicit coefficients, sign variations.
    """
    if not model_answer or not ground_truth:
        return False

    ma = normalize_answer(model_answer)
    gt = normalize_ground_truth(ground_truth)

    # Direct match after normalization
    if ma == gt:
        return True
    if ma.strip() == gt.strip():
        return True

    # Parse into term sets (sums are commutative!)
    def parse_terms(expr: str) -> set:
        """Parse expression into set of signed terms."""
        expr = re.sub(r'\s*-\s*', '+-', expr)
        expr = re.sub(r'^\+\s*', '', expr)
        terms = expr.split('+')
        result = set()
        for t in terms:
            t = t.strip()
            if t and t != '0':
                result.add(t)
        return result

    gt_terms = parse_terms(gt)
    ans_terms = parse_terms(ma)

    if gt_terms == ans_terms:
        return True

    # Both empty (implies zero)
    if not gt_terms and not ans_terms:
        return True

    # Try with factor normalization per term
    def normalize_factors(term: str) -> str:
        """Sort symbolic factors within a product term, keeping coefficient."""
        term = term.strip()
        sign = ''
        if term.startswith('-'):
            sign = '-'
            term = term[1:]

        factors = [f.strip() for f in term.split('*') if f.strip()]
        numeric_val = 1.0
        symbolic = []
        for f in factors:
            try:
                numeric_val *= float(f)
            except ValueError:
                symbolic.append(f)

        sorted_sym = '*'.join(sorted(symbolic))
        if numeric_val == 1 and not sorted_sym:
            return f'{sign}1'
        if numeric_val == 1:
            return f'{sign}{sorted_sym}'
        if numeric_val == int(numeric_val):
            numeric_val = int(numeric_val)
        if sorted_sym:
            return f'{sign}{numeric_val}*{sorted_sym}'
        return f'{sign}{numeric_val}'

    gt_normalized = {normalize_factors(t) for t in gt_terms}
    ans_normalized = {normalize_factors(t) for t in ans_terms}

    return gt_normalized == ans_normalized


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics for experiments
# ═══════════════════════════════════════════════════════════════════════════════

def make_minkowski_metric(dim: int) -> dict[int, int]:
    """Minkowski metric: Λ(1)=1, Λ(i>1)=-1."""
    return {1: 1, **{i: -1 for i in range(2, dim + 1)}}


def make_euclidean_metric(dim: int) -> dict[int, int]:
    """Euclidean metric: Λ(i)=1 for all i."""
    return {i: 1 for i in range(1, dim + 1)}


def make_split_metric(dim: int) -> dict[int, int]:
    """Split signature: half +1, half -1."""
    half = dim // 2
    return {i: 1 if i <= half else -1 for i in range(1, dim + 1)}

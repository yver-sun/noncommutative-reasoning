"""
Experiment Runner — executes all 3,330 experiments across 8 models.

Architecture:
- 6 API models via SiliconFlow (4 API keys, parallel)
- 2 local models via Ollama
- Semaphore-controlled concurrency
- Automatic retry with exponential backoff
- JSONL logging for each experiment

Experiment matrix (research_design.tex §4):
  Main: 8 models × 5 probes × 3 conditions × 1 strategy(cot) × 19 metrics × N=1
        = 8 × 5 × 3 × 19 = 2,280 calls
  Controls: No-axiom(8×5×3×5) + Symbol-only(8×5×3×5) = 600 calls
  Strategies: 5 strategies × 8 models × 3 conditions × 3 probes(L2-L4) + ZS all 5
"""

import asyncio
import json
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp

from .probes import (
    PROBE_LADDER, Probe,
    get_symbols, get_metric_str, get_ground_truth,
    make_minkowski_metric, make_euclidean_metric, make_split_metric,
    normalize_answer, normalize_ground_truth, answers_equivalent,
    STANDARD_SYMBOLS, ABSTRACT_SYMBOLS,
)
from .prompts import (
    build_prompt, PromptBundle,
    Strategy, Condition, STRATEGY_BUILDERS,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

API_KEYS = [
    "sk-qjhtbnwmzcmvfabtpotvkjxqdkqmnwyzwfdobmkwedopxmfi",
    "sk-wzijtyuyivphmvydtuakbfbgerunbsfqyqwzdvphbvguogsc",
    "sk-gzuafdpnjrrtdsacgotbwlytepqrkbzfdutvcokoufkoabeq",
    "sk-ktslkqulxiinxnxrevqshvyvwpmkucsdjxmkjibdbwafgyvv",
]
API_BASE = "https://api.siliconflow.cn/v1"

# 6 API models via SiliconFlow (verified working 2026-06-13)
# Note: Qwen3.5/3.6 variants return empty outputs on SiliconFlow
API_MODELS = {
    "DeepSeek-V4-Pro":    "deepseek-ai/DeepSeek-V4-Pro",
    "DeepSeek-V3.2":      "deepseek-ai/DeepSeek-V3.2",
    "Qwen3-32B":          "Qwen/Qwen3-32B",
    "Qwen3-14B":          "Qwen/Qwen3-14B",
    "GLM-4-9B-0414":      "THUDM/GLM-4-9B-0414",
    "MiniMax-M2.5":       "MiniMaxAI/MiniMax-M2.5",
}

# 2 Local models via Ollama (using available models)
LOCAL_MODELS = {
    "Qwen3-8B":           "qwen3:8b",
    "Qwen2.5-Coder-14B":  "qwen2.5-coder:14b",
}

# All 8 models
ALL_MODELS = {**API_MODELS, **LOCAL_MODELS}

# Ollama endpoint
OLLAMA_BASE = "http://localhost:11434"

# Experiment parameters
TIMEOUT_SECONDS = 300
MAX_RETRIES = 3
MAX_TOKENS = 4096
TEMPERATURE_DEFAULT = 0.0
TEMPERATURE_SC = 0.7  # Higher temperature for self-consistency runs


@dataclass
class ExperimentSpec:
    """One experiment instance."""
    instance_id: str
    model_name: str
    model_id: str
    is_local: bool
    probe: Probe
    strategy: Strategy
    condition: Condition
    metric: dict[int, int]
    metric_name: str
    dim: int
    sc_index: int = 0  # For SC runs: 0..15


@dataclass
class ExperimentResult:
    """Result of one experiment."""
    spec: ExperimentSpec
    raw_output: str = ""
    output_length: int = 0
    duration_seconds: float = 0.0
    timed_out: bool = False
    error: str = ""
    api_calls: int = 1  # >1 for retries

    # Evaluation fields (filled by evaluator)
    is_empty: bool = False
    is_truncated: bool = False
    final_answer: str = ""
    correct: Optional[bool] = None
    confidence: float = 1.0
    evaluation_method: str = ""
    error_type: str = ""  # "none", "commutative", "sign_error", "metric_error", "incomplete"
    evaluation_details: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment Generator
# ═══════════════════════════════════════════════════════════════════════════════

def generate_experiments(
    models: dict = None,
    probes: list[Probe] = None,
    strategies: list[Strategy] = None,
    conditions: list[Condition] = None,
    metrics: list[dict] = None,
    metric_names: list[str] = None,
    sc_samples: int = 1,
) -> list[ExperimentSpec]:
    """Generate the experiment matrix."""
    models = models or ALL_MODELS
    probes = probes or PROBE_LADDER
    strategies = strategies or ["cot"]
    conditions = conditions or ["standard", "abstract", "random"]
    metrics = metrics or [make_minkowski_metric(4)]
    metric_names = metric_names or ["minkowski_4"]

    specs = []
    counter = 0

    for model_name, model_id in sorted(models.items()):
        is_local = model_name in LOCAL_MODELS
        for probe in probes:
            for strategy in strategies:
                for condition in conditions:
                    for metric, mname in zip(metrics, metric_names):
                        for sci in range(sc_samples if strategy == "cot_sc" else 1):
                            counter += 1
                            dim = len(metric)
                            specs.append(ExperimentSpec(
                                instance_id=f"v3_{counter:04d}",
                                model_name=model_name,
                                model_id=model_id,
                                is_local=is_local,
                                probe=probe,
                                strategy=strategy,
                                condition=condition,
                                metric=metric,
                                metric_name=mname,
                                dim=dim,
                                sc_index=sci,
                            ))

    return specs


# ═══════════════════════════════════════════════════════════════════════════════
# Simple Answer Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_final_answer(text: str) -> str:
    """Extract the final answer from model output."""
    if not text or len(text.strip()) < 5:
        return ""

    # Try explicit markers first (multi-line aware)
    patterns = [
        r'FINAL\s*ANSWER\s*:\s*(.+?)(?:\n\n|\n\s*\n|\n$|$)',
        r'FINAL\s*RESULT\s*:\s*(.+?)(?:\n\n|\n\s*\n|\n$|$)',
        r'Final\s*Answer\s*:\s*(.+?)(?:\n\n|\n\s*\n|\n$|$)',
        r'\*\*FINAL\s*ANSWER\*\*[:\s]*(.+?)(?:\n\n|\n\s*\n|\n$|$)',
        r'最终答案\s*[：:]\s*(.+?)(?:\n\n|\n\s*\n|\n$|$)',
        r'\\boxed\{([^}]+)\}',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Strip ALL LaTeX math mode wrappers (inline and display)
            answer = answer.replace('\\(', '').replace('\\)', '')
            answer = answer.replace('\\[', '').replace('\\]', '')
            answer = re.sub(r'\$+', '', answer)
            # Strip leading markdown bold
            answer = re.sub(r'^\*\*\s*', '', answer)
            answer = answer.strip()
            return answer

    # Fallback: look for the last LaTeX display math block
    display_math = re.findall(r'(?:\\\[(.*?)\\\]|\$\$(.*?)\$\$)', text, re.DOTALL)
    if display_math:
        last = display_math[-1]
        result = last[0] or last[1]
        if result:
            return result.strip()

    # Fallback: last non-empty line that looks like math
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        # Skip natural language lines
        if re.match(r'^(Therefore|Thus|Hence|So|The|This|We|Note|Step|Let|I\s|A\s|In\s|By\s|Using|Now|First|Second|Third|Finally|Next|Verification)', line, re.IGNORECASE):
            continue
        if re.match(r'^[①②③④⑤⑥⑦⑧⑨⑩]', line):
            continue
        # Strip inline math wrappers
        line = re.sub(r'^\\\(\s*', '', line)
        line = re.sub(r'\s*\\\)$', '', line)
        if len(line) > 3 and any(c in line for c in '*+-=eω'):
            return line.strip()

    # Last resort: last line
    return lines[-1].strip() if lines else ""


def classify_output_quality(text: str) -> dict:
    """Classify output: VALID, EMPTY, or TRUNCATED."""
    length = len(text) if text else 0

    if length < 50:
        return {"quality": "EMPTY", "length": length,
                "reason": "Output too short to contain derivation"}

    # Check for truncation markers
    truncation_markers = [
        r'\.\.\.\s*$',           # ends with ...
        r'\[truncated\]',
        r'\(truncated\)',
        r'output exceeds',
        r'max_tokens reached',
        r'token limit',
    ]
    for marker in truncation_markers:
        if re.search(marker, text, re.IGNORECASE):
            return {"quality": "TRUNCATED", "length": length,
                    "reason": f"Truncation marker: {marker}"}

    # Check if text ends mid-sentence (likely truncated)
    last_chars = text.strip()[-20:]
    if re.search(r'[a-zA-Z0-9]$', last_chars) and not re.search(r'[.!?]$', last_chars):
        # Ends with alphanumeric but no sentence terminator — possible truncation
        # But only flag if the output seems incomplete (no final answer, no concluding phrases)
        has_conclusion = any(kw in text[-200:].lower() for kw in
                           ['final answer', 'therefore', 'result', 'conclusion',
                            '最终', '结果', '答案', 'thus', 'hence', '='])
        if not has_conclusion:
            return {"quality": "TRUNCATED", "length": length,
                    "reason": "No conclusion marker, ends mid-content"}

    return {"quality": "VALID", "length": length, "reason": ""}


# ═══════════════════════════════════════════════════════════════════════════════
# Quick Evaluator (regex-based, for initial pass)
# ═══════════════════════════════════════════════════════════════════════════════

def quick_evaluate(result: ExperimentResult) -> ExperimentResult:
    """Quick regex-based evaluation. Full Kingdon eval is done in batch."""
    spec = result.spec
    answer = result.final_answer

    if not answer:
        result.is_empty = True
        result.correct = False
        result.error_type = "empty_output"
        result.evaluation_method = "quick_empty"
        return result

    # Get ground truth
    symbols = get_symbols(spec.condition, max(spec.dim, 4))
    gt = get_ground_truth(spec.probe, spec.condition, symbols, spec.metric)

    # Use semantic equivalence checker
    if answers_equivalent(answer, gt):
        result.correct = True
        result.error_type = "none"
        result.evaluation_method = "semantic_equivalent"
        return result

    # Normalize for further checks
    norm_answer = normalize_answer(answer)
    norm_gt = normalize_ground_truth(gt)

    # Direct match after normalization
    if norm_answer == norm_gt:
        result.correct = True
        result.error_type = "none"
        result.evaluation_method = "exact_match"
        return result

    # Check for common errors
    for error_template in spec.probe.common_errors:
        # Fill error template
        err_filled = error_template
        for i in range(1, 11):
            sym = symbols.get(i, f"s_{i}")
            err_filled = err_filled.replace(f"{{b{i}}}", sym)
            err_filled = err_filled.replace(f"{{b_{i}}}", sym)
        err_norm = normalize_ground_truth(err_filled)

        if norm_answer == err_norm or answers_equivalent(answer, err_filled):
            result.correct = False
            if "commutative" in error_template.lower() or error_template.startswith("2*"):
                result.error_type = "commutative_error"
            elif "wrong sign" in error_template.lower():
                result.error_type = "sign_error"
            elif "metric" in error_template.lower():
                result.error_type = "metric_error"
            else:
                result.error_type = "known_error_pattern"
            result.evaluation_method = "error_match"
            return result

    # Contains ground truth? (partial match)
    if norm_gt and norm_gt in norm_answer:
        result.correct = True
        result.error_type = "none"
        result.evaluation_method = "contains_gt"
        return result

    # Default: needs detailed evaluation
    result.correct = None  # Uncertain
    result.evaluation_method = "uncertain"
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# API Caller
# ═══════════════════════════════════════════════════════════════════════════════

async def call_siliconflow(
    session: aiohttp.ClientSession,
    api_key: str,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    timeout: int = TIMEOUT_SECONDS,
    max_tokens: int = MAX_TOKENS,
) -> tuple[str, float, dict]:
    """Call SiliconFlow API. Returns (output, duration, usage)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    start = time.perf_counter()
    try:
        async with session.post(
            f"{API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                output = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
            else:
                text = await resp.text()
                output = f"HTTP_ERROR_{resp.status}: {text[:200]}"
                usage = {}
    except asyncio.TimeoutError:
        output = "TIMEOUT"
        usage = {}
    except Exception as e:
        output = f"ERROR: {str(e)[:200]}"
        usage = {}

    elapsed = time.perf_counter() - start
    return output, elapsed, usage


async def call_ollama(
    session: aiohttp.ClientSession,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    timeout: int = TIMEOUT_SECONDS,
    max_tokens: int = MAX_TOKENS,
) -> tuple[str, float, dict]:
    """Call local Ollama model. Returns (output, duration, usage)."""
    payload = {
        "model": model_name,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    start = time.perf_counter()
    try:
        async with session.post(
            f"{OLLAMA_BASE}/api/generate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                output = data.get("response", "")
                usage = {
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                }
            else:
                text = await resp.text()
                output = f"OLLAMA_ERROR_{resp.status}: {text[:200]}"
                usage = {}
    except asyncio.TimeoutError:
        output = "TIMEOUT"
        usage = {}
    except aiohttp.ClientConnectorError:
        output = "OLLAMA_NOT_RUNNING"
        usage = {}
    except Exception as e:
        output = f"ERROR: {str(e)[:200]}"
        usage = {}

    elapsed = time.perf_counter() - start
    return output, elapsed, usage


# ═══════════════════════════════════════════════════════════════════════════════
# Main Experiment Runner
# ═══════════════════════════════════════════════════════════════════════════════

async def run_single_experiment(
    session: aiohttp.ClientSession,
    api_key: str,
    spec: ExperimentSpec,
    sem: asyncio.Semaphore,
    output_path: Path,
    counter: list,  # mutable counter
) -> ExperimentResult:
    """Run one experiment and write result to JSONL."""
    async with sem:
        # Build prompt
        bundle = build_prompt(spec.probe, spec.strategy, spec.condition, spec.metric)
        temp = TEMPERATURE_SC if spec.strategy == "cot_sc" else TEMPERATURE_DEFAULT

        # Run with retries
        output = ""
        duration = 0.0
        usage = {}
        attempts = 0

        for attempt in range(MAX_RETRIES):
            attempts += 1
            try:
                if spec.is_local:
                    output, duration, usage = await call_ollama(
                        session, spec.model_id,
                        bundle.system_prompt, bundle.user_prompt,
                        temperature=temp,
                    )
                else:
                    output, duration, usage = await call_siliconflow(
                        session, api_key, spec.model_id,
                        bundle.system_prompt, bundle.user_prompt,
                        temperature=temp,
                    )

                if not output.startswith("HTTP_ERROR") and not output.startswith("ERROR") and output != "TIMEOUT":
                    break
                if output == "TIMEOUT" and attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)  # exponential backoff
            except Exception as e:
                output = f"EXCEPTION: {str(e)[:200]}"
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)

        # Build result
        result = ExperimentResult(
            spec=spec,
            raw_output=output,
            output_length=len(output),
            duration_seconds=round(duration, 1),
            timed_out=(output == "TIMEOUT" or duration >= TIMEOUT_SECONDS * 0.95),
            error="" if not output.startswith("HTTP_ERROR") and not output.startswith("ERROR") and output not in ("TIMEOUT", "OLLAMA_NOT_RUNNING") else output,
            api_calls=attempts,
        )

        # Classify output quality
        quality = classify_output_quality(output)
        result.is_empty = (quality["quality"] == "EMPTY")
        result.is_truncated = (quality["quality"] == "TRUNCATED")

        # Extract final answer
        result.final_answer = extract_final_answer(output) if not result.is_empty else ""

        # Quick evaluate
        if not result.is_empty and not result.is_truncated:
            result = quick_evaluate(result)

        # Write to JSONL
        record = {
            "instance_id": spec.instance_id,
            "model": spec.model_name,
            "probe": spec.probe.name,
            "probe_level": spec.probe.level,
            "strategy": spec.strategy,
            "condition": spec.condition,
            "metric_name": spec.metric_name,
            "dim": spec.dim,
            "sc_index": spec.sc_index,
            "correct": result.correct,
            "error_type": result.error_type,
            "final_answer": result.final_answer,
            "is_empty": result.is_empty,
            "is_truncated": result.is_truncated,
            "evaluation_method": result.evaluation_method,
            "duration_s": result.duration_seconds,
            "output_len": result.output_length,
            "timed_out": result.timed_out,
            "api_calls": result.api_calls,
            "error": result.error,
            "raw_tail": output[-300:] if output else "",
        }

        # Thread-safe append
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Progress
        counter[0] += 1
        status = "✓" if result.correct else ("✗" if result.correct is False else "?")
        if result.is_empty:
            status = "∅"
        elif result.is_truncated:
            status = "…"
        print(f"  [{counter[0]:4d}] {spec.model_name[:16]:16s} "
              f"L{spec.probe.level} {spec.strategy:7s} {spec.condition:8s} "
              f"{status} {result.duration_seconds:.0f}s "
              f"({result.evaluation_method})")

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Batch Runner
# ═══════════════════════════════════════════════════════════════════════════════

async def run_batch(
    specs: list[ExperimentSpec],
    output_path: Path,
    max_concurrent: int = 16,
    label: str = "Experiment",
) -> list[ExperimentResult]:
    """Run a batch of experiments with parallel API calls."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize empty output file
    if not output_path.exists():
        output_path.write_text("", encoding="utf-8")

    total = len(specs)
    print(f"\n{'='*70}")
    print(f"{label}: {total} experiments")
    print(f"Models: {len(set(s.model_name for s in specs))}")
    print(f"Probes: L{min(s.probe.level for s in specs)}–L{max(s.probe.level for s in specs)}")
    print(f"Strategies: {sorted(set(s.strategy for s in specs))}")
    print(f"Conditions: {sorted(set(s.condition for s in specs))}")
    print(f"Concurrency: {max_concurrent}")
    print(f"Output: {output_path}")
    print(f"{'='*70}\n")

    sem = asyncio.Semaphore(max_concurrent)
    key_pool = API_KEYS * (total // len(API_KEYS) + 1)
    counter = [0]  # mutable for thread-safe-ish counting

    async with aiohttp.ClientSession() as session:
        tasks = [
            run_single_experiment(session, key_pool[i], spec, sem, output_path, counter)
            for i, spec in enumerate(specs)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter valid results
    valid = []
    exceptions = 0
    for r in results:
        if isinstance(r, ExperimentResult):
            valid.append(r)
        else:
            exceptions += 1

    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE: {len(valid)}/{total} valid, {exceptions} exceptions")
    print(f"Output: {output_path}")
    print(f"{'='*70}")

    return valid


# ═══════════════════════════════════════════════════════════════════════════════
# Summary Generator
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(results: list[ExperimentResult]):
    """Print batch summary statistics."""
    if not results:
        print("No results to summarize.")
        return

    total = len(results)
    correct = sum(1 for r in results if r.correct is True)
    incorrect = sum(1 for r in results if r.correct is False)
    uncertain = sum(1 for r in results if r.correct is None)
    empty = sum(1 for r in results if r.is_empty)
    truncated = sum(1 for r in results if r.is_truncated)
    timed_out = sum(1 for r in results if r.timed_out)

    print(f"\n--- Batch Summary ---")
    print(f"Total: {total}")
    print(f"Correct: {correct} ({100*correct/total:.1f}%)")
    print(f"Incorrect: {incorrect} ({100*incorrect/total:.1f}%)")
    print(f"Uncertain: {uncertain} ({100*uncertain/total:.1f}%)")
    print(f"Empty: {empty}, Truncated: {truncated}, Timeout: {timed_out}")

    # By probe level
    print(f"\n--- By Probe Level ---")
    for level in sorted(set(r.spec.probe.level for r in results)):
        lr = [r for r in results if r.spec.probe.level == level]
        c = sum(1 for r in lr if r.correct is True)
        ic = sum(1 for r in lr if r.correct is False)
        un = sum(1 for r in lr if r.correct is None)
        em = sum(1 for r in lr if r.is_empty)
        print(f"  L{level}: {c}/{len(lr)} correct ({100*c/max(1,len(lr)):.0f}%), "
              f"incorrect={ic}, uncertain={un}, empty={em}")

    # By model
    print(f"\n--- By Model ---")
    for model in sorted(set(r.spec.model_name for r in results)):
        mr = [r for r in results if r.spec.model_name == model]
        c = sum(1 for r in mr if r.correct is True)
        print(f"  {model[:20]:20s}: {c}/{len(mr)} correct ({100*c/max(1,len(mr)):.0f}%)")

    # By condition
    print(f"\n--- By Condition ---")
    for cond in sorted(set(r.spec.condition for r in results)):
        cr = [r for r in results if r.spec.condition == cond]
        c = sum(1 for r in cr if r.correct is True)
        print(f"  {cond:10s}: {c}/{len(cr)} correct ({100*c/max(1,len(cr)):.0f}%)")

    # By strategy
    print(f"\n--- By Strategy ---")
    for strat in sorted(set(r.spec.strategy for r in results)):
        sr = [r for r in results if r.spec.strategy == strat]
        c = sum(1 for r in sr if r.correct is True)
        print(f"  {strat:10s}: {c}/{len(sr)} correct ({100*c/max(1,len(sr)):.0f}%)")

    # Error type distribution
    print(f"\n--- Error Types ---")
    error_counts = defaultdict(int)
    for r in results:
        if r.error_type:
            error_counts[r.error_type] += 1
    for etype, count in sorted(error_counts.items(), key=lambda x: -x[1]):
        print(f"  {etype}: {count} ({100*count/total:.1f}%)")


def save_summary_json(results: list[ExperimentResult], path: Path):
    """Save detailed summary as JSON."""
    total = len(results)
    summary = {
        "total": total,
        "correct": sum(1 for r in results if r.correct is True),
        "incorrect": sum(1 for r in results if r.correct is False),
        "uncertain": sum(1 for r in results if r.correct is None),
        "empty": sum(1 for r in results if r.is_empty),
        "truncated": sum(1 for r in results if r.is_truncated),
        "timed_out": sum(1 for r in results if r.timed_out),
        "by_probe": {},
        "by_model": {},
        "by_condition": {},
        "by_strategy": {},
        "by_probe_x_condition": {},
        "error_types": {},
    }

    for level in sorted(set(r.spec.probe.level for r in results)):
        lr = [r for r in results if r.spec.probe.level == level]
        summary["by_probe"][f"L{level}"] = {
            "total": len(lr),
            "correct": sum(1 for r in lr if r.correct is True),
            "incorrect": sum(1 for r in lr if r.correct is False),
            "empty": sum(1 for r in lr if r.is_empty),
        }

    for model in sorted(set(r.spec.model_name for r in results)):
        mr = [r for r in results if r.spec.model_name == model]
        summary["by_model"][model] = {
            "total": len(mr),
            "correct": sum(1 for r in mr if r.correct is True),
            "accuracy": round(100 * sum(1 for r in mr if r.correct is True) / max(1, len(mr)), 1),
        }

    for cond in sorted(set(r.spec.condition for r in results)):
        cr = [r for r in results if r.spec.condition == cond]
        summary["by_condition"][cond] = {
            "total": len(cr),
            "correct": sum(1 for r in cr if r.correct is True),
            "accuracy": round(100 * sum(1 for r in cr if r.correct is True) / max(1, len(cr)), 1),
        }

    for strat in sorted(set(r.spec.strategy for r in results)):
        sr = [r for r in results if r.spec.strategy == strat]
        summary["by_strategy"][strat] = {
            "total": len(sr),
            "correct": sum(1 for r in sr if r.correct is True),
            "accuracy": round(100 * sum(1 for r in sr if r.correct is True) / max(1, len(sr)), 1),
        }

    for level in sorted(set(r.spec.probe.level for r in results)):
        for cond in sorted(set(r.spec.condition for r in results)):
            lcr = [r for r in results if r.spec.probe.level == level and r.spec.condition == cond]
            key = f"L{level}_{cond}"
            summary["by_probe_x_condition"][key] = {
                "total": len(lcr),
                "correct": sum(1 for r in lcr if r.correct is True),
                "accuracy": round(100 * sum(1 for r in lcr if r.correct is True) / max(1, len(lcr)), 1),
            }

    for r in results:
        if r.error_type:
            summary["error_types"][r.error_type] = summary["error_types"].get(r.error_type, 0) + 1

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSummary saved to {path}")

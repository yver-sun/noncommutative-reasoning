#!/usr/bin/env python3
"""
COMPREHENSIVE RESEARCH STUDY — A(S) Symbol Isolation Framework
==============================================================
15 models (11 API + 4 local) × 5 probes × 3 conditions
Target: ~2,500 experiments in 6-8 hours

Usage:
  python run_comprehensive_study.py              # Run all phases
  python run_comprehensive_study.py --phase 1    # Main experiment only
  python run_comprehensive_study.py --phase 2    # Strategy comparison only
  python run_comprehensive_study.py --phase 3    # Controls only
  python run_comprehensive_study.py --resume     # Resume from checkpoint
"""

import asyncio
import json
import os
import re
import sys
import time
import copy
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp

sys.path.insert(0, str(Path(__file__).parent))
from experiment_v3.probes import (
    PROBE_LADDER, PROBE_L0, PROBE_L1, PROBE_L2, PROBE_L3, PROBE_L4,
    get_symbols, get_metric_str, get_ground_truth, get_question,
    make_minkowski_metric, make_euclidean_metric, make_split_metric,
    normalize_answer, normalize_ground_truth, answers_equivalent,
)
from experiment_v3.prompts import (
    build_prompt, PromptBundle,
    Strategy, Condition, STRATEGY_BUILDERS,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

API_KEYS = [
    "sk-qjhtbnwmzcmvfabtpotvkjxqdkqmnwyzwfdobmkwedopxmfi",
    "sk-wzijtyuyivphmvydtuakbfbgerunbsfqyqwzdvphbvguogsc",
    "sk-gzuafdpnjrrtdsacgotbwlytepqrkbzfdutvcokoufkoabeq",
    "sk-ktslkqulxiinxnxrevqshvyvwpmkucsdjxmkjibdbwafgyvv",
]
API_BASE = "https://api.siliconflow.cn/v1"
OLLAMA_BASE = "http://localhost:11434"

# ── All 11 API models (verified working) ──
API_MODELS = {
    "DeepSeek-V4-Pro":    "deepseek-ai/DeepSeek-V4-Pro",
    "DeepSeek-V3.2":      "deepseek-ai/DeepSeek-V3.2",
    "DeepSeek-R1":        "deepseek-ai/DeepSeek-R1",
    "DeepSeek-V4-Flash":  "deepseek-ai/DeepSeek-V4-Flash",
    "DeepSeek-V3":        "deepseek-ai/DeepSeek-V3",
    "Qwen3-32B":          "Qwen/Qwen3-32B",
    "Qwen3-14B":          "Qwen/Qwen3-14B",
    "Qwen3-8B":           "Qwen/Qwen3-8B",
    "GLM-4-9B-0414":      "THUDM/GLM-4-9B-0414",
    "GLM-Z1-9B-0414":     "THUDM/GLM-Z1-9B-0414",
    "MiniMax-M2.5":       "MiniMaxAI/MiniMax-M2.5",
}

# ── 4 Local models (Ollama) ──
LOCAL_MODELS = {
    "Qwen3-8B-Local":       "qwen3:8b",
    "Qwen2.5-Coder-14B":    "qwen2.5-coder:14b",
    "DeepSeek-R1-8B-Local": "deepseek-r1:8b",
    "Qwen2.5-14B":          "qwen2.5:14b",
}

ALL_MODELS = {**API_MODELS, **LOCAL_MODELS}

# ── Experiment Parameters ──
TIMEOUT = 300
MAX_TOKENS = 4096
MAX_RETRIES = 2

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExpSpec:
    instance_id: str
    model_name: str
    model_id: str
    is_local: bool
    probe_name: str
    probe_level: int
    strategy: str
    condition: str
    metric: dict
    metric_name: str
    dim: int
    rep: int = 0
    sc_index: int = 0

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_specs(models, probes, strategies, conditions,
                   metrics, metric_names, reps=1, sc_samples=1,
                   existing_ids=None):
    """Generate experiment specs, skipping already-completed ones."""
    existing_ids = existing_ids or set()
    specs = []
    counter = 0

    for model_name, model_id in sorted(models.items()):
        is_local = model_name in LOCAL_MODELS
        for probe in probes:
            for strategy in strategies:
                for condition in conditions:
                    for metric, mname in zip(metrics, metric_names):
                        for rep in range(reps):
                            for sci in range(sc_samples if strategy == "cot_sc" else 1):
                                iid = f"v4_{model_name[:12]}_L{probe.level}_{strategy}_{condition}_{mname}_r{rep}_sc{sci}"
                                if iid not in existing_ids:
                                    counter += 1
                                    specs.append(ExpSpec(
                                        instance_id=iid,
                                        model_name=model_name,
                                        model_id=model_id,
                                        is_local=is_local,
                                        probe_name=probe.name,
                                        probe_level=probe.level,
                                        strategy=strategy,
                                        condition=condition,
                                        metric=metric,
                                        metric_name=mname,
                                        dim=len(metric),
                                        rep=rep,
                                        sc_index=sci,
                                    ))
    return specs


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_final_answer(text: str) -> str:
    if not text or len(text.strip()) < 5:
        return ""
    patterns = [
        r'FINAL\s*ANSWER\s*:\s*(.+?)(?:\n\n|\n\s*\n|\n$|$)',
        r'FINAL\s*RESULT\s*:\s*(.+?)(?:\n\n|\n\s*\n|\n$|$)',
        r'\*\*FINAL\s*ANSWER\*\*[:\s]*(.+?)(?:\n\n|\n\s*\n|\n$|$)',
        r'最终答案\s*[：:]\s*(.+?)(?:\n\n|\n\s*\n|\n$|$)',
        r'\\boxed\{([^}]+)\}',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            answer = answer.replace('\\(', '').replace('\\)', '')
            answer = answer.replace('\\[', '').replace('\\]', '')
            answer = re.sub(r'\$+', '', answer)
            answer = re.sub(r'^\*\*\s*', '', answer)
            return answer.strip()

    display_math = re.findall(r'(?:\\\[(.*?)\\\]|\$\$(.*?)\$\$)', text, re.DOTALL)
    if display_math:
        last = display_math[-1]
        result = last[0] or last[1]
        if result:
            return result.strip()

    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if re.match(r'^(Therefore|Thus|Hence|So|The|This|We|Note|Step|Let|I\s|A\s|In\s|By\s|Using|Now|First|Second|Third|Finally|Next|Verif)', line, re.IGNORECASE):
            continue
        line = line.replace('\\(', '').replace('\\)', '')
        if len(line) > 3 and any(c in line for c in '*+-=eEwW'):
            return line.strip()
    return lines[-1].strip() if lines else ""


def classify_output(text: str) -> dict:
    length = len(text) if text else 0
    if length < 50:
        return {"quality": "EMPTY", "len": length}
    truncation_markers = [r'\.\.\.\s*$', r'\[truncated\]', r'max_tokens reached']
    for marker in truncation_markers:
        if re.search(marker, text, re.IGNORECASE):
            return {"quality": "TRUNCATED", "len": length}
    return {"quality": "VALID", "len": length}


def evaluate_answer(answer, probe, condition, metric):
    """Evaluate one answer. Returns (correct, error_type, method)."""
    if not answer:
        return None, "empty_output", "empty"

    symbols = get_symbols(condition, max(len(metric), 4))
    gt = get_ground_truth(probe, condition, symbols, metric)

    if answers_equivalent(answer, gt):
        return True, "none", "semantic_equivalent"

    # Check common errors
    for error_template in probe.common_errors:
        ef = error_template
        for i in range(1, 11):
            sym = symbols.get(i, f"s_{i}")
            ef = ef.replace(f"{{b{i}}}", sym)
        if answers_equivalent(answer, ef):
            if "commutative" in error_template.lower():
                return False, "commutative_error", "error_match"
            elif "sign" in error_template.lower():
                return False, "sign_error", "error_match"
            elif "metric" in error_template.lower():
                return False, "metric_error", "error_match"
            else:
                return False, "known_error", "error_match"

    # Contains GT?
    na = normalize_answer(answer)
    ng = normalize_ground_truth(gt)
    if ng and ng in na:
        return True, "none", "contains_gt"

    return None, "uncertain", "uncertain"


# ═══════════════════════════════════════════════════════════════════════════════
# API CALLERS
# ═══════════════════════════════════════════════════════════════════════════════

async def call_api(session, api_key, model_id, system_prompt, user_prompt,
                   temperature=0.0, timeout=TIMEOUT, max_tokens=MAX_TOKENS):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": [{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    start = time.perf_counter()
    try:
        async with session.post(f"{API_BASE}/chat/completions", headers=headers,
                                json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data["choices"][0]["message"]["content"], time.perf_counter() - start, data.get("usage", {})
            else:
                text = await resp.text()
                return f"HTTP_{resp.status}:{text[:200]}", time.perf_counter() - start, {}
    except asyncio.TimeoutError:
        return "TIMEOUT", time.perf_counter() - start, {}
    except Exception as e:
        return f"ERR:{str(e)[:200]}", time.perf_counter() - start, {}


async def call_ollama(session, model_name, system_prompt, user_prompt,
                      temperature=0.0, timeout=TIMEOUT, max_tokens=MAX_TOKENS):
    payload = {
        "model": model_name,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    start = time.perf_counter()
    try:
        async with session.post(f"{OLLAMA_BASE}/api/generate", json=payload,
                                timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status == 200:
                data = await resp.json()
                usage = {"prompt_tokens": data.get("prompt_eval_count", 0),
                         "completion_tokens": data.get("eval_count", 0)}
                return data.get("response", ""), time.perf_counter() - start, usage
            else:
                text = await resp.text()
                return f"OLLAMA_{resp.status}:{text[:200]}", time.perf_counter() - start, {}
    except asyncio.TimeoutError:
        return "TIMEOUT", time.perf_counter() - start, {}
    except aiohttp.ClientConnectorError:
        return "OLLAMA_NOT_RUNNING", time.perf_counter() - start, {}
    except Exception as e:
        return f"ERR:{str(e)[:200]}", time.perf_counter() - start, {}


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

async def run_one(session, api_key, spec, api_sem, local_sem, output_path, counter, start_time):
    sem = local_sem if spec.is_local else api_sem
    async with sem:
        # Build prompt
        probe = [p for p in PROBE_LADDER if p.level == spec.probe_level][0]
        bundle = build_prompt(probe, spec.strategy, spec.condition, spec.metric)
        temp = 0.7 if spec.strategy == "cot_sc" else 0.0

        # Run with retries
        output = ""
        duration = 0.0
        usage = {}
        for attempt in range(MAX_RETRIES + 1):
            try:
                if spec.is_local:
                    output, duration, usage = await call_ollama(
                        session, spec.model_id, bundle.system_prompt, bundle.user_prompt, temperature=temp)
                else:
                    output, duration, usage = await call_api(
                        session, api_key, spec.model_id, bundle.system_prompt, bundle.user_prompt, temperature=temp)

                if not output.startswith("HTTP_") and not output.startswith("ERR") and output not in ("TIMEOUT", "OLLAMA_NOT_RUNNING"):
                    break
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                output = f"EXC:{str(e)[:200]}"
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2 ** attempt)

        # Classify output
        quality = classify_output(output)
        is_empty = quality["quality"] == "EMPTY"
        is_truncated = quality["quality"] == "TRUNCATED"
        timed_out = output == "TIMEOUT"

        # Extract and evaluate answer
        answer = extract_final_answer(output) if not is_empty else ""
        correct, error_type, eval_method = evaluate_answer(answer, probe, spec.condition, spec.metric)

        # Build record
        record = {
            "instance_id": spec.instance_id,
            "model": spec.model_name,
            "probe": spec.probe_name,
            "probe_level": spec.probe_level,
            "strategy": spec.strategy,
            "condition": spec.condition,
            "metric_name": spec.metric_name,
            "dim": spec.dim,
            "rep": spec.rep,
            "correct": correct,
            "error_type": error_type,
            "final_answer": answer,
            "is_empty": is_empty,
            "is_truncated": is_truncated,
            "evaluation_method": eval_method,
            "duration_s": round(duration, 1),
            "output_len": len(output),
            "timed_out": timed_out,
            "raw_tail": output[-300:] if output else "",
        }

        # Write immediately (crash-safe)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Progress
        counter[0] += 1
        elapsed = time.perf_counter() - start_time
        rate = counter[0] / (elapsed / 3600) if elapsed > 0 else 0
        status = "OK" if correct else ("ER" if correct is False else ("EM" if is_empty else "??"))
        print(f"[{counter[0]:5d}] {spec.model_name[:18]:18s} L{spec.probe_level} "
              f"{spec.strategy:7s} {spec.condition:8s} {status} {duration:.0f}s "
              f"({rate:.0f}/hr)")

        return record


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

async def run_batch(specs, output_path, api_concurrent=12, local_concurrent=2, label="Batch"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing IDs (resume support)
    existing_ids = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        existing_ids.add(json.loads(line)["instance_id"])
                    except json.JSONDecodeError:
                        pass

    # Filter out already-completed
    new_specs = [s for s in specs if s.instance_id not in existing_ids]
    skipped = len(specs) - len(new_specs)

    if not new_specs:
        print(f"All {len(specs)} experiments already completed!")
        return []

    total = len(new_specs)
    print(f"\n{'='*70}")
    print(f"{label}: {total} experiments ({skipped} already done, {len(existing_ids)} total)")
    print(f"API models: {len(set(s.model_name for s in new_specs if not s.is_local))}")
    print(f"Local models: {len(set(s.model_name for s in new_specs if s.is_local))}")
    print(f"API concurrent: {api_concurrent}, Local concurrent: {local_concurrent}")
    print(f"Output: {output_path}")
    print(f"{'='*70}\n")

    api_sem = asyncio.Semaphore(api_concurrent)
    local_sem = asyncio.Semaphore(local_concurrent)
    key_pool = API_KEYS * ((total // len(API_KEYS)) + 1)
    counter = [0]
    start_time = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        tasks = [run_one(session, key_pool[i % len(API_KEYS)], spec,
                        api_sem, local_sem, output_path, counter, start_time)
                for i, spec in enumerate(new_specs)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_results = [r for r in results if isinstance(r, dict)]
    exceptions = sum(1 for r in results if not isinstance(r, dict))

    elapsed = time.perf_counter() - start_time
    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE: {len(valid_results)}/{total} valid, {exceptions} exceptions")
    print(f"Duration: {elapsed/3600:.1f} hours")
    print(f"Output: {output_path}")
    print(f"{'='*70}")

    return valid_results


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(output_path):
    results = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    total = len(results)
    ok = sum(1 for r in results if r.get("correct") is True)
    err = sum(1 for r in results if r.get("correct") is False)
    unk = sum(1 for r in results if r.get("correct") is None)
    emp = sum(1 for r in results if r.get("is_empty"))

    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY: {total} experiments")
    print(f"  Correct: {ok} ({100*ok/total:.1f}%)")
    print(f"  Wrong:   {err} ({100*err/total:.1f}%)")
    print(f"  Uncertain: {unk}")
    print(f"  Empty:   {emp}")
    print(f"{'='*70}")

    # By model
    print(f"\n--- By Model ---")
    models = defaultdict(list)
    for r in results:
        models[r["model"]].append(r)
    for model in sorted(models, key=lambda m: -len(models[m])):
        mr = models[model]
        c = sum(1 for r in mr if r["correct"] is True)
        e = sum(1 for r in mr if r.get("is_empty"))
        print(f"  {model[:25]:25s}: {c}/{len(mr)} correct ({100*c/max(1,len(mr)):.0f}%), {e} empty")

    # By probe × condition
    print(f"\n--- Accuracy by Probe x Condition ---")
    for level in range(5):
        for cond in ["standard", "abstract", "random"]:
            lr = [r for r in results if r["probe_level"] == level and r["condition"] == cond
                  and not r.get("is_empty")]
            if lr:
                c = sum(1 for r in lr if r["correct"] is True)
                print(f"  L{level} {cond:10s}: {c}/{len(lr)} ({100*c/len(lr):.0f}%)")

    # Save summary
    summary_path = Path(output_path).parent / f"{Path(output_path).stem}_summary.json"
    summary = {
        "total": total, "correct": ok, "wrong": err, "uncertain": unk, "empty": emp,
        "by_model": {m: {"total": len(mr), "correct": sum(1 for r in mr if r["correct"] is True)}
                     for m, mr in models.items()},
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {summary_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

METRICS = {
    "minkowski_4": make_minkowski_metric(4),
    "euclidean_4": make_euclidean_metric(4),
    "split_4": make_split_metric(4),
}

async def phase1_main(output_dir):
    """Main experiment: 15 models × 5 probes × 3 conditions × reps."""
    print("\n" + "█"*70)
    print("█  PHASE 1: MAIN EXPERIMENT")
    print("█"*70)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"phase1_main_{now}.jsonl"

    # API: 11 models × 5 probes × 3 conditions × 3 metrics × 3 reps = 1,485
    # Local: 4 models × 5 probes × 3 conditions × 1 metric × 3 reps = 180
    # Total: ~1,665

    specs = []
    metric_list = list(METRICS.values())
    metric_names = list(METRICS.keys())

    # API models: all 3 metrics, N=3
    for model_name, model_id in API_MODELS.items():
        for probe in PROBE_LADDER:
            for condition in ["standard", "abstract", "random"]:
                for mname, metric in METRICS.items():
                    for rep in range(3):
                        specs.append(ExpSpec(
                            instance_id=f"p1_{model_name[:12]}_L{probe.level}_cot_{condition}_{mname}_r{rep}",
                            model_name=model_name, model_id=model_id, is_local=False,
                            probe_name=probe.name, probe_level=probe.level,
                            strategy="cot", condition=condition,
                            metric=metric, metric_name=mname, dim=len(metric), rep=rep))

    # Local models: Minkowski only, N=3
    for model_name, model_id in LOCAL_MODELS.items():
        for probe in PROBE_LADDER:
            for condition in ["standard", "abstract", "random"]:
                for rep in range(3):
                    specs.append(ExpSpec(
                        instance_id=f"p1_{model_name[:12]}_L{probe.level}_cot_{condition}_minkowski4_r{rep}",
                        model_name=model_name, model_id=model_id, is_local=True,
                        probe_name=probe.name, probe_level=probe.level,
                        strategy="cot", condition=condition,
                        metric=make_minkowski_metric(4), metric_name="minkowski_4", dim=4, rep=rep))

    print(f"Total specs: {len(specs)}")
    return await run_batch(specs, output_path, api_concurrent=12, local_concurrent=2,
                          label="Phase 1: Main Experiment")


async def phase2_strategies(output_dir):
    """Strategy comparison: 5 strategies × key models × L2-L4 × 3 conditions × N=3."""
    print("\n" + "█"*70)
    print("█  PHASE 2: STRATEGY COMPARISON")
    print("█"*70)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"phase2_strategies_{now}.jsonl"

    # Focus on representative models (top + weak)
    strategy_models = {
        "DeepSeek-V4-Pro": "deepseek-ai/DeepSeek-V4-Pro",
        "DeepSeek-V3.2": "deepseek-ai/DeepSeek-V3.2",
        "Qwen3-32B": "Qwen/Qwen3-32B",
        "GLM-4-9B-0414": "THUDM/GLM-4-9B-0414",
        "MiniMax-M2.5": "MiniMaxAI/MiniMax-M2.5",
        "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
        "GLM-Z1-9B-0414": "THUDM/GLM-Z1-9B-0414",
        "Qwen3-14B": "Qwen/Qwen3-14B",
    }

    strategies = ["zs", "cot", "cot_sc", "cot_pot", "cot_sv"]
    probes_l234 = [PROBE_L2, PROBE_L3, PROBE_L4]

    specs = []
    for model_name, model_id in strategy_models.items():
        for probe in probes_l234:
            for strategy in strategies:
                for condition in ["standard", "abstract", "random"]:
                    for rep in range(3):
                        sc_samples = 5 if strategy == "cot_sc" else 1
                        for sci in range(sc_samples):
                            specs.append(ExpSpec(
                                instance_id=f"p2_{model_name[:10]}_L{probe.level}_{strategy}_{condition}_r{rep}_sc{sci}",
                                model_name=model_name, model_id=model_id, is_local=False,
                                probe_name=probe.name, probe_level=probe.level,
                                strategy=strategy, condition=condition,
                                metric=make_minkowski_metric(4), metric_name="minkowski_4", dim=4,
                                rep=rep, sc_index=sci))

    print(f"Total specs: {len(specs)}")
    return await run_batch(specs, output_path, api_concurrent=12, local_concurrent=0,
                          label="Phase 2: Strategy Comparison")


async def phase3_controls(output_dir):
    """Control: no-axiom condition, 6 API models × 5 probes × 3 conditions × N=3."""
    print("\n" + "█"*70)
    print("█  PHASE 3: CONTROL EXPERIMENTS")
    print("█"*70)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"phase3_controls_{now}.jsonl"

    control_models = {
        "DeepSeek-V4-Pro": "deepseek-ai/DeepSeek-V4-Pro",
        "DeepSeek-V3.2": "deepseek-ai/DeepSeek-V3.2",
        "Qwen3-32B": "Qwen/Qwen3-32B",
        "GLM-4-9B-0414": "THUDM/GLM-4-9B-0414",
        "MiniMax-M2.5": "MiniMaxAI/MiniMax-M2.5",
        "Qwen3-14B": "Qwen/Qwen3-14B",
    }

    specs = []
    for model_name, model_id in control_models.items():
        for probe in PROBE_LADDER:
            for condition in ["standard", "abstract", "random"]:
                for rep in range(3):
                    specs.append(ExpSpec(
                        instance_id=f"p3_{model_name[:10]}_L{probe.level}_noaxiom_{condition}_r{rep}",
                        model_name=model_name, model_id=model_id, is_local=False,
                        probe_name=probe.name, probe_level=probe.level,
                        strategy="zs", condition=condition,
                        metric=make_minkowski_metric(4), metric_name="minkowski_4", dim=4, rep=rep))

    print(f"Total specs: {len(specs)}")
    return await run_batch(specs, output_path, api_concurrent=12, local_concurrent=0,
                          label="Phase 3: Controls")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], help="Run specific phase")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--output-dir", type=str, default="data/comprehensive_v4")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "█"*70)
    print("█  A(S) COMPREHENSIVE RESEARCH STUDY")
    print("█  15 Models (11 API + 4 Local)")
    print("█  5 Probes × 3 Conditions × 3 Metrics")
    print("█  Target: ~2,500 experiments")
    print("█"*70)

    all_results = []

    if args.phase == 1 or not args.phase:
        results = await phase1_main(output_dir)
        all_results.extend(results or [])

    if args.phase == 2 or not args.phase:
        results = await phase2_strategies(output_dir)
        all_results.extend(results or [])

    if args.phase == 3 or not args.phase:
        results = await phase3_controls(output_dir)
        all_results.extend(results or [])

    # Combine and summarize
    print("\n" + "█"*70)
    print("█  ALL PHASES COMPLETE")
    print("█"*70)

    # Find latest output
    all_jsonls = sorted(output_dir.glob("phase*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for jl in all_jsonls:
        print_summary(str(jl))

    print(f"\nAll results in: {output_dir}")
    print("Run: python run_analysis_pipeline.py --data-dir data/comprehensive_v4")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Capture 2 complete CoT derivations that produce wrong answers for annotation."""
import json, time, re, requests
from pathlib import Path
import sys; sys.path.insert(0, '.')
from experiment_v3.probes import *
from experiment_v3.prompts import build_prompt

API_KEY = "sk-qjhtbnwmzcmvfabtpotvkjxqdkqmnwyzwfdobmkwedopxmfi"
API_BASE = "https://api.siliconflow.cn/v1"

# Target 2 cases likely to produce interesting errors
CASES = [
    {"model_name": "Qwen3-14B", "model_id": "Qwen/Qwen3-14B", "probe_level": 4, "condition": "standard"},
    {"model_name": "Qwen3-32B", "model_id": "Qwen/Qwen3-32B", "probe_level": 3, "condition": "random"},
]

metric = make_minkowski_metric(4)
for case in CASES:
    probe = [p for p in PROBE_LADDER if p.level == case['probe_level']][0]
    bundle = build_prompt(probe, 'cot', case['condition'], metric)
    symbols = get_symbols(case['condition'], 4)
    gt = get_ground_truth(probe, case['condition'], symbols, metric)

    print(f"\n=== {case['model_name']} L{case['probe_level']} {case['condition']} ===")

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": case['model_id'],
               "messages": [{"role":"system","content":bundle.system_prompt},
                            {"role":"user","content":bundle.user_prompt}],
               "max_tokens": 8192, "temperature": 0.0, "stream": False}

    start = time.perf_counter()
    resp = requests.post(f"{API_BASE}/chat/completions", headers=headers,
                        json=payload, timeout=600)
    dur = time.perf_counter() - start

    if resp.status_code == 200:
        data = resp.json()
        output = data["choices"][0]["message"]["content"]
    else:
        output = f"HTTP{resp.status_code}: {resp.text[:200]}"

    # Extract answer
    answer = "?"
    for pat in [r'(?:###?\s*)?FINAL\s*(?:ANSWER|RESULT)\s*:?\s*(.+?)(?:\n\n|\n\s*\n|\n\s*$|$)',
                r'\\boxed\{([^}]+)\}']:
        m = re.search(pat, output, re.DOTALL | re.IGNORECASE)
        if m: answer = m.group(1).strip()[:200]; break

    correct = answers_equivalent(answer, gt) if answer and answer != '?' else False

    fname = f'data/comprehensive_v4/error_annotation_full_{case["model_name"][:15]}_L{case["probe_level"]}_{case["condition"]}.txt'
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(f"Model: {case['model_name']}\n")
        f.write(f"Probe: L{case['probe_level']} {probe.name}\n")
        f.write(f"Condition: {case['condition']}\n")
        f.write(f"Duration: {dur:.0f}s\n")
        f.write(f"Answer correct: {correct}\n")
        f.write(f"Ground Truth: {gt}\n")
        f.write(f"Extracted Answer: {answer}\n")
        f.write(f"\n{'='*60}\nFULL DERIVATION:\n{'='*60}\n")
        f.write(output)

    print(f"Duration: {dur:.0f}s, Output: {len(output)} chars")
    print(f"Correct: {correct}")
    print(f"Answer: [{answer[:150]}]")
    print(f"Saved: {fname}")

print("\nDone. Annotation files ready.")

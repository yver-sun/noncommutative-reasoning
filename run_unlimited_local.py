#!/usr/bin/env python3
"""
Unlimited-Time Local Model Experiment
======================================
Remove timeout constraints on local models. Record latency as informative data.
Tests: Llama3-8B, Qwen2.5-1.5B, Gemma3-4B
Each: 5 probes x 3 conditions x N=3 = 45 experiments
No timeout — let them run to completion naturally.
Records: duration, output length, tokens/sec estimation.
"""
import sys, json, time, re, requests
from pathlib import Path
from datetime import datetime
sys.path.insert(0, '.')
from experiment_v3.probes import *
from experiment_v3.prompts import build_prompt

OLLAMA_BASE = "http://localhost:11434"
MODELS = {
    "Llama3-8B-Local": "llama3:latest",
    "Qwen2.5-1.5B-Local": "qwen2.5:1.5b",
    "Gemma3-4B-Local": "gemma3:latest",
}
# No timeout — let it run as long as needed
REQUEST_TIMEOUT = 3600  # 1 hour max, effectively unlimited
MAX_TOKENS = 16384  # generous output window

def extract_answer(text):
    if not text or len(text.strip()) < 5:
        return ""
    for pat in [r'(?:###?\s*)?FINAL\s*(?:ANSWER|RESULT)\s*:?\s*(.+?)(?:\n\n|\n\s*\n|\n\s*$|$)',
                r'\\boxed\{([^}]+)\}']:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            ans = m.group(1).strip()
            ans = ans.replace('\\(', '').replace('\\)', '').replace('\\[', '').replace('\\]', '')
            ans = re.sub(r'\$+', '', ans)
            return ans.strip()
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if re.match(r'^(Therefore|Thus|Hence|So|The|This|We|Note|Step|Let|I |A |In |By )', line, re.IGNORECASE):
            continue
        line = line.replace('\\(', '').replace('\\)', '')
        if len(line) > 3 and any(c in line for c in '*+-=eEwW'):
            return line.strip()
    return lines[-1].strip() if lines else ""

def run_one(model_name, model_id, probe, condition, metric, rep, output_path, counter, total):
    bundle = build_prompt(probe, 'cot', condition, metric)
    output = ""
    start = time.perf_counter()

    for attempt in range(2):
        try:
            payload = {"model": model_id, "system": bundle.system_prompt,
                       "prompt": bundle.user_prompt, "stream": False,
                       "options": {"temperature": 0.0, "num_predict": MAX_TOKENS}}
            resp = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload,
                                timeout=REQUEST_TIMEOUT)
            dur = time.perf_counter() - start
            if resp.status_code == 200:
                data = resp.json()
                output = data.get("response", "")
                # Record generation metrics
                eval_count = data.get("eval_count", 0)
                eval_dur_ns = data.get("eval_duration", 0)
                tokens_per_sec = eval_count / (eval_dur_ns / 1e9) if eval_dur_ns > 0 else 0
            else:
                output = f"OLLAMA_ERR_{resp.status_code}"
                eval_count = 0
                tokens_per_sec = 0
            break
        except requests.Timeout:
            dur = time.perf_counter() - start
            output = f"NO_COMPLETION_AFTER_{int(dur)}s"
            eval_count = 0
            tokens_per_sec = 0
        except Exception as e:
            dur = time.perf_counter() - start
            output = f"ERR:{str(e)[:100]}"
            eval_count = 0
            tokens_per_sec = 0
            if attempt < 1:
                time.sleep(5)

    is_empty = len(output) < 50 or output.startswith("NO_COMPLETION") or output.startswith("ERR") or output.startswith("OLLAMA_ERR")
    answer = extract_answer(output) if not is_empty else ""
    correct, error_type = None, "empty"
    if not is_empty and answer:
        symbols = get_symbols(condition, 4)
        gt = get_ground_truth(probe, condition, symbols, metric)
        if answers_equivalent(answer, gt):
            correct, error_type = True, "none"
        else:
            na = normalize_answer(answer)
            ng = normalize_ground_truth(gt)
            if ng and ng in na:
                correct, error_type = True, "contains_gt"
            else:
                correct, error_type = False, "algebraic"

    iid = f"v8_unlimited_{model_name[:15]}_L{probe.level}_cot_{condition}_r{rep}"
    record = {
        "instance_id": iid, "model": model_name,
        "probe": probe.name, "probe_level": probe.level,
        "strategy": "cot", "condition": condition,
        "metric_name": "minkowski_4", "dim": 4, "rep": rep,
        "correct": correct, "error_type": error_type, "final_answer": answer,
        "is_empty": is_empty,
        "duration_s": round(dur, 1), "output_len": len(output),
        "generated_tokens": eval_count if 'eval_count' in dir() else 0,
        "tokens_per_sec": round(tokens_per_sec, 2) if 'tokens_per_sec' in dir() else 0,
        "raw_tail": output[-500:] if output else "",
        "unlimited_time": True,
    }

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    status = "OK" if correct else ("ER" if correct is False else "??")
    tps_str = f"{tokens_per_sec:.0f} tok/s" if 'tokens_per_sec' in dir() and tokens_per_sec > 0 else ""
    print(f"[{counter[0]+1:3d}/{total}] {model_name:<22s} L{probe.level} {condition:8s} "
          f"{status} {dur:.0f}s {tps_str}")


def main():
    metric = make_minkowski_metric(4)
    specs = []
    for mname, mid in MODELS.items():
        for probe in PROBE_LADDER:
            for cond in ['standard', 'abstract', 'random']:
                for rep in range(3):
                    specs.append((mname, mid, probe, cond, rep))

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f'data/comprehensive_v4/v8_unlimited_time_{now}.jsonl')
    total = len(specs)
    print(f"Unlimited-Time Local Experiment: {total} experiments")
    print(f"Models: {list(MODELS.keys())}")
    print(f"No timeout — recording full latency as data")
    print(f"Output: {output_path}\n")

    counter = [0]
    start_time = time.perf_counter()

    for mname, mid, probe, cond, rep in specs:
        run_one(mname, mid, probe, cond, metric, rep, output_path, counter, total)
        time.sleep(2)  # brief pause between requests

    elapsed = time.perf_counter() - start_time
    print(f"\nUnlimited-Time DONE: {counter[0]}/{total} in {elapsed/60:.1f}min")
    print(f"Output: {output_path}")


if __name__ == '__main__':
    main()

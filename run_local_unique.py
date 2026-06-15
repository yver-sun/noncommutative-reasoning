#!/usr/bin/env python3
"""
Test UNIQUE local models NOT available on SiliconFlow API.
These provide genuine model diversity: different architectures, ultra-small scale.

Models:
  - qwen2.5:1.5b (986MB, ultra-small, Qwen2.5 family)
  - gemma3:latest (3.3GB, Google architecture)
  - llama3:latest (4.7GB, Meta architecture)

Each: 5 probes × 3 conditions × N=3 = 45 experiments
Total: 135 experiments

Justification for paper:
  These models are NOT available via commercial APIs. They test the lower
  bound of non-commutative algebraic reasoning across different model
  architectures and scales, providing genuine diversity beyond API models.
"""
import sys, json, time, re
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent))
from experiment_v3.probes import *
from experiment_v3.prompts import build_prompt
import requests

OLLAMA_BASE = "http://localhost:11434"

UNIQUE_LOCAL_MODELS = {
    "Qwen2.5-1.5B-Local": "qwen2.5:1.5b",      # 986MB ultra-small
    "Gemma3-4B-Local": "gemma3:latest",          # Google architecture
    "Llama3-8B-Local": "llama3:latest",          # Meta architecture
}

def extract_answer(text):
    if not text or len(text.strip()) < 5:
        return ""
    for pat in [r'(?:###?\s*)?FINAL\s*(?:ANSWER|RESULT)\s*:?\s*(.+?)(?:\n\n|\n\s*\n|\n\s*$|$)',
                r'\\boxed\{([^}]+)\}']:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            ans = m.group(1).strip()
            ans = ans.replace('\\(', '').replace('\\)', '')
            ans = ans.replace('\\[', '').replace('\\]', '')
            ans = re.sub(r'\$+', '', ans)
            return ans.strip()
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if re.match(r'^(Therefore|Thus|Hence|So|The|This|We|Note|Step|Let|I |A |In |By |Using|Now|First|Second)', line, re.IGNORECASE):
            continue
        line = line.replace('\\(', '').replace('\\)', '')
        if len(line) > 3 and any(c in line for c in '*+-=eEwW'):
            return line.strip()
    return lines[-1].strip() if lines else ""


def run_one(model_name, model_id, probe, condition, metric, rep, output_path, counter, total):
    bundle = build_prompt(probe, 'cot', condition, metric)
    output, dur = "", 0.0

    for attempt in range(3):
        try:
            payload = {"model": model_id, "system": bundle.system_prompt,
                       "prompt": bundle.user_prompt, "stream": False,
                       "options": {"temperature": 0.0, "num_predict": 4096}}
            start = time.perf_counter()
            resp = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=600)
            dur = time.perf_counter() - start
            if resp.status_code == 200:
                output = resp.json().get("response", "")
            else:
                output = f"O{resp.status_code}"
            if output and not output.startswith("O"):
                break
            if attempt < 2:
                time.sleep(2)
        except requests.Timeout:
            output = "TIMEOUT_600s"; dur = 600
            if attempt < 2:
                time.sleep(3)
        except Exception as e:
            output = f"ERR:{str(e)[:100]}"
            if attempt < 2:
                time.sleep(3)

    is_empty = len(output) < 50 or output.startswith("TIMEOUT")
    answer = extract_answer(output) if not is_empty else ""
    correct, error_type, eval_method = None, "empty", "empty"
    if not is_empty and answer:
        symbols = get_symbols(condition, 4)
        gt = get_ground_truth(probe, condition, symbols, metric)
        if answers_equivalent(answer, gt):
            correct, error_type, eval_method = True, "none", "semantic_equivalent"
        else:
            na = normalize_answer(answer)
            ng = normalize_ground_truth(gt)
            if ng and ng in na:
                correct, error_type, eval_method = True, "none", "contains_gt"
            else:
                correct, error_type, eval_method = False, "algebraic", "mismatch"

    iid = f"v7_local_unique_{model_name[:15]}_L{probe.level}_cot_{condition}_r{rep}"
    record = {
        "instance_id": iid, "model": model_name,
        "probe": probe.name, "probe_level": probe.level,
        "strategy": "cot", "condition": condition,
        "metric_name": "minkowski_4", "dim": 4, "rep": rep,
        "correct": correct, "error_type": error_type, "final_answer": answer,
        "is_empty": is_empty, "evaluation_method": eval_method,
        "duration_s": round(dur, 1), "output_len": len(output),
        "raw_tail": output[-300:] if output else "",
        "local_unique": True,  # Marks as unique local model
    }

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    status = "OK" if correct else ("??" if correct is None else "ER")
    if is_empty: status = "EM"
    print(f"[{counter[0]+1:3d}/{total}] {model_name:<20s} L{probe.level} "
          f"{condition:8s} {status} {dur:.0f}s")


def main():
    metric = make_minkowski_metric(4)
    specs = []
    for mname, mid in UNIQUE_LOCAL_MODELS.items():
        for probe in PROBE_LADDER:
            for cond in ['standard', 'abstract', 'random']:
                for rep in range(3):
                    specs.append((mname, mid, probe, cond, rep))

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f'data/comprehensive_v4/v7_local_unique_{now}.jsonl')
    total = len(specs)

    print(f"Unique Local Models: {total} experiments")
    for mname in UNIQUE_LOCAL_MODELS:
        print(f"  {mname} ({UNIQUE_LOCAL_MODELS[mname]})")
    print(f"Output: {output_path}\n")

    counter = [0]
    start_time = time.perf_counter()

    for mname, mid, probe, cond, rep in specs:
        run_one(mname, mid, probe, cond, metric, rep, output_path, counter, total)

    elapsed = time.perf_counter() - start_time
    print(f"\nUnique Local DONE: {counter[0]}/{total} in {elapsed/60:.1f}min")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

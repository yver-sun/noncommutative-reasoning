#!/usr/bin/env python3
"""Redo Qwen3-8B-Local experiments (Ollama is back online)."""
import sys, json, time, asyncio, aiohttp, re
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent))
from experiment_v3.probes import *
from experiment_v3.prompts import build_prompt

OLLAMA_BASE = "http://localhost:11434"
MODEL_ID = "qwen3:8b"
MODEL_NAME = "Qwen3-8B-Local"

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

async def run_one(session, spec, output_path, counter, start_time):
    probe = [p for p in PROBE_LADDER if p.level == spec['probe_level']][0]
    bundle = build_prompt(probe, spec['strategy'], spec['condition'], spec['metric'])
    output, dur = "", 0.0

    for attempt in range(3):
        try:
            payload = {"model": MODEL_ID, "system": bundle.system_prompt,
                       "prompt": bundle.user_prompt, "stream": False,
                       "options": {"temperature": 0.0, "num_predict": 4096}}
            start = time.perf_counter()
            async with session.post(f"{OLLAMA_BASE}/api/generate", json=payload,
                                    timeout=aiohttp.ClientTimeout(total=300)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    output = data.get("response", "")
                else:
                    output = f"O{resp.status}"
            dur = time.perf_counter() - start
            if output and not output.startswith("O") and output != "TIMEOUT":
                break
            if attempt < 2:
                await asyncio.sleep(2)
        except asyncio.TimeoutError:
            output = "TIMEOUT"; dur = 300
            if attempt < 2:
                await asyncio.sleep(3)
        except Exception as e:
            output = f"ERR:{str(e)[:80]}"
            if attempt < 2:
                await asyncio.sleep(3)

    is_empty = len(output) < 50
    answer = extract_answer(output) if not is_empty else ""
    correct, error_type, eval_method = None, "empty", "empty"
    if not is_empty and answer:
        symbols = get_symbols(spec['condition'], 4)
        gt = get_ground_truth(probe, spec['condition'], symbols, spec['metric'])
        if answers_equivalent(answer, gt):
            correct, error_type, eval_method = True, "none", "semantic_equivalent"
        else:
            na = normalize_answer(answer)
            ng = normalize_ground_truth(gt)
            if ng and ng in na:
                correct, error_type, eval_method = True, "none", "contains_gt"
            else:
                correct, error_type, eval_method = False, "algebraic", "mismatch"

    record = {
        "instance_id": spec['instance_id'], "model": MODEL_NAME,
        "probe": spec['probe_name'], "probe_level": spec['probe_level'],
        "strategy": spec['strategy'], "condition": spec['condition'],
        "metric_name": spec['metric_name'], "dim": spec['dim'], "rep": spec['rep'],
        "correct": correct, "error_type": error_type, "final_answer": answer,
        "is_empty": is_empty, "evaluation_method": eval_method,
        "duration_s": round(dur, 1), "output_len": len(output),
        "raw_tail": output[-300:] if output else "",
        "retry": True,  # Marks this as a retry of failed v7
    }

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    counter[0] += 1
    elapsed = time.perf_counter() - start_time
    status = "OK" if correct else ("??" if correct is None else "ER")
    if is_empty: status = "EM"
    print(f"[{counter[0]:3d}] Qwen3-8B-Local L{spec['probe_level']} "
          f"{spec['condition']:8s} {status} {dur:.0f}s")
    return record

async def main():
    metric = make_minkowski_metric(4)
    specs = []
    for probe in PROBE_LADDER:
        for cond in ['standard', 'abstract', 'random']:
            for rep in range(2, 5):
                specs.append({
                    'instance_id': f"v7b_{MODEL_NAME[:15]}_L{probe.level}_cot_{cond}_r{rep}",
                    'probe_name': probe.name, 'probe_level': probe.level,
                    'strategy': 'cot', 'condition': cond, 'metric': metric,
                    'metric_name': 'minkowski_4', 'dim': 4, 'rep': rep,
                })

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f'data/comprehensive_v4/v7_local_retry_{now}.jsonl')
    print(f"Redo Qwen3-8B-Local: {len(specs)} experiments (N=3)")
    print(f"Output: {output_path}\n")

    counter = [0]
    start_time = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        # Local model — run 1 at a time
        for spec in specs:
            await run_one(session, spec, output_path, counter, start_time)

    elapsed = time.perf_counter() - start_time
    valid = [1 for _ in range(counter[0])]  # all are valid records
    print(f"\nLocal redo DONE: {counter[0]}/{len(specs)} in {elapsed/60:.1f}min")
    print(f"Output: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())

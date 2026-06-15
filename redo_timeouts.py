#!/usr/bin/env python3
"""
Redo timeout experiments with extended timeout (600s) and larger max_tokens (8192).
Targets: DeepSeek-R1 (38), Qwen3-32B (26), Qwen3-8B (65) = 129 total.
R1 generates extremely long CoT — 4096 tokens + 300s is insufficient.
"""
import sys, json, time, asyncio, aiohttp, re
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent))
from experiment_v3.probes import *
from experiment_v3.prompts import build_prompt

API_KEYS = [
    "sk-qjhtbnwmzcmvfabtpotvkjxqdkqmnwyzwfdobmkwedopxmfi",
    "sk-wzijtyuyivphmvydtuakbfbgerunbsfqyqwzdvphbvguogsc",
    "sk-gzuafdpnjrrtdsacgotbwlytepqrkbzfdutvcokoufkoabeq",
    "sk-ktslkqulxiinxnxrevqshvyvwpmkucsdjxmkjibdbwafgyvv",
]
API_BASE = "https://api.siliconflow.cn/v1"

API_MODELS = {
    "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    "Qwen3-32B": "Qwen/Qwen3-32B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
}

TIMEOUT = 600  # seconds (was 300)
MAX_TOKENS = 8192  # was 4096

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


async def run_one(session, api_key, spec, sem, output_path, counter, start_time):
    async with sem:
        probe = [p for p in PROBE_LADDER if p.level == spec['probe_level']][0]
        bundle = build_prompt(probe, spec['strategy'], spec['condition'], spec['metric'])
        output, dur = "", 0.0

        for attempt in range(2):  # Fewer retries since we have more timeout headroom
            try:
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {"model": spec['model_id'],
                           "messages": [{"role": "system", "content": bundle.system_prompt},
                                        {"role": "user", "content": bundle.user_prompt}],
                           "max_tokens": MAX_TOKENS, "temperature": 0.0, "stream": False}
                start = time.perf_counter()
                async with session.post(f"{API_BASE}/chat/completions", headers=headers,
                                        json=payload, timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        output = data["choices"][0]["message"]["content"]
                        # Check if truncated (output near max_tokens)
                        if 'usage' in data:
                            completion_tokens = data['usage'].get('completion_tokens', 0)
                            if completion_tokens >= MAX_TOKENS - 10:
                                output = output + f"\n[TRUNCATED: {completion_tokens} tokens]"
                    else:
                        output = f"H{resp.status}:{await resp.text()[:100]}"
                dur = time.perf_counter() - start
                if output and not output.startswith("H") and not output.startswith("ERR"):
                    break
                if attempt < 1:
                    await asyncio.sleep(2)
            except asyncio.TimeoutError:
                output = f"TIMEOUT_{TIMEOUT}s"
                dur = TIMEOUT
                if attempt < 1:
                    await asyncio.sleep(3)
            except Exception as e:
                output = f"ERR:{str(e)[:100]}"
                if attempt < 1:
                    await asyncio.sleep(2)

        is_empty = len(output) < 50 or output.startswith("TIMEOUT") or output.startswith("H")
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

        # Mark improvement from original
        orig_was_timeout = spec.get('orig_timeout', False)

        record = {
            "instance_id": spec['instance_id'], "model": spec['model_name'],
            "probe": spec['probe_name'], "probe_level": spec['probe_level'],
            "strategy": spec['strategy'], "condition": spec['condition'],
            "metric_name": spec['metric_name'], "dim": spec['dim'], "rep": spec['rep'],
            "correct": correct, "error_type": error_type, "final_answer": answer,
            "is_empty": is_empty, "evaluation_method": eval_method,
            "duration_s": round(dur, 1), "output_len": len(output),
            "raw_tail": output[-400:] if output else "",
            "timeout_retry": True,  # Marks as 600s retry
            "orig_timeout_300s": orig_was_timeout,
        }

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        counter[0] += 1
        elapsed = time.perf_counter() - start_time
        status = "OK" if correct else ("??" if correct is None else "ER")
        if is_empty: status = "TO" if "TIMEOUT" in output else "EM"
        print(f"[{counter[0]:4d}] {spec['model_name'][:18]:18s} L{spec['probe_level']} "
              f"{spec['condition']:8s} r{spec['rep']} {status} {dur:.0f}s")

        return record


async def main():
    # Load existing v7 data
    data_path = Path('data/comprehensive_v4/v7_expanded_20260614_131740.jsonl')
    all_v7 = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_v7.append(json.loads(line))

    # Find timeout experiments from ALL models with timeouts
    # Phase A has completed for R1, Qwen3-32B, and Qwen3-8B
    timeouts = [r for r in all_v7
                if r.get('duration_s', 0) >= 290
                and r['model'] in ('DeepSeek-R1', 'Qwen3-32B', 'Qwen3-8B')]

    if not timeouts:
        print("No timeout experiments found to redo!")
        return

    # Also check if we already have retry data for these
    existing_retries = set()
    for f in Path('data/comprehensive_v4').glob('v7_timeout_retry_*.jsonl'):
        with open(f, 'r', encoding='utf-8') as fh:
            for line in fh:
                if line.strip():
                    existing_retries.add(json.loads(line).get('instance_id', ''))

    # Build specs
    specs = []
    skipped = 0
    for r in timeouts:
        iid = r['instance_id']
        if iid in existing_retries:
            skipped += 1
            continue
        mname = r['model']
        if mname not in API_MODELS:
            continue
        specs.append({
            'instance_id': iid,
            'model_name': mname,
            'model_id': API_MODELS[mname],
            'probe_name': r['probe'],
            'probe_level': r['probe_level'],
            'strategy': r.get('strategy', 'cot'),
            'condition': r['condition'],
            'metric': make_minkowski_metric(r.get('dim', 4)),
            'metric_name': r.get('metric_name', 'minkowski_4'),
            'dim': r.get('dim', 4),
            'rep': r['rep'],
            'orig_timeout': True,
        })

    if skipped:
        print(f"Skipping {skipped} already-retried experiments")

    if not specs:
        print("All timeout experiments already retried!")
        return

    # Check Phase A status — skip Qwen3-8B if Phase A is still running
    models_in_specs = set(s['model_name'] for s in specs)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f'data/comprehensive_v4/v7_timeout_retry_{now}.jsonl')
    total = len(specs)

    print(f"Timeout Retry: {total} experiments")
    print(f"  Models: {models_in_specs}")
    print(f"  Timeout: {TIMEOUT}s, Max Tokens: {MAX_TOKENS}")
    print(f"  Output: {output_path}\n")

    # Use 8 concurrent (share with Phase A's 16)
    sem = asyncio.Semaphore(8)
    counter = [0]
    start_time = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        tasks = []
        for s in specs:
            ki = hash(s['model_name']) % 4
            tasks.append(run_one(session, API_KEYS[ki], s, sem, output_path, counter, start_time))
        results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.perf_counter() - start_time
    valid = [r for r in results if isinstance(r, dict)]
    ok = sum(1 for r in valid if r.get('correct') is True)
    recovered = sum(1 for r in valid if r.get('correct') is True and r.get('orig_timeout_300s'))
    still_timeout = sum(1 for r in valid if r.get('is_empty') and 'TIMEOUT' in r.get('raw_tail',''))

    print(f"\nTimeout Retry DONE: {len(valid)}/{total} in {elapsed/60:.1f}min")
    print(f"  Recovered (was timeout, now OK): {recovered}")
    print(f"  Still timeout at 600s: {still_timeout}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())

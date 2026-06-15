#!/usr/bin/env python3
"""Phase A: Expand N from 3 to 10 for all 12 models × 5 probes × 3 conditions."""
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
OLLAMA_BASE = "http://localhost:11434"

API_MODELS = {
    "DeepSeek-V4-Pro": "deepseek-ai/DeepSeek-V4-Pro",
    "DeepSeek-V3.2": "deepseek-ai/DeepSeek-V3.2",
    "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    "DeepSeek-V4-Flash": "deepseek-ai/DeepSeek-V4-Flash",
    "DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
    "Qwen3-32B": "Qwen/Qwen3-32B",
    "Qwen3-14B": "Qwen/Qwen3-14B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "GLM-4-9B-0414": "THUDM/GLM-4-9B-0414",
    "GLM-Z1-9B-0414": "THUDM/GLM-Z1-9B-0414",
    "MiniMax-M2.5": "MiniMaxAI/MiniMax-M2.5",
}
LOCAL_MODELS = {"Qwen3-8B-Local": "qwen3:8b"}

def extract_answer(text):
    if not text or len(text.strip()) < 5:
        return ""
    for pat in [r'(?:###?\s*)?FINAL\s*ANSWER\s*:?\s*(.+?)(?:\n\n|\n\s*\n|\n\s*$|$)',
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
        is_local = spec['is_local']
        output, dur = "", 0.0

        for attempt in range(3):
            try:
                if is_local:
                    payload = {"model": spec['model_id'], "system": bundle.system_prompt,
                               "prompt": bundle.user_prompt, "stream": False,
                               "options": {"temperature": 0.0, "num_predict": 4096}}
                    start = time.perf_counter()
                    async with session.post(f"{OLLAMA_BASE}/api/generate", json=payload,
                                            timeout=aiohttp.ClientTimeout(total=300)) as resp:
                        output = (await resp.json()).get("response", "") if resp.status == 200 else f"O{resp.status}"
                    dur = time.perf_counter() - start
                else:
                    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                    payload = {"model": spec['model_id'],
                               "messages": [{"role": "system", "content": bundle.system_prompt},
                                            {"role": "user", "content": bundle.user_prompt}],
                               "max_tokens": 4096, "temperature": 0.0, "stream": False}
                    start = time.perf_counter()
                    async with session.post(f"{API_BASE}/chat/completions", headers=headers,
                                            json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                        if resp.status == 200:
                            output = (await resp.json())["choices"][0]["message"]["content"]
                        else:
                            output = f"H{resp.status}"
                    dur = time.perf_counter() - start
                if not output.startswith("H") and not output.startswith("O") and output != "TIMEOUT":
                    break
                if attempt < 2:
                    await asyncio.sleep(1)
            except asyncio.TimeoutError:
                output = "TIMEOUT"; dur = 300
                if attempt < 2:
                    await asyncio.sleep(2)
            except Exception as e:
                output = f"ERR:{str(e)[:80]}"
                if attempt < 2:
                    await asyncio.sleep(1)

        is_empty = len(output) < 50
        answer = extract_answer(output) if not is_empty else ""
        correct, error_type, eval_method = None, "empty", "empty"
        if not is_empty:
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
                    correct, error_type, eval_method = None, "uncertain", "uncertain"

        record = {
            "instance_id": spec['instance_id'], "model": spec['model_name'],
            "probe": spec['probe_name'], "probe_level": spec['probe_level'],
            "strategy": spec['strategy'], "condition": spec['condition'],
            "metric_name": spec['metric_name'], "dim": spec['dim'], "rep": spec['rep'],
            "correct": correct, "error_type": error_type, "final_answer": answer,
            "is_empty": is_empty, "evaluation_method": eval_method,
            "duration_s": round(dur, 1), "output_len": len(output),
            "raw_tail": output[-300:] if output else "",
        }

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        counter[0] += 1
        elapsed = time.perf_counter() - start_time
        rate = counter[0] / (elapsed / 3600) if elapsed > 0 else 0
        status = "OK" if correct else ("??" if correct is None else "ER")
        if is_empty:
            status = "EM"
        print(f"[{counter[0]:4d}] {spec['model_name'][:18]:18s} L{spec['probe_level']} "
              f"{spec['condition']:8s} {status} {dur:.0f}s")

        return record


async def main():
    # Load existing IDs
    data_path = Path('data/comprehensive_v4/v6_20260613_201800.jsonl')
    existing_ids = set()
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                existing_ids.add(json.loads(line)['instance_id'])

    # Generate specs: N=3→10 for API, N=2→5 for local
    specs = []
    for mname, mid in API_MODELS.items():
        for probe in PROBE_LADDER:
            for cond in ['standard', 'abstract', 'random']:
                for rep in range(3, 10):
                    iid = f"v7_{mname[:15]}_L{probe.level}_cot_{cond}_r{rep}"
                    if iid not in existing_ids:
                        specs.append({
                            'instance_id': iid, 'model_name': mname, 'model_id': mid,
                            'is_local': False, 'probe_name': probe.name,
                            'probe_level': probe.level, 'strategy': 'cot',
                            'condition': cond, 'metric': make_minkowski_metric(4),
                            'metric_name': 'minkowski_4', 'dim': 4, 'rep': rep,
                        })

    for mname, mid in LOCAL_MODELS.items():
        for probe in PROBE_LADDER:
            for cond in ['standard', 'abstract', 'random']:
                for rep in range(2, 5):
                    iid = f"v7_{mname[:15]}_L{probe.level}_cot_{cond}_r{rep}"
                    if iid not in existing_ids:
                        specs.append({
                            'instance_id': iid, 'model_name': mname, 'model_id': mid,
                            'is_local': True, 'probe_name': probe.name,
                            'probe_level': probe.level, 'strategy': 'cot',
                            'condition': cond, 'metric': make_minkowski_metric(4),
                            'metric_name': 'minkowski_4', 'dim': 4, 'rep': rep,
                        })

    output_dir = Path('data/comprehensive_v4')
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"v7_expanded_{now}.jsonl"
    total = len(specs)

    n_api = sum(1 for s in specs if not s['is_local'])
    n_local = sum(1 for s in specs if s['is_local'])
    print(f"Phase A: {total} experiments ({n_api} API + {n_local} local)")
    print(f"Target: N=10 API, N=5 local")
    print(f"Output: {output_path}\n")

    api_sem = asyncio.Semaphore(16)
    local_sem = asyncio.Semaphore(1)
    counter = [0]
    start_time = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        tasks = []
        for s in specs:
            ki = hash(s['model_name']) % 4 if not s['is_local'] else 0
            sem = local_sem if s['is_local'] else api_sem
            tasks.append(run_one(session, API_KEYS[ki], s, sem, output_path, counter, start_time))
        results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.perf_counter() - start_time
    valid = [r for r in results if isinstance(r, dict)]
    ok = sum(1 for r in valid if r.get('correct') is True)
    print(f"\nDONE: {len(valid)}/{total} in {elapsed/3600:.1f}h, OK={ok}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())

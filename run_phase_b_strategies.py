#!/usr/bin/env python3
"""
Phase B: Strategy Comparison Experiment
========================================
5 strategies × 5 models × L2-L4 probes × 3 conditions × N=3 = 675 experiments

Strategies:
  1. ZS (Zero-Shot): Axioms + direct question, no guidance
  2. CoT (Chain-of-Thought): Step-by-step reasoning guidance
  3. CoT+SC@5 (Self-Consistency): CoT × 5 samples, majority vote
  4. CoT+PoT (Program-of-Thought): Code-style algebraic reasoning
  5. CoT+Self-Verify: CoT + explicit verification phase

Models (spanning performance range):
  - DeepSeek-V4-Pro (top performer)
  - DeepSeek-V3.2 (strong)
  - Qwen3-32B (good)
  - GLM-4-9B-0414 (weak)
  - MiniMax-M2.5 (strong)

Probes: L2, L3, L4 only (L0-L1 have ceiling effects)
"""
import sys, json, time, asyncio, aiohttp, re
from pathlib import Path
from datetime import datetime
from collections import Counter
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

# 5 representative models spanning performance range
PHASE_B_MODELS = {
    "DeepSeek-V4-Pro": "deepseek-ai/DeepSeek-V4-Pro",
    "DeepSeek-V3.2": "deepseek-ai/DeepSeek-V3.2",
    "Qwen3-32B": "Qwen/Qwen3-32B",
    "GLM-4-9B-0414": "THUDM/GLM-4-9B-0414",
    "MiniMax-M2.5": "MiniMaxAI/MiniMax-M2.5",
}

STRATEGIES = ["zs", "cot", "cot_sc", "cot_pot", "cot_sv"]
PROBES = [p for p in PROBE_LADDER if p.level in [2, 3, 4]]  # L2, L3, L4 only

SC_SAMPLES = 5  # Self-consistency samples per experiment

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
    """Run a single experiment. For cot_sc, runs SC_SAMPLES times and takes majority."""
    async with sem:
        probe = [p for p in PROBE_LADDER if p.level == spec['probe_level']][0]
        strategy = spec['strategy']
        metric = spec['metric']

        if strategy == 'cot_sc':
            # Self-consistency: run N times with temperature=0.7, majority vote
            answers = []
            durations = []
            for sc_i in range(SC_SAMPLES):
                bundle = build_prompt(probe, 'cot_sc', spec['condition'], metric)
                output, dur = "", 0.0
                for attempt in range(2):
                    try:
                        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                        payload = {"model": spec['model_id'],
                                   "messages": [{"role": "system", "content": bundle.system_prompt},
                                                {"role": "user", "content": bundle.user_prompt}],
                                   "max_tokens": 4096, "temperature": 0.7, "stream": False}
                        start = time.perf_counter()
                        async with session.post(f"{API_BASE}/chat/completions", headers=headers,
                                                json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                            if resp.status == 200:
                                output = (await resp.json())["choices"][0]["message"]["content"]
                            else:
                                output = f"H{resp.status}"
                        dur = time.perf_counter() - start
                        if not output.startswith("H"):
                            break
                        if attempt < 1:
                            await asyncio.sleep(1)
                    except asyncio.TimeoutError:
                        output = "TIMEOUT"; dur = 300
                    except Exception as e:
                        output = f"ERR:{str(e)[:80]}"
                if output and not output.startswith("H") and output != "TIMEOUT":
                    ans = extract_answer(output)
                    if ans:
                        answers.append(ans)
                durations.append(dur)

            # Majority vote
            if answers:
                answer = Counter(answers).most_common(1)[0][0]
                sc_agreement = answers.count(answer) / len(answers)
            else:
                answer = ""
                sc_agreement = 0.0
            dur = sum(durations)
            output = f"[SC@{SC_SAMPLES}: {len(answers)}/{SC_SAMPLES} valid, agreement={sc_agreement:.2f}] " + (answers[0][:200] if answers else "EMPTY")
        else:
            # Single-shot strategy
            bundle = build_prompt(probe, strategy, spec['condition'], metric)
            output, dur = "", 0.0
            temperature = 0.0
            for attempt in range(3):
                try:
                    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                    payload = {"model": spec['model_id'],
                               "messages": [{"role": "system", "content": bundle.system_prompt},
                                            {"role": "user", "content": bundle.user_prompt}],
                               "max_tokens": 4096, "temperature": temperature, "stream": False}
                    start = time.perf_counter()
                    async with session.post(f"{API_BASE}/chat/completions", headers=headers,
                                            json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                        if resp.status == 200:
                            output = (await resp.json())["choices"][0]["message"]["content"]
                        else:
                            output = f"H{resp.status}"
                    dur = time.perf_counter() - start
                    if not output.startswith("H") and output != "TIMEOUT":
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
            answer = extract_answer(output)

        is_empty = len(output) < 50
        answer = extract_answer(output) if not is_empty else ""
        correct, error_type, eval_method = None, "empty", "empty"
        if not is_empty and answer:
            symbols = get_symbols(spec['condition'], 4)
            gt = get_ground_truth(probe, spec['condition'], symbols, metric)
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
            "instance_id": spec['instance_id'], "model": spec['model_name'],
            "probe": spec['probe_name'], "probe_level": spec['probe_level'],
            "strategy": spec['strategy'], "condition": spec['condition'],
            "metric_name": spec['metric_name'], "dim": spec['dim'], "rep": spec['rep'],
            "correct": correct, "error_type": error_type, "final_answer": answer,
            "is_empty": is_empty, "evaluation_method": eval_method,
            "duration_s": round(dur, 1), "output_len": len(output),
            "raw_tail": output[-300:] if output else "",
        }
        if strategy == 'cot_sc':
            record['sc_samples'] = SC_SAMPLES
            record['sc_agreement'] = sc_agreement if answers else 0.0

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        counter[0] += 1
        elapsed = time.perf_counter() - start_time
        status = "OK" if correct else ("??" if correct is None else "ER")
        if is_empty: status = "EM"
        print(f"[{counter[0]:4d}] {spec['model_name'][:18]:18s} {spec['strategy']:8s} "
              f"L{spec['probe_level']} {spec['condition']:8s} {status} {dur:.0f}s")

        return record


async def main():
    # Check for existing Phase B data to avoid duplicates
    existing_ids = set()
    data_dir = Path('data/comprehensive_v4')
    for f in data_dir.glob('phase_b_*.jsonl'):
        with open(f, 'r', encoding='utf-8') as fh:
            for line in fh:
                if line.strip():
                    existing_ids.add(json.loads(line).get('instance_id', ''))

    # Generate specs
    metric = make_minkowski_metric(4)
    specs = []
    for mname, mid in PHASE_B_MODELS.items():
        for strategy in STRATEGIES:
            for probe in PROBES:
                for cond in ['standard', 'abstract', 'random']:
                    for rep in range(3):
                        iid = f"phaseB_{mname[:15]}_{strategy}_L{probe.level}_{cond}_r{rep}"
                        if iid not in existing_ids:
                            specs.append({
                                'instance_id': iid,
                                'model_name': mname, 'model_id': mid,
                                'is_local': False, 'probe_name': probe.name,
                                'probe_level': probe.level, 'strategy': strategy,
                                'condition': cond, 'metric': metric,
                                'metric_name': 'minkowski_4', 'dim': 4, 'rep': rep,
                            })

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = data_dir / f"phase_b_strategies_{now}.jsonl"
    total = len(specs)

    # Count sc experiments
    n_sc = sum(1 for s in specs if s['strategy'] == 'cot_sc')
    n_single = total - n_sc
    est_calls = n_single + n_sc * SC_SAMPLES
    print(f"Phase B: {total} experiments × up to {SC_SAMPLES} SC samples = ~{est_calls} API calls")
    print(f"  Models: {list(PHASE_B_MODELS.keys())}")
    print(f"  Strategies: {STRATEGIES}")
    print(f"  Probes: L2, L3, L4")
    print(f"  Conditions × N=3")
    print(f"Output: {output_path}\n")

    sem = asyncio.Semaphore(16)
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
    print(f"\nPhase B DONE: {len(valid)}/{total} in {elapsed/3600:.1f}h, OK={ok}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())

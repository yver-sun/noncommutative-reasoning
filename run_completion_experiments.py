#!/usr/bin/env python3
"""
Completion Experiments: Fill all remaining gaps.
=================================================
1. Phase B completion: GLM-4-9B-0414 + MiniMax-M2.5 (2 models x 5 strats x 3 probes x 3 cond x 3 rep = 270)
2. Qwen3-8B 600s retry: remaining 11 timeout experiments
3. No-axiom ablation extension: Qwen3-8B + Llama3-8B (2 models x 5 probes x 3 cond x 3 rep = 90)
Total: ~371 experiments, ~40 min at 16 concurrent
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
TIMEOUT = 600
MAX_TOKENS = 8192

# ── Phase B completion models ──
PHASE_B_MODELS = {
    "GLM-4-9B-0414": "THUDM/GLM-4-9B-0414",
    "MiniMax-M2.5": "MiniMaxAI/MiniMax-M2.5",
}
STRATEGIES = ["zs", "cot", "cot_sc", "cot_pot", "cot_sv"]
PROBES_L234 = [p for p in PROBE_LADDER if p.level in [2,3,4]]

# ── Qwen3-8B timeout retry ──
QWEN8B = {"Qwen3-8B": "Qwen/Qwen3-8B"}

# ── No-axiom extension ──
NOAXIOM_MODELS = {
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Llama3-8B-Local": None,  # Skip local, already in unlimited experiment
}

SC_SAMPLES = 5

def extract_answer(text):
    if not text or len(text.strip()) < 5: return ""
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
        if re.match(r'^(Therefore|Thus|Hence|So|The|This|We|Note|Step|Let|I |A |In |By )', line, re.IGNORECASE): continue
        line = line.replace('\\(', '').replace('\\)', '')
        if len(line) > 3 and any(c in line for c in '*+-=eEwW'): return line.strip()
    return lines[-1].strip() if lines else ""

async def run_one(session, api_key, spec, sem, output_path, counter, start_time):
    async with sem:
        probe = [p for p in PROBE_LADDER if p.level == spec['probe_level']][0]
        strategy = spec.get('strategy', 'cot')
        condition = spec['condition']
        metric = spec['metric']
        output, dur = "", 0.0
        is_noaxiom = spec.get('no_axiom', False)

        if is_noaxiom:
            # Build no-axiom prompt
            from experiment_v3.prompts import get_question, get_symbols
            symbols = get_symbols(condition, 4)
            dim = len(metric)
            basis_list = ", ".join(symbols[i] for i in range(1, dim+1))
            lam_lines = "\n".join(f"- Λ({i}) = {v}" for i, v in sorted(metric.items()))
            if condition == "standard":
                system = f"You are a mathematical assistant. Compute the following expressions using standard mathematical rules.\nBasis elements: {basis_list}\nMetric: {lam_lines}\nProvide your complete derivation and final answer."
            elif condition == "abstract":
                system = f"You are a mathematical assistant. Compute the following expressions.\nSymbols: {basis_list}\nNote: ω_i * ω_i = Λ(i) where:\n{lam_lines}\nProvide your complete derivation and final answer."
            else:
                system = f"You are a mathematical assistant. Compute the following expressions.\nSymbols: {basis_list}\nS_i * S_i = Λ(i) where:\n{lam_lines}\nProvide your complete derivation and final answer."
            question = get_question(probe, condition, symbols, metric)
            user = f"{question}\n\nShow your derivation step-by-step.\nOutput your final answer as: FINAL ANSWER: <your answer>"
        elif strategy == 'cot_sc':
            answers = []
            for sc_i in range(SC_SAMPLES):
                bundle = build_prompt(probe, 'cot_sc', condition, metric)
                for attempt in range(2):
                    try:
                        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                        payload = {"model": spec['model_id'],
                                   "messages": [{"role":"system","content":bundle.system_prompt},
                                                {"role":"user","content":bundle.user_prompt}],
                                   "max_tokens": MAX_TOKENS, "temperature": 0.7, "stream": False}
                        t0 = time.perf_counter()
                        async with session.post(f"{API_BASE}/chat/completions", headers=headers,
                                                json=payload, timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as resp:
                            if resp.status == 200:
                                out = (await resp.json())["choices"][0]["message"]["content"]
                            else: out = f"H{resp.status}"
                        if not out.startswith("H"): break
                    except: out = f"ERR"
                if out and not out.startswith("H") and not out.startswith("ERR"):
                    ans = extract_answer(out)
                    if ans: answers.append(ans)
                dur += time.perf_counter() - t0 if 't0' in dir() else 0
            answer = Counter(answers).most_common(1)[0][0] if answers else ""
            output = f"[SC@{SC_SAMPLES}] " + (answers[0][:200] if answers else "EMPTY")
        else:
            bundle = build_prompt(probe, strategy, condition, metric)
            for attempt in range(3):
                try:
                    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                    payload = {"model": spec['model_id'],
                               "messages": [{"role":"system","content":bundle.system_prompt},
                                            {"role":"user","content":bundle.user_prompt}],
                               "max_tokens": MAX_TOKENS, "temperature": 0.0, "stream": False}
                    t0 = time.perf_counter()
                    async with session.post(f"{API_BASE}/chat/completions", headers=headers,
                                            json=payload, timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as resp:
                        if resp.status == 200:
                            output = (await resp.json())["choices"][0]["message"]["content"]
                        else: output = f"H{resp.status}"
                    dur = time.perf_counter() - t0
                    if not output.startswith("H"): break
                    if attempt < 2: await asyncio.sleep(1)
                except asyncio.TimeoutError:
                    output = f"TIMEOUT_{TIMEOUT}s"; dur = TIMEOUT
                    if attempt < 2: await asyncio.sleep(2)
                except Exception as e:
                    output = f"ERR:{str(e)[:80]}"
                    if attempt < 2: await asyncio.sleep(1)
            if strategy != 'cot_sc':
                answer = extract_answer(output)

        is_empty = len(output) < 50 or output.startswith("TIMEOUT")
        answer = extract_answer(output) if not is_empty else ""
        correct, error_type = None, "empty"
        if not is_empty and answer:
            symbols = get_symbols(condition, 4)
            gt = get_ground_truth(probe, condition, symbols, metric)
            if answers_equivalent(answer, gt):
                correct, error_type = True, "none"
            else:
                na = normalize_answer(answer); ng = normalize_ground_truth(gt)
                if ng and ng in na: correct, error_type = True, "contains_gt"
                else: correct, error_type = False, "algebraic"

        record = {
            "instance_id": spec['instance_id'], "model": spec['model_name'],
            "probe": spec['probe_name'], "probe_level": spec['probe_level'],
            "strategy": spec.get('strategy','cot'), "condition": condition,
            "metric_name": spec.get('metric_name','minkowski_4'), "dim": spec.get('dim',4),
            "rep": spec['rep'], "correct": correct, "error_type": error_type,
            "final_answer": answer, "is_empty": is_empty,
            "duration_s": round(dur, 1), "output_len": len(output),
            "raw_tail": output[-300:] if output else "",
            "no_axiom": is_noaxiom,
        }

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        counter[0] += 1
        status = "OK" if correct else ("??" if correct is None else "ER")
        if is_empty: status = "EM"
        tag = "NOAX" if is_noaxiom else spec.get('strategy','cot')[:6]
        print(f"[{counter[0]:4d}] {spec['model_name'][:18]:18s} {tag:6s} "
              f"L{spec['probe_level']} {condition:8s} {status} {dur:.0f}s")
        return record

async def main():
    existing_ids = set()
    data_dir = Path('data/comprehensive_v4')
    for pat in ['phase_b_*.jsonl','phase_c_*.jsonl','v7_timeout_retry_*.jsonl',
                'v8_completion_*.jsonl']:
        for f in data_dir.glob(pat):
            with open(f, 'r', encoding='utf-8') as fh:
                for line in fh:
                    if line.strip():
                        existing_ids.add(json.loads(line).get('instance_id',''))

    metric = make_minkowski_metric(4)
    specs = []

    # 1. Phase B completion
    for mname, mid in PHASE_B_MODELS.items():
        for strategy in STRATEGIES:
            for probe in PROBES_L234:
                for cond in ['standard','abstract','random']:
                    for rep in range(3):
                        iid = f"phaseB_{mname[:15]}_{strategy}_L{probe.level}_{cond}_r{rep}"
                        if iid not in existing_ids:
                            specs.append({'instance_id': iid, 'model_name': mname, 'model_id': mid,
                                         'probe_name': probe.name, 'probe_level': probe.level,
                                         'strategy': strategy, 'condition': cond, 'metric': metric,
                                         'metric_name': 'minkowski_4', 'dim': 4, 'rep': rep})

    # 2. Qwen3-8B 600s retry completion (only missing ones)
    with open('data/comprehensive_v4/v7_expanded_20260614_131740.jsonl', 'r', encoding='utf-8') as f:
        v7 = [json.loads(line) for line in f if line.strip()]
    q8_timeouts = [r for r in v7 if r['model']=='Qwen3-8B' and r.get('duration_s',0) >= 290]
    for r in q8_timeouts:
        iid = r['instance_id']
        if iid not in existing_ids:
            specs.append({'instance_id': iid, 'model_name': 'Qwen3-8B',
                         'model_id': 'Qwen/Qwen3-8B', 'probe_name': r['probe'],
                         'probe_level': r['probe_level'], 'strategy': 'cot',
                         'condition': r['condition'], 'metric': metric,
                         'metric_name': 'minkowski_4', 'dim': 4, 'rep': r['rep']})

    # 3. No-axiom extension: Qwen3-8B API
    for probe in PROBE_LADDER:
        for cond in ['standard','abstract','random']:
            for rep in range(3):
                iid = f"phaseC2_Qwen3-8B_noaxiom_L{probe.level}_{cond}_r{rep}"
                if iid not in existing_ids:
                    specs.append({'instance_id': iid, 'model_name': 'Qwen3-8B',
                                 'model_id': 'Qwen/Qwen3-8B', 'probe_name': probe.name,
                                 'probe_level': probe.level, 'strategy': 'no_axiom',
                                 'condition': cond, 'metric': metric,
                                 'metric_name': 'minkowski_4', 'dim': 4, 'rep': rep,
                                 'no_axiom': True})

    if not specs:
        print("All experiments already complete!")
        return

    n_phaseb = sum(1 for s in specs if s.get('strategy') in STRATEGIES)
    n_timeout = sum(1 for s in specs if s['model_name']=='Qwen3-8B' and not s.get('no_axiom'))
    n_noax = sum(1 for s in specs if s.get('no_axiom'))
    n_sc = sum(1 for s in specs if s.get('strategy')=='cot_sc')
    est_calls = len(specs) + n_sc * (SC_SAMPLES - 1)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = data_dir / f"v8_completion_{now}.jsonl"

    print(f"Completion Experiments: {len(specs)} specs, ~{est_calls} API calls")
    print(f"  Phase B (GLM-4 + MiniMax-M2.5): {n_phaseb} experiments")
    print(f"  Qwen3-8B 600s retry: {n_timeout} remaining")
    print(f"  No-axiom extension (Qwen3-8B): {n_noax} experiments")
    print(f"Output: {output_path}\n")

    sem = asyncio.Semaphore(16)
    counter = [0]
    start_time = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        tasks = []
        for s in specs:
            ki = hash(s['model_name']) % 4
            tasks.append(run_one(session, API_KEYS[ki], s, sem, output_path, counter, start_time))
        await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.perf_counter() - start_time
    print(f"\nCompletion DONE: {counter[0]}/{len(specs)} in {elapsed/60:.1f}min")
    print(f"Output: {output_path}")

if __name__ == '__main__':
    asyncio.run(main())

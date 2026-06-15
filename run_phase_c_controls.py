#!/usr/bin/env python3
"""
Phase C: Ablation Controls — No-Axiom Condition
================================================
Tests whether axiom isolation matters: give the same algebraic expressions
WITHOUT axiom descriptions, relying on pre-trained geometric algebra knowledge.

6 models × 5 probes × 3 conditions × N=3 = 270 experiments

Models: DeepSeek-V4-Pro, DeepSeek-V3.2, DeepSeek-R1, Qwen3-32B,
        GLM-4-9B-0414, MiniMax-M2.5
"""
import sys, json, time, asyncio, aiohttp, re
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent))
from experiment_v3.probes import *
from experiment_v3.prompts import get_question, get_symbols

API_KEYS = [
    "sk-qjhtbnwmzcmvfabtpotvkjxqdkqmnwyzwfdobmkwedopxmfi",
    "sk-wzijtyuyivphmvydtuakbfbgerunbsfqyqwzdvphbvguogsc",
    "sk-gzuafdpnjrrtdsacgotbwlytepqrkbzfdutvcokoufkoabeq",
    "sk-ktslkqulxiinxnxrevqshvyvwpmkucsdjxmkjibdbwafgyvv",
]
API_BASE = "https://api.siliconflow.cn/v1"

PHASE_C_MODELS = {
    "DeepSeek-V4-Pro": "deepseek-ai/DeepSeek-V4-Pro",
    "DeepSeek-V3.2": "deepseek-ai/DeepSeek-V3.2",
    "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    "Qwen3-32B": "Qwen/Qwen3-32B",
    "GLM-4-9B-0414": "THUDM/GLM-4-9B-0414",
    "MiniMax-M2.5": "MiniMaxAI/MiniMax-M2.5",
}

def build_no_axiom_system(condition, metric, symbols):
    """System prompt WITHOUT axiom isolation — relies on pre-trained knowledge."""
    dim = len(metric)
    basis_list = ", ".join(symbols[i] for i in range(1, dim + 1))
    lam_lines = "\n".join(f"- Λ({i}) = {v}" for i, v in sorted(metric.items()))

    if condition == "standard":
        return f"""You are a mathematical assistant helping with algebraic computations.
Compute the following expressions using Clifford algebra rules.
Basis elements: {basis_list}
Metric: {lam_lines}
Provide your complete derivation and final answer."""
    elif condition == "abstract":
        return f"""You are a mathematical assistant. Compute the following expressions.
The system uses symbols: {basis_list}
Note: these symbols satisfy ω_i * ω_i = Λ(i) where:
{lam_lines}
Provide your complete derivation and final answer."""
    else:
        return f"""You are a mathematical assistant. Compute the following expressions.
Symbols: {basis_list}
These symbols satisfy: S_i * S_i = Λ(i) where:
{lam_lines}
Provide your complete derivation and final answer."""

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
        metric = spec['metric']
        symbols = get_symbols(spec['condition'], 4)
        system_prompt = build_no_axiom_system(spec['condition'], metric, symbols)
        question = get_question(probe, spec['condition'], symbols, metric)
        user_prompt = f"""{question}

Show your derivation step-by-step.
Output your final answer as: FINAL ANSWER: <your answer>"""

        output, dur = "", 0.0
        for attempt in range(3):
            try:
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {"model": spec['model_id'],
                           "messages": [{"role": "system", "content": system_prompt},
                                        {"role": "user", "content": user_prompt}],
                           "max_tokens": 4096, "temperature": 0.0, "stream": False}
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

        is_empty = len(output) < 50
        answer = extract_answer(output) if not is_empty else ""
        correct, error_type, eval_method = None, "empty", "empty"
        if not is_empty and answer:
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
            "strategy": "no_axiom", "condition": spec['condition'],
            "metric_name": spec['metric_name'], "dim": spec['dim'], "rep": spec['rep'],
            "correct": correct, "error_type": error_type, "final_answer": answer,
            "is_empty": is_empty, "evaluation_method": eval_method,
            "duration_s": round(dur, 1), "output_len": len(output),
            "raw_tail": output[-300:] if output else "",
            "axiom_isolation": False,  # Key: this is the NO-axiom control
        }

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        counter[0] += 1
        elapsed = time.perf_counter() - start_time
        status = "OK" if correct else ("??" if correct is None else "ER")
        if is_empty: status = "EM"
        print(f"[{counter[0]:4d}] {spec['model_name'][:18]:18s} no_axiom  "
              f"L{spec['probe_level']} {spec['condition']:8s} {status} {dur:.0f}s")

        return record


async def main():
    existing_ids = set()
    data_dir = Path('data/comprehensive_v4')
    for f in data_dir.glob('phase_c_*.jsonl'):
        with open(f, 'r', encoding='utf-8') as fh:
            for line in fh:
                if line.strip():
                    existing_ids.add(json.loads(line).get('instance_id', ''))

    metric = make_minkowski_metric(4)
    specs = []
    for mname, mid in PHASE_C_MODELS.items():
        for probe in PROBE_LADDER:
            for cond in ['standard', 'abstract', 'random']:
                for rep in range(3):
                    iid = f"phaseC_{mname[:15]}_noaxiom_L{probe.level}_{cond}_r{rep}"
                    if iid not in existing_ids:
                        specs.append({
                            'instance_id': iid,
                            'model_name': mname, 'model_id': mid,
                            'is_local': False, 'probe_name': probe.name,
                            'probe_level': probe.level, 'strategy': 'no_axiom',
                            'condition': cond, 'metric': metric,
                            'metric_name': 'minkowski_4', 'dim': 4, 'rep': rep,
                        })

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = data_dir / f"phase_c_noaxiom_{now}.jsonl"
    total = len(specs)

    print(f"Phase C (No-Axiom Controls): {total} experiments")
    print(f"  Models: {list(PHASE_C_MODELS.keys())}")
    print(f"  Probes: L0-L4, 3 conditions × N=3")
    print(f"  Key contrast: WITHOUT axiom isolation system prompt")
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
    print(f"\nPhase C DONE: {len(valid)}/{total} in {elapsed/3600:.1f}h, OK={ok}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())

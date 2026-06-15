#!/usr/bin/env python3
"""
Optimized Parallel Experiment Runner v5
- 4 API keys, each running its own set of models
- Local models run in parallel via Ollama
- Crash-safe: append-only JSONL, resume supported
"""

import asyncio, json, sys, time, re, copy
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import aiohttp

sys.path.insert(0, str(Path(__file__).parent))
from experiment_v3.probes import *
from experiment_v3.prompts import build_prompt

# ═══════════════════ CONFIG ═══════════════════

API_KEYS = [
    "sk-qjhtbnwmzcmvfabtpotvkjxqdkqmnwyzwfdobmkwedopxmfi",
    "sk-wzijtyuyivphmvydtuakbfbgerunbsfqyqwzdvphbvguogsc",
    "sk-gzuafdpnjrrtdsacgotbwlytepqrkbzfdutvcokoufkoabeq",
    "sk-ktslkqulxiinxnxrevqshvyvwpmkucsdjxmkjibdbwafgyvv",
]
API_BASE = "https://api.siliconflow.cn/v1"
OLLAMA_BASE = "http://localhost:11434"

# Models assigned to each key (3 per key, except last has 2)
KEY_MODELS = [
    {  # Key 1
        "DeepSeek-V4-Pro": "deepseek-ai/DeepSeek-V4-Pro",
        "GLM-Z1-9B-0414": "THUDM/GLM-Z1-9B-0414",
        "Qwen3-14B": "Qwen/Qwen3-14B",
    },
    {  # Key 2
        "DeepSeek-V3.2": "deepseek-ai/DeepSeek-V3.2",
        "GLM-4-9B-0414": "THUDM/GLM-4-9B-0414",
        "Qwen3-8B": "Qwen/Qwen3-8B",
    },
    {  # Key 3
        "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
        "MiniMax-M2.5": "MiniMaxAI/MiniMax-M2.5",
        "DeepSeek-V4-Flash": "deepseek-ai/DeepSeek-V4-Flash",
    },
    {  # Key 4
        "Qwen3-32B": "Qwen/Qwen3-32B",
        "DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
    },
]

# Local models (Ollama)
LOCAL_MODELS = {
    "Qwen3-8B-Local": "qwen3:8b",
    "Qwen2.5-Coder-14B": "qwen2.5-coder:14b",
}

TIMEOUT = 300
MAX_TOKENS = 4096
MAX_RETRIES = 2

# ═══════════════════ HELPERS ═══════════════════

def extract_answer(text):
    if not text or len(text.strip()) < 5:
        return ""
    patterns = [
        r'FINAL\s*ANSWER\s*:\s*(.+?)(?:\n\n|\n\s*\n|\n$|$)',
        r'FINAL\s*RESULT\s*:\s*(.+?)(?:\n\n|\n\s*\n|\n$|$)',
        r'\*\*FINAL\s*ANSWER\*\*[:\s]*(.+?)(?:\n\n|\n\s*\n|\n$|$)',
        r'\\boxed\{([^}]+)\}',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            ans = m.group(1).strip()
            ans = ans.replace('\\(', '').replace('\\)', '')
            ans = ans.replace('\\[', '').replace('\\]', '')
            ans = re.sub(r'\$+', '', ans)
            ans = re.sub(r'^\*\*\s*', '', ans)
            return ans.strip()

    display = re.findall(r'(?:\\\[(.*?)\\\]|\$\$(.*?)\$\$)', text, re.DOTALL)
    if display:
        last = display[-1]
        r = last[0] or last[1]
        if r: return r.strip()

    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if re.match(r'^(Therefore|Thus|Hence|So|The|This|We|Note|Step|Let|I\s|A\s|In\s|By\s|Using|Now|First|Second|Third|Finally|Next|Verif)', line, re.IGNORECASE):
            continue
        line = line.replace('\\(', '').replace('\\)', '')
        if len(line) > 3 and any(c in line for c in '*+-=eEwW'):
            return line.strip()
    return lines[-1].strip() if lines else ""


def classify(text):
    l = len(text) if text else 0
    if l < 50: return "EMPTY", l
    if re.search(r'\.\.\.\s*$|truncated|max_tokens', text, re.IGNORECASE):
        return "TRUNCATED", l
    return "VALID", l


def evaluate(answer, probe, condition, metric):
    if not answer:
        return None, "empty_output", "empty"
    symbols = get_symbols(condition, max(len(metric), 4))
    gt = get_ground_truth(probe, condition, symbols, metric)
    if answers_equivalent(answer, gt):
        return True, "none", "semantic_equivalent"
    for err_t in probe.common_errors:
        ef = err_t
        for i in range(1, 11):
            sym = symbols.get(i, f"s_{i}")
            ef = ef.replace(f"{{b{i}}}", sym)
        if answers_equivalent(answer, ef):
            if "commutative" in err_t.lower():
                return False, "commutative_error", "error_match"
            elif "sign" in err_t.lower():
                return False, "sign_error", "error_match"
            elif "metric" in err_t.lower():
                return False, "metric_error", "error_match"
            return False, "known_error", "error_match"
    na = normalize_answer(answer)
    ng = normalize_ground_truth(gt)
    if ng and ng in na:
        return True, "none", "contains_gt"
    return None, "uncertain", "uncertain"


# ═══════════════════ API CALLER ═══════════════════

async def call_api(session, api_key, model_id, sys_prompt, usr_prompt, temp=0.0):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model_id, "messages": [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt}],
               "max_tokens": MAX_TOKENS, "temperature": temp, "stream": False}
    start = time.perf_counter()
    try:
        async with session.post(f"{API_BASE}/chat/completions", headers=headers,
                                json=payload, timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data["choices"][0]["message"]["content"], time.perf_counter() - start, data.get("usage", {})
            text = await resp.text()
            return f"HTTP{resp.status}:{text[:150]}", time.perf_counter() - start, {}
    except asyncio.TimeoutError:
        return "TIMEOUT", time.perf_counter() - start, {}
    except Exception as e:
        return f"ERR:{str(e)[:150]}", time.perf_counter() - start, {}


async def call_ollama(session, model_name, sys_prompt, usr_prompt, temp=0.0):
    payload = {"model": model_name, "system": sys_prompt, "prompt": usr_prompt,
               "stream": False, "options": {"temperature": temp, "num_predict": MAX_TOKENS}}
    start = time.perf_counter()
    try:
        async with session.post(f"{OLLAMA_BASE}/api/generate", json=payload,
                                timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as resp:
            if resp.status == 200:
                data = await resp.json()
                u = {"prompt_tokens": data.get("prompt_eval_count", 0),
                     "completion_tokens": data.get("eval_count", 0)}
                return data.get("response", ""), time.perf_counter() - start, u
            return f"OLLAMA{resp.status}", time.perf_counter() - start, {}
    except asyncio.TimeoutError:
        return "TIMEOUT", time.perf_counter() - start, {}
    except aiohttp.ClientConnectorError:
        return "OLLAMA_DOWN", time.perf_counter() - start, {}
    except Exception as e:
        return f"ERR:{str(e)[:100]}", time.perf_counter() - start, {}


# ═══════════════════ WORKER ═══════════════════

async def run_model_experiments(api_key, models, specs, output_path, counter, total_start, label):
    """Run all specs for assigned models using one API key."""
    sem = asyncio.Semaphore(6)  # 6 concurrent per key

    async def run_one(spec):
        async with sem:
            probe = [p for p in PROBE_LADDER if p.level == spec["probe_level"]][0]
            bundle = build_prompt(probe, spec["strategy"], spec["condition"], spec["metric"])
            temp = 0.7 if spec["strategy"] == "cot_sc" else 0.0

            is_local = spec["model_name"] in LOCAL_MODELS
            for attempt in range(MAX_RETRIES + 1):
                try:
                    if is_local:
                        output, dur, usage = await call_ollama(
                            session, spec["model_id"], bundle.system_prompt, bundle.user_prompt, temp)
                    else:
                        output, dur, usage = await call_api(
                            session, api_key, spec["model_id"], bundle.system_prompt, bundle.user_prompt, temp)
                    if not output.startswith("HTTP") and not output.startswith("ERR") and output not in ("TIMEOUT", "OLLAMA_DOWN"):
                        break
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(2 ** attempt)
                except Exception as e:
                    output = f"EXC:{str(e)[:150]}"
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(1)

            quality, out_len = classify(output)
            is_empty = quality == "EMPTY"
            is_truncated = quality == "TRUNCATED"
            timed_out = output == "TIMEOUT"
            answer = extract_answer(output) if not is_empty else ""
            correct, error_type, eval_method = evaluate(answer, probe, spec["condition"], spec["metric"])

            record = {
                "instance_id": spec["instance_id"],
                "model": spec["model_name"], "probe": spec["probe_name"],
                "probe_level": spec["probe_level"], "strategy": spec["strategy"],
                "condition": spec["condition"], "metric_name": spec["metric_name"],
                "dim": spec["dim"], "rep": spec.get("rep", 0),
                "correct": correct, "error_type": error_type, "final_answer": answer,
                "is_empty": is_empty, "is_truncated": is_truncated,
                "evaluation_method": eval_method,
                "duration_s": round(dur, 1), "output_len": len(output),
                "timed_out": timed_out,
                "raw_tail": output[-300:] if output else "",
            }

            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            counter[0] += 1
            elapsed = time.perf_counter() - total_start
            rate = counter[0] / (elapsed / 3600) if elapsed > 0 else 0
            status = "OK" if correct else ("ER" if correct is False else ("EM" if is_empty else "??"))
            print(f"[{counter[0]:5d}] {spec['model_name'][:18]:18s} L{spec['probe_level']} "
                  f"{spec['strategy']:7s} {spec['condition']:8s} {status} {dur:.0f}s ({rate:.0f}/hr)")

            return record

    tasks = [run_one(s) for s in specs]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, dict)]


# ═══════════════════ SPEC GENERATOR ═══════════════════

def generate_specs_for_models(models, probes, strategies, conditions, metrics, metric_names, reps, prefix=""):
    specs = []
    for mname, mid in sorted(models.items()):
        is_local = mname in LOCAL_MODELS
        for probe in probes:
            for strategy in strategies:
                for condition in conditions:
                    for metric, mn in zip(metrics, metric_names):
                        for rep in range(reps):
                            sc_samples = 5 if strategy == "cot_sc" else 1
                            for sci in range(sc_samples):
                                specs.append({
                                    "instance_id": f"{prefix}_{mname[:10]}_L{probe.level}_{strategy}_{condition}_{mn}_r{rep}_sc{sci}",
                                    "model_name": mname, "model_id": mid, "is_local": is_local,
                                    "probe_name": probe.name, "probe_level": probe.level,
                                    "strategy": strategy, "condition": condition,
                                    "metric": metric, "metric_name": mn,
                                    "dim": len(metric), "rep": rep, "sc_index": sci,
                                })
    return specs


# ═══════════════════ MAIN ═══════════════════

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="main", help="main|strategies|controls|all")
    parser.add_argument("--reps", type=int, default=3, help="Replications per cell")
    parser.add_argument("--output-dir", type=str, default="data/comprehensive_v4")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_api_models = {}
    for km in KEY_MODELS:
        all_api_models.update(km)

    metric = make_minkowski_metric(4)

    # ── Generate specs ──
    all_specs = []

    if args.phase in ("main", "all"):
        # Main: all API models + local, CoT, 5 probes, 3 conditions
        api_specs = generate_specs_for_models(
            all_api_models, PROBE_LADDER, ["cot"],
            ["standard", "abstract", "random"], [metric], ["minkowski_4"],
            args.reps, prefix="main")
        local_specs = generate_specs_for_models(
            LOCAL_MODELS, PROBE_LADDER, ["cot"],
            ["standard", "abstract", "random"], [metric], ["minkowski_4"],
            max(2, args.reps // 2), prefix="main_local")
        all_specs.extend(api_specs)
        all_specs.extend(local_specs)

    if args.phase in ("strategies", "all"):
        # Strategy comparison on key models
        key_models = {m: all_api_models[m] for m in
                      ["DeepSeek-V4-Pro", "DeepSeek-V3.2", "Qwen3-32B",
                       "GLM-4-9B-0414", "MiniMax-M2.5"] if m in all_api_models}
        strat_specs = generate_specs_for_models(
            key_models, [PROBE_L2, PROBE_L3, PROBE_L4],
            ["zs", "cot", "cot_sc", "cot_pot", "cot_sv"],
            ["standard", "abstract", "random"], [metric], ["minkowski_4"],
            max(2, args.reps // 2), prefix="strat")
        all_specs.extend(strat_specs)

    if args.phase in ("controls", "all"):
        # Controls: no-axiom on key models
        ctrl_models = {m: all_api_models[m] for m in
                       ["DeepSeek-V4-Pro", "DeepSeek-V3.2", "Qwen3-32B",
                        "GLM-4-9B-0414", "MiniMax-M2.5", "Qwen3-14B"] if m in all_api_models}
        ctrl_specs = generate_specs_for_models(
            ctrl_models, PROBE_LADDER, ["zs"],
            ["standard", "abstract", "random"], [metric], ["minkowski_4"],
            max(2, args.reps // 2), prefix="ctrl")
        all_specs.extend(ctrl_specs)

    # Distribute API specs by key
    output_path = output_dir / f"v5_{args.phase}_{now}.jsonl"

    # Separate local and API specs
    local_specs_list = [s for s in all_specs if s["is_local"]]
    api_specs_list = [s for s in all_specs if not s["is_local"]]

    # Assign API specs to keys based on model
    key_specs = {i: [] for i in range(len(API_KEYS))}
    for spec in api_specs_list:
        mname = spec["model_name"]
        for ki, km in enumerate(KEY_MODELS):
            if mname in km:
                key_specs[ki].append(spec)
                break

    total_api = sum(len(v) for v in key_specs.values())
    total_local = len(local_specs_list)
    total = total_api + total_local

    print(f"\n{'='*70}")
    print(f"PARALLEL EXPERIMENT RUNNER v5 — Phase: {args.phase}")
    print(f"API models: {len(all_api_models)}, Local models: {len(LOCAL_MODELS)}")
    print(f"API specs: {total_api} ({', '.join(f'Key{k}:{len(v)}' for k, v in key_specs.items())})")
    print(f"Local specs: {total_local}")
    print(f"Total: {total} experiments")
    print(f"Reps: {args.reps}")
    print(f"Output: {output_path}")
    print(f"{'='*70}\n")

    counter = [0]
    total_start = time.perf_counter()
    all_results = []

    async with aiohttp.ClientSession() as session:
        # Launch all 4 API key workers in parallel + 1 local worker
        tasks = []
        for ki in range(len(API_KEYS)):
            if key_specs[ki]:
                tasks.append(run_model_experiments(
                    API_KEYS[ki], KEY_MODELS[ki], key_specs[ki],
                    output_path, counter, total_start,
                    f"Key{ki+1}"))

        # Local models in separate worker
        if local_specs_list:
            tasks.append(run_model_experiments(
                "local", LOCAL_MODELS, local_specs_list,
                output_path, counter, total_start, "Local"))

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        for br in batch_results:
            if isinstance(br, list):
                all_results.extend(br)

    elapsed = time.perf_counter() - total_start
    print(f"\n{'='*70}")
    print(f"ALL DONE: {len(all_results)}/{total} experiments")
    print(f"Duration: {elapsed/3600:.1f} hours ({elapsed/60:.0f} min)")
    print(f"Output: {output_path}")
    print(f"{'='*70}")

    # Quick summary
    ok = sum(1 for r in all_results if r.get("correct") is True)
    emp = sum(1 for r in all_results if r.get("is_empty"))
    print(f"Correct: {ok}/{len(all_results)} ({100*ok/max(1,len(all_results)):.1f}%), Empty: {emp}")

    # Model breakdown
    print(f"\n--- By Model ---")
    models = defaultdict(list)
    for r in all_results:
        models[r["model"]].append(r)
    for m in sorted(models, key=lambda m: -len(models[m])):
        mr = models[m]
        c = sum(1 for r in mr if r["correct"] is True)
        print(f"  {m[:25]:25s}: {c}/{len(mr)} ({100*c/max(1,len(mr)):.0f}%)")

    # Probe x Condition
    print(f"\n--- Accuracy by Probe x Condition ---")
    for level in range(5):
        for cond in ["standard", "abstract", "random"]:
            lr = [r for r in all_results if r.get("probe_level") == level
                  and r.get("condition") == cond and not r.get("is_empty")]
            if lr:
                c = sum(1 for r in lr if r["correct"] is True)
                print(f"  L{level} {cond:10s}: {c}/{len(lr)} ({100*c/len(lr):.0f}%)")


if __name__ == "__main__":
    asyncio.run(main())

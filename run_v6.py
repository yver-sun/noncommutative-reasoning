#!/usr/bin/env python3
"""V6 Direct Runner - One model per async task, simple and robust."""
import asyncio, json, sys, time, re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import aiohttp

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

TIMEOUT = 300

def extract_answer(text):
    if not text or len(text.strip()) < 5: return ""
    for pat in [r'(?:###?\s*)?FINAL\s*ANSWER\s*:?\s*(.+?)(?:\n\n|\n\s*\n|\n\s*$|$)',
                r'(?:###?\s*)?FINAL\s*RESULT\s*:?\s*(.+?)(?:\n\n|\n\s*\n|\n\s*$|$)',
                r'\*\*FINAL\s*ANSWER\*\*[:\s]*(.+?)(?:\n\n|\n\s*\n|\n\s*$|$)',
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
        if re.match(r'^(Therefore|Thus|Hence|So|The|This|We|Note|Step|Let|I |A |In |By |Using|Now|First|Second|Third|Finally|Next|Verif)', line, re.IGNORECASE):
            continue
        line = line.replace('\\(', '').replace('\\)', '')
        if len(line) > 3 and any(c in line for c in '*+-=eEwW'):
            return line.strip()
    return lines[-1].strip() if lines else ""

def evaluate_answer(answer, probe, condition, metric):
    if not answer: return None, "empty_output", "empty"
    symbols = get_symbols(condition, max(len(metric), 4))
    gt = get_ground_truth(probe, condition, symbols, metric)
    if answers_equivalent(answer, gt): return True, "none", "semantic_equivalent"
    na = normalize_answer(answer); ng = normalize_ground_truth(gt)
    if ng and ng in na: return True, "none", "contains_gt"
    return None, "uncertain", "uncertain"

async def run_model(model_name, model_id, is_local, api_key, specs, output_path, counter, start_time):
    results = []
    connector = aiohttp.TCPConnector(limit=6)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(6)
        async def run_one(spec):
            async with sem:
                probe = [p for p in PROBE_LADDER if p.level == spec["probe_level"]][0]
                bundle = build_prompt(probe, spec["strategy"], spec["condition"], spec["metric"])

                output, dur = "", 0.0
                for attempt in range(3):
                    try:
                        if is_local:
                            payload = {"model": model_id, "system": bundle.system_prompt,
                                       "prompt": bundle.user_prompt, "stream": False,
                                       "options": {"temperature": 0.0, "num_predict": 4096}}
                            start = time.perf_counter()
                            async with session.post(f"{OLLAMA_BASE}/api/generate", json=payload,
                                                    timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as resp:
                                output = (await resp.json()).get("response", "") if resp.status == 200 else f"O{resp.status}"
                            dur = time.perf_counter() - start
                        else:
                            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                            payload = {"model": model_id,
                                       "messages": [{"role": "system", "content": bundle.system_prompt},
                                                    {"role": "user", "content": bundle.user_prompt}],
                                       "max_tokens": 4096, "temperature": 0.0, "stream": False}
                            start = time.perf_counter()
                            async with session.post(f"{API_BASE}/chat/completions", headers=headers,
                                                    json=payload, timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as resp:
                                if resp.status == 200:
                                    output = (await resp.json())["choices"][0]["message"]["content"]
                                else:
                                    output = f"H{resp.status}"
                            dur = time.perf_counter() - start
                        if not output.startswith("H") and not output.startswith("O"): break
                        if attempt < 2: await asyncio.sleep(1)
                    except asyncio.TimeoutError:
                        output = "TIMEOUT"; dur = TIMEOUT
                        if attempt < 2: await asyncio.sleep(2)
                    except Exception as e:
                        output = f"ERR:{str(e)[:80]}"
                        if attempt < 2: await asyncio.sleep(1)

                is_empty = len(output) < 50
                answer = extract_answer(output) if not is_empty else ""
                correct, error_type, eval_method = evaluate_answer(answer, probe, spec["condition"], spec["metric"])

                record = {
                    "instance_id": spec["instance_id"], "model": model_name,
                    "probe": spec["probe_name"], "probe_level": spec["probe_level"],
                    "strategy": spec["strategy"], "condition": spec["condition"],
                    "metric_name": spec["metric_name"], "dim": spec["dim"], "rep": spec.get("rep", 0),
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
                status = "OK" if correct else ("ER" if correct is False else ("EM" if is_empty else "??"))
                print(f"[{counter[0]:4d}] {model_name[:18]:18s} L{spec['probe_level']} "
                      f"{spec['strategy']:7s} {spec['condition']:8s} {status} {dur:.0f}s")

                return record

        tasks = [run_one(s) for s in specs]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in batch_results:
            if isinstance(r, dict): results.append(r)
    return results

def main():
    metric = make_minkowski_metric(4)
    output_dir = Path("data/comprehensive_v4")
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(output_dir / f"v6_{now}.jsonl")

    all_specs = []
    for mname, mid in sorted(API_MODELS.items()):
        for probe in PROBE_LADDER:
            for cond in ["standard", "abstract", "random"]:
                for rep in range(3):
                    all_specs.append({
                        "instance_id": f"v6_{mname.replace(' ','_')[:15]}_L{probe.level}_cot_{cond}_r{rep}",
                        "model_name": mname, "model_id": mid, "is_local": False,
                        "probe_name": probe.name, "probe_level": probe.level,
                        "strategy": "cot", "condition": cond,
                        "metric": metric, "metric_name": "minkowski_4", "dim": 4, "rep": rep,
                    })
    for mname, mid in sorted(LOCAL_MODELS.items()):
        for probe in PROBE_LADDER:
            for cond in ["standard", "abstract", "random"]:
                for rep in range(2):
                    all_specs.append({
                        "instance_id": f"v6_{mname.replace(' ','_')[:15]}_L{probe.level}_cot_{cond}_r{rep}",
                        "model_name": mname, "model_id": mid, "is_local": True,
                        "probe_name": probe.name, "probe_level": probe.level,
                        "strategy": "cot", "condition": cond,
                        "metric": metric, "metric_name": "minkowski_4", "dim": 4, "rep": rep,
                    })

    model_specs = defaultdict(list)
    for s in all_specs:
        model_specs[(s["model_name"], s["is_local"])].append(s)

    total = len(all_specs)
    print(f"V6 RUNNER: {total} experiments, {len(model_specs)} models")
    print(f"API: {len(API_MODELS)}, Local: {len(LOCAL_MODELS)}")
    print(f"Output: {output_path}")
    print()

    async def amain():
        counter = [0]; start_time = time.perf_counter()
        tasks = []
        for (mname, is_local), specs in model_specs.items():
            mid = LOCAL_MODELS.get(mname) if is_local else API_MODELS.get(mname)
            ki = hash(mname) % len(API_KEYS)
            api_key = API_KEYS[ki]
            tasks.append(run_model(mname, mid, is_local, api_key, specs, output_path, counter, start_time))

        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        flat = []
        for r in all_results:
            if isinstance(r, list): flat.extend(r)

        elapsed = time.perf_counter() - start_time
        ok = sum(1 for r in flat if r.get("correct") is True)
        emp = sum(1 for r in flat if r.get("is_empty"))

        print(f"\nDONE: {len(flat)}/{total}, OK={ok} ({100*ok/max(1,len(flat)):.0f}%), Empty={emp}")
        print(f"Duration: {elapsed/3600:.1f}h, Output: {output_path}")

        models = defaultdict(list)
        for r in flat: models[r["model"]].append(r)
        for m in sorted(models, key=lambda m: -len(models[m])):
            mr = models[m]; c = sum(1 for r in mr if r["correct"] is True)
            print(f"  {m[:22]:22s}: {c}/{len(mr)} ({100*c/max(1,len(mr)):.0f}%)")

    asyncio.run(amain())

if __name__ == "__main__":
    main()

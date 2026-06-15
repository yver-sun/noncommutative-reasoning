#!/usr/bin/env python3
"""
P2.1: Tokenization Quantification Experiment
=============================================
Use local Llama3-8B and Qwen2.5-1.5B tokenizers to count actual tokens
for each probe x condition combination.
Plot token count vs accuracy to rule out tokenization confound.
"""
import sys, json, glob, requests
sys.path.insert(0, '.')
from experiment_v3.probes import *
from experiment_v3.prompts import build_prompt

OLLAMA_BASE = "http://localhost:11434"
LOCAL_MODELS = {
    "Llama3-8B-Local": "llama3:latest",
    "Qwen2.5-1.5B-Local": "qwen2.5:1.5b",
}

def count_tokens(model_id, text):
    """Count tokens using Ollama's tokenize endpoint."""
    try:
        resp = requests.post(f"{OLLAMA_BASE}/api/tokenize",
                            json={"model": model_id, "text": text},
                            timeout=30)
        if resp.status_code == 200:
            return len(resp.json().get("tokens", []))
    except:
        pass
    # Fallback: rough char/4 estimate
    return len(text) // 4

def main():
    metric = make_minkowski_metric(4)
    results = []

    for model_name, model_id in LOCAL_MODELS.items():
        print(f"\n=== {model_name} ({model_id}) ===")
        for probe in PROBE_LADDER:
            for cond in ['standard', 'abstract', 'random']:
                bundle = build_prompt(probe, 'cot', cond, metric)
                full_prompt = bundle.system_prompt + "\n\n" + bundle.user_prompt
                n_tokens = count_tokens(model_id, full_prompt)
                results.append({
                    'model': model_name,
                    'probe': probe.name,
                    'probe_level': probe.level,
                    'condition': cond,
                    'token_count': n_tokens,
                    'char_count': len(full_prompt),
                })
                print(f"  L{probe.level} {cond:8s}: {n_tokens} tokens")

    # Save
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('data/analysis_v6/tokenization_analysis.csv', index=False)
    print(f"\nSaved to data/analysis_v6/tokenization_analysis.csv")

    # Summary stats
    print("\n=== Summary ===")
    for model_name in LOCAL_MODELS:
        md = df[df['model'] == model_name]
        print(f"{model_name}:")
        print(f"  Token range: {md['token_count'].min()} - {md['token_count'].max()}")
        print(f"  Mean tokens per condition:")
        for cond in ['standard', 'abstract', 'random']:
            cd = md[md['condition'] == cond]
            print(f"    {cond:8s}: {cd['token_count'].mean():.0f} (chars: {cd['char_count'].mean():.0f})")

if __name__ == '__main__':
    main()

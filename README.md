# Non-Commutative Algebraic Reasoning in Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete code, data, and experimental infrastructure for the paper:

**"Non-Commutative Algebraic Reasoning in Large Language Models: A Five-Level Probe Study Maps Reasoning Boundaries Through Systematic Intervention"**

Submitted to *Cognitive Systems Research* (Elsevier).

## Repository Structure

```
.
├── README.md                          # This file
├── paper/
│   ├── csr_manuscript.tex             # Main manuscript (LaTeX)
│   ├── csr_manuscript.pdf             # Main manuscript (PDF)
│   ├── csr_supplementary.tex          # Supplementary materials (LaTeX)
│   ├── csr_supplementary.pdf          # Supplementary materials (PDF)
│   └── figures_r/                     # All figures (PNG + PDF)
├── experiment_v3/
│   ├── probes.py                      # Probe ladder definitions + ground truth
│   ├── prompts.py                     # 5 strategies × 3 conditions = 15 prompts
│   ├── runner.py                      # Async experiment execution engine
│   ├── analysis.py                    # Statistical analysis pipeline
│   └── figures.py                     # Figure generation
├── data/
│   ├── comprehensive_v4/              # Raw experiment data (JSONL)
│   │   ├── v6_*.jsonl                 # Initial N=3 experiment (484 records)
│   │   ├── v7_expanded_*.jsonl        # Phase A N=10 expansion (1200 records)
│   │   ├── v7_local_unique_*.jsonl    # Unique local models (135 records)
│   │   ├── v7_timeout_retry_*.jsonl   # 600s timeout retry (129 records)
│   │   ├── phase_b_*.jsonl            # Strategy comparison (355 records)
│   │   ├── phase_c_*.jsonl            # No-axiom ablation (315 records)
│   │   └── v8_unlimited_time_*.jsonl  # Unlimited-time local (135 records)
│   └── analysis_v6/
│       ├── full_analysis.json         # Summary statistics
│       └── full_error_taxonomy.json   # Error taxonomy results
├── run_expansion.py                   # Phase A: N=3→10 expansion
├── run_phase_b_strategies.py          # Phase B: Strategy comparison
├── run_phase_c_controls.py            # Phase C: No-axiom ablation
├── redo_timeouts.py                   # 600s timeout retry
├── run_local_unique.py                # Unique local model experiment
├── run_unlimited_local.py             # Unlimited-time local experiment
├── run_full_analysis.py               # Comprehensive statistical analysis
├── error_taxonomy.py                  # 3-tier error classification
├── verify_derivations.py              # Ground truth self-audit
├── tokenization_analysis.py           # Token-count quantification
└── capture_error_cases.py             # Full-output error capture
```

## Experiment Summary

- **2,383 controlled experiments** across 14 LLMs (11 API + 3 local)
- **5 probe levels** (L0--L4) of escalating anti-commutation complexity
- **3 symbol conditions** (Standard, Abstract, Random)
- **5 prompt strategies** (ZS, CoT, CoT+SC@5, CoT+PoT, CoT+Self-Verify)
- **No-axiom ablation** (6 models, 315 experiments)
- **600s timeout retry** (3 models, 129 experiments)
- **Unlimited-time local experiment** (3 models, 135 experiments)

## Key Findings

1. **Difficulty hierarchy**: L0(94.7%) → L1(98.6%) → **L2(61.4%)** → L3(59.2%) → L4(46.9%)
2. **L1→L2 cliff**: -37.2 pp marks the transition from single-rule to multi-rule coordination
3. **Two reasoning mechanisms**: de novo rule application (axiom-dependent, <75% baseline) vs. pre-trained knowledge retrieval (axiom-independent, >80%)
4. **Self-Consistency @5**: 63% timeout but near-perfect accuracy on completed trials — resource boundary, not reasoning boundary
5. **60% of errors**: System or format artifacts, not reasoning failures (3-tier taxonomy)

## Reproduction

### Requirements
```bash
pip install aiohttp requests numpy scipy pandas kingdon
```

### Core Experiment
```bash
# Phase A: N=10 expansion (requires SiliconFlow API keys)
python run_expansion.py

# Phase B: Strategy comparison
python run_phase_b_strategies.py

# Phase C: No-axiom ablation
python run_phase_c_controls.py

# Full analysis
python run_full_analysis.py
```

### Ground Truth Verification
```bash
python verify_derivations.py    # 15/15 probe×condition combinations verified
python error_taxonomy.py        # 3-tier error classification
python deep_audit.py            # 86 automated audit tests (85/86 passed)
```

## Citation

```bibtex
@article{noncommutative2026,
  title={Non-Commutative Algebraic Reasoning in Large Language Models:
         A Five-Level Probe Study Maps Reasoning Boundaries
         Through Systematic Intervention},
  author={Independent Researcher},
  journal={Cognitive Systems Research},
  year={2026},
  note={Under review}
}
```

## License

MIT License. See LICENSE file for details.

## Contact

Corresponding author. Contact available upon request.

# PIPS: PageIndex-Inspired Prompt Slicing

## Purpose

Enterprise assistants often send one huge system prompt (here, 15 policy sections, ~30k tokens) with every request, even when a question only needs one or two sections. PIPS tests whether the prompt can be **sliced down to only the relevant sections before calling the LLM**, without losing answer quality.

A full-prompt baseline is compared against four PageIndex-style routing strategies on 10 test queries. Each answer is scored 0–10 by an LLM judge against ground-truth facts. Details: [docs/methodology.md](docs/methodology.md).

## Results

| Mode | Avg full tokens | Avg sliced tokens | Token reduction | Routing accuracy | Full score | Sliced score | Delta |
|---|---|---|---|---|---|---|---|
| Manual keywords | 30105 | 4221 | 86.0% | 100% | 8.2 | 8.2 | +0.0 |
| LLM-generated keywords | 30105 | 4197 | 86.1% | 100% | 8.1 | 8.2 | +0.1 |
| Manual keywords + cross-encoder | 30105 | 4160 | 86.2% | 100% | 8.1 | 7.9 | −0.2 |
| LLM keywords + cross-encoder | 30105 | 4247 | 85.9% | 100% | 8.3 | 8.0 | −0.3 |

Slicing cuts input tokens by about 86% while answer quality stays within ±0.3 points. Source: [`results/comparison_summary.txt`](results/comparison_summary.txt).

## Repository structure

```text
PIPS/
├── README.md
├── notebooks/
│   └── PIPS_tests.ipynb                ← START HERE: verifies every test, outputs saved inline
├── src/pips/                           Source code (Python package)
│   ├── prompts/
│   │   └── enterprise_assistant.py     The 15-section enterprise system prompt
│   ├── routing/
│   │   ├── keyword_tree.py             PageIndex keyword tree: scoring + prompt slicing
│   │   ├── llm_keyword_index.py        Claude-generated keyword index
│   │   └── cross_encoder_reranker.py   Local cross-encoder reranking
│   ├── evaluation/
│   │   └── benchmark.py                10 test queries, LLM calls, LLM-as-judge, report
│   ├── cli/
│   │   ├── benchmark.py                `pips-benchmark` command
│   │   └── compare.py                  `pips-compare` command
│   └── paths.py                        Project root / data / results paths
├── docs/
│   ├── README.md                       Docs index
│   └── methodology.md                  Routing modes, metrics, output files
├── data/
│   └── generated_keywords.json         Cached LLM-generated keyword index
├── results/                            Saved benchmark outputs (all 4 modes)
├── pyproject.toml                      Package metadata, dependencies, CLI commands
├── requirements.txt                    Same dependencies, for plain pip
├── .python-version                     3.11
└── .env.example                        Copy to .env and add your Anthropic API key
```

## Installation

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Or with pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Configuration

```bash
cp .env.example .env    # then set ANTHROPIC_API_KEY
```

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | (required) | Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Model used for answers and the judge |
| `BENCHMARK_INDEX_MODE` | `manual` | Default for `pips-benchmark --mode` |
| `REBUILD_LLM_INDEX` | `0` | `1` regenerates `data/generated_keywords.json` |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L6-v2` | Local reranker model |
| `CROSS_ENCODER_CANDIDATE_K` | `5` | Keyword candidates passed to the reranker |

## Verify the results (notebook)

Open [`notebooks/PIPS_tests.ipynb`](notebooks/PIPS_tests.ipynb). Outputs from the last run are saved in it, so it can be read without running anything. To re-run it, use the project's virtualenv as the kernel and choose Run All.

| Section | API cost |
|---|---|
| 1. Routing check: which sections each mode picks for each query | Free |
| 2. Full benchmark per mode (off by default, toggle in `RUN_MODES`) | 40 calls per mode |
| 3. Comparison across all modes | Free |
| 4. Answer inspector: full vs sliced answers with judge reasoning | Free |

## Run the benchmark (command line)

```bash
pips-benchmark --help
pips-benchmark --mode manual     # one mode, 40 API calls
pips-benchmark --mode all        # all 4 modes, 160 API calls
pips-compare                     # rebuild results/comparison_summary.txt (free)
```

Modes: `manual`, `llm`, `cross`, `llm_cross`, `both` (manual + llm), `all`.

## Use as a library

```python
from pips.prompts.enterprise_assistant import PROMPT_SECTIONS, SLICED_BASE_PROMPT
from pips.routing.keyword_tree import PageIndexTree

tree = PageIndexTree()
tree.build(PROMPT_SECTIONS)
prompt, sections = tree.slice("How do I set up SSO with Okta?", SLICED_BASE_PROMPT, top_k=2)
print(sections)
```

## Outputs

| File | Contents |
|---|---|
| `results/benchmark_results_<mode>.json` | Per query: selected sections, tokens, latency, answers, judge score and reasoning |
| `results/report_<mode>.txt` | Human-readable report for one mode |
| `results/comparison_summary.txt` | All modes side by side |

# Methodology

How the PIPS benchmark works: the routing modes, what gets measured, and how to read the output files.

## Core Idea

The full enterprise prompt contains 15 policy sections. A normal full-prompt call sends every section to the LLM for every query, even when the question only needs one or two sections.

PageIndex-style slicing builds a lightweight map from query terms to relevant sections, retrieves only the most relevant sections, and sends a much smaller sliced prompt to the LLM.

```text
Full prompt baseline:
  query → all 15 sections → Claude answer → Claude judge

Prompt slicing:
  query → section routing → top 2 sections → Claude answer → Claude judge
```

The benchmark records exact Anthropic input token usage from `resp.usage.input_tokens`, so token counts reflect the actual API calls.

## Four Experiment Modes

### 1. Manual Keyword PageIndex

```bash
pips-benchmark --mode manual
```

This uses the human-written `keywords` lists inside `PROMPT_SECTIONS`.

```text
query
↓
manual section keywords
↓
PageIndex keyword scoring
↓
top 2 sections
↓
sliced prompt
```

Use this as a controlled baseline. It answers: if a good keyword map exists, how much token reduction can prompt slicing achieve?

### 2. LLM-Generated Keyword PageIndex

```bash
pips-benchmark --mode llm
```

This asks Claude to generate routing keywords from each section's content, then saves them to:

```text
data/generated_keywords.json
```

The generated index is reused in later runs. To rebuild it:

```bash
pips-benchmark --mode llm --rebuild-llm-index
```

This mode is closer to the PageIndex idea because the keyword map is derived from document content instead of being manually supplied.

### 3. Manual Keywords + Cross-Encoder Reranking

```bash
pips-benchmark --mode cross
```

This first uses manual keywords to retrieve candidate sections, then uses a local Hugging Face cross-encoder to rerank those candidate sections.

```text
query
↓
manual keyword candidates, top 5
↓
cross-encoder scores (query, section_content)
↓
top 2 reranked sections
↓
sliced prompt
```

The cross-encoder is not a vector database. It does not store embeddings. It directly scores query-section pairs at runtime.

Default model:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

### 4. LLM Keywords + Cross-Encoder Reranking

```bash
pips-benchmark --mode llm_cross
```

This is the strongest proposed method:

```text
query
↓
LLM-generated keyword candidates, top 5
↓
cross-encoder reranking
↓
top 2 sections
↓
sliced prompt
```

This combines automatic PageIndex construction with semantic reranking before the sliced prompt is created.

## Running Multiple Modes

Run manual + LLM-generated keyword modes:

```bash
pips-benchmark --mode both
```

Run all four modes:

```bash
pips-benchmark --mode all
```

Each mode makes 40 Anthropic API calls:

```text
10 queries × (full answer + full judge + sliced answer + sliced judge)
```

The first LLM keyword run also makes additional Anthropic calls to generate the keyword index. Cross-encoder reranking runs locally, but the answer and judge calls still use Anthropic.

## Output Files

Manual:

```text
results/benchmark_results_manual.json
results/report_manual.txt
```

LLM-generated keywords:

```text
results/benchmark_results_llm.json
results/report_llm.txt
```

Manual keywords + cross-encoder:

```text
results/benchmark_results_cross.json
results/report_cross.txt
```

LLM keywords + cross-encoder:

```text
results/benchmark_results_llm_cross.json
results/report_llm_cross.txt
```

Partial checkpoint files are also written after every completed query. If credits run out or the run is interrupted, completed rows are preserved:

```text
results/benchmark_results_<mode>_partial.json
results/report_<mode>_partial.txt
```

## Consolidated Comparison

After running any modes, generate one comparison table:

```bash
pips-compare
```

This reads available benchmark JSON files and writes:

```text
results/comparison_summary.txt
```

The comparison table includes:

- average full-prompt input tokens
- average sliced-prompt input tokens
- average token reduction
- routing accuracy
- full-prompt answer score
- sliced-prompt answer score
- quality delta

## What The Metrics Mean

`Full Tok` is the exact Anthropic input token count when the full system prompt is sent.

`Slc Tok` is the exact Anthropic input token count when only selected sections are sent.

`Reduc%` is:

```text
(1 - sliced_tokens / full_tokens) × 100
```

`Routing` is correct when the expected section appears in the selected sections.

`FScr` is the LLM judge score for the full-prompt answer.

`SScr` is the LLM judge score for the sliced-prompt answer.

The judge compares:

```text
user query + ground-truth facts + AI answer
```

It is a semantic LLM-as-judge accuracy check, not string matching.

## Lost-In-The-Middle Interpretation

The full prompt baseline sends all 15 sections, so relevant information may be diluted or buried in the middle of a long context.

PageIndex slicing avoids this by selecting only the most relevant sections before generation.

Cross-encoder reranking is used before prompt construction. It improves which sections enter the sliced prompt; it does not repair the full prompt after it has already been sent.

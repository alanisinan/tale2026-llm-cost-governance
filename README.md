# Token-governance proof of concept (TALE 2026)

Materials, code, and raw data for the proof-of-concept measurement in
*When AI Bills Explode: Integrating LLM Cost Governance into Cybersecurity Education and Workforce Training*.

The study measures, for four cybersecurity learning tasks, how much Tier 1 (prompt) and Tier 1+3
(prompt + model routing) governance reduce token use and cost, and what happens to output quality
as scored against fixed answer keys.

## Tasks (`tasks/<task>/`)

| Task | Artifact | Answer key |
|---|---|---|
| `phishing` | Raw university-themed phishing email | Verdict + 8 planted indicators |
| `codereview` | Flask `app.py` for a lab-submission portal | 8 planted vulnerabilities |
| `incident` | SOC analyst notes for a phishing-to-exfiltration incident | 8 facts + 5 required report elements |
| `ioc` | Sandbox report for a fictional loader | 12 IOCs + 4 benign distractors (scored by F1) |

Each task folder contains:

- `core.txt`: the artifact the task needs.
- `extra_before.txt` / `extra_after.txt`: realistic but irrelevant material a learner might paste, used only in the naive condition. Examples are forwarded threads, full message headers, unrelated repository files, shift logs, and vendor boilerplate.
- `naive_prompt.txt` / `governed_prompt.txt`: the two prompt styles.
- `key.json`: the answer key.

All organizations, domains, hashes, and people are fictional. IP addresses come from the documentation
ranges (RFC 5737), apart from 8.8.8.8, which is used as a benign distractor.

## Conditions

| Code | Name | Model | Prompt and context | Output controls |
|---|---|---|---|---|
| N | Naive | Flagship | Verbose learner-style request plus the full pasted context | None; provider-default reasoning |
| P | Prompt-governed (Tier 1) | Flagship | Lean prompt plus trimmed context | Structured output with length limit, `max_tokens` cap, minimal reasoning budget where reasoning is on by default |
| F | Full-governed (Tier 1+3) | Low-cost | Same as P | Same as P |

The naive and governed prompts ask for the same core deliverable. The answer-key items are present in
both contexts.

Model pairs (flagship → low-cost), all accessed through OpenRouter:

| Provider | Flagship | Low-cost |
|---|---|---|
| OpenAI | GPT-4o | GPT-4o-mini |
| Anthropic | Claude Sonnet 4.6 | Claude Haiku 4.5 |
| Google | Gemini 2.5 Pro | Gemini 2.5 Flash |
| xAI | Grok 4.7 | Grok 4.3 |
| Meta | Llama 4 Maverick | Llama 4 Scout |

Settings live in `run.py`: `PROVIDERS`, `REASONING_BOUND`, and `ANSWER_CAP`.

## Scoring

`judge.py` sends each output to two judges from different vendors: `openai/gpt-5.4-mini` and
`google/gemini-3.8-flash`. The judges are blind to model and condition. Each returns one binary
decision per answer-key item (and per distractor for the IOC task). Quality is computed per task:

- Phishing: fraction of indicators found, multiplied by verdict correctness.
- Code review and incident report: fraction of key items found.
- IOC extraction: F1, where false positives are distractors reported as malicious.

The final quality score is the mean of the two judges.

The judge instructions were revised once, during the 15-call pilot and before the main run. One judge
had been treating reference details in parentheses (for example, a domain name inside a conceptual
phishing indicator) as required. The fix was to give conceptual items and exact-value items (IOCs)
separate rules. The final rules in `judge.py` were fixed before the main run and applied unchanged to
all 180 outputs. `analyze.py` reports inter-judge agreement
(percent agreement and Cohen's kappa). It also runs a deterministic string-match check of IOC recall
against the judges.

## Reproduce

```bash
echo "OPENROUTER_API_KEY=sk-or-..." > .env              # never commit this file
python3 run.py --runs 3                                 # 180 calls -> results/generations.jsonl
python3 judge.py                                        # 360 judgments -> results/judgments.jsonl
python3 -m venv .venv && .venv/bin/pip install matplotlib
.venv/bin/python analyze.py                             # results/summary.json, per_cell.csv, figure
```

Every raw response is logged with its provider-reported token usage and cost, the upstream provider,
the generation id, and the finish reason. `results/pilot*.jsonl` holds the 15-call pilot used to
calibrate the protocol; it is not included in the reported results.

## Citation

S. A. Noman and H. A. Noman, "When AI Bills Explode: Integrating LLM Cost Governance into
Cybersecurity Education and Workforce Training," in *Proc. IEEE Int. Conf. Teaching, Assessment,
and Learning for Engineering (TALE)*, 2026.

```bibtex
@inproceedings{noman2026aibills,
  author    = {Noman, Sinan Ameen and Noman, Haitham Ameen},
  title     = {When {AI} Bills Explode: Integrating {LLM} Cost Governance into Cybersecurity
               Education and Workforce Training},
  booktitle = {Proc. IEEE Int. Conf. Teaching, Assessment, and Learning for Engineering (TALE)},
  year      = {2026}
}
```

Authors: Sinan Ameen Noman (University of Alabama at Tuscaloosa, USA) and
Haitham Ameen Noman (Princess Sumaya University for Technology, Jordan).

## License

- Code (`run.py`, `judge.py`, `analyze.py`): MIT; see [`LICENSE`](LICENSE).
- Task materials, answer keys, and results (`tasks/`, `results/`): CC BY 4.0; see [`LICENSE-DATA`](LICENSE-DATA).

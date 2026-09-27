"""Blind answer-key grading of PoC outputs by two independent LLM judges (OpenRouter).

Each judge sees only the task description, the fixed answer key, and the response text; it never
sees the model name or condition. Judges return one binary decision per key item (and, for the
IOC task, per benign distractor). Results are appended to results/judgments.jsonl.

Usage: python3 judge.py [--gen results/generations.jsonl] [--out results/judgments.jsonl]
"""
import argparse
import json
import re
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from run import ROOT, call, load_env

JUDGES = ["openai/gpt-5.4-mini", "google/gemini-3.8-flash"]

CONCEPT_RULES = """- Mark an item 1 if the response clearly identifies the substance of the item. Paraphrase is fine.
- Names, values, and details in parentheses inside an item are reference information for you; the
  response does NOT need to repeat them (e.g. "Reply-To points to an unrelated external domain" is enough).
- Mark an item 0 if it is missing, only generically implied (e.g. "suspicious link" without saying the
  link text and target differ), or stated incorrectly (any specific value the response gives must be correct)."""

VALUE_RULES = """- Each item is a specific indicator value. Mark it 1 only if that exact value appears in the response
  as a malicious indicator (defanged forms such as [.] or hxxp are fine; case does not matter).
- Mark it 0 if the value is missing, truncated, or altered."""

TEMPLATE = """You are grading a response to a cybersecurity learning task against a fixed answer key.
You do not know who or what wrote the response.

TASK: {description}

ANSWER KEY ITEMS:
{items}
{extra}
GRADING RULES:
{rules}
- Ignore writing style, length, formatting, and any content not covered by the key.

Return JSON only, exactly in this shape:
{shape}

RESPONSE TO GRADE:
<<<
{content}
>>>"""


def build(task, content):
    key = json.loads((ROOT / "tasks" / task / "key.json").read_text())
    items = "\n".join(f"- {it['id']}: {it['text']}" for it in key["items"])
    shape = {"items": {it["id"]: "0 or 1" for it in key["items"]}}
    extra = ""
    if key.get("distractors"):
        extra = ("\nBENIGN DISTRACTORS (must NOT be reported as malicious IOCs). Mark 1 if the response "
                 "presents the item as a malicious indicator/IOC; mark 0 if it is absent or described as "
                 "benign/legitimate:\n"
                 + "\n".join(f"- {d['id']}: {d['text']}" for d in key["distractors"]) + "\n")
        shape["distractors"] = {d["id"]: "0 or 1" for d in key["distractors"]}
    if key.get("verdict"):
        extra += ("\nVERDICT: also mark verdict_phishing 1 if the response's overall conclusion is that "
                  "the email is phishing/malicious, else 0.\n")
        shape["verdict_phishing"] = "0 or 1"
    rules = VALUE_RULES if key.get("distractors") else CONCEPT_RULES
    return TEMPLATE.format(description=key["description"], items=items, extra=extra, rules=rules,
                           shape=json.dumps(shape), content=content)


def parse(text):
    text = text or ""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def judge_one(rec, judge):
    body = {"model": judge, "messages": [{"role": "user", "content": build(rec["task"], rec["content"])}],
            "usage": {"include": True}, "reasoning": {"effort": "low"},
            "response_format": {"type": "json_object"}, "max_tokens": 4000}
    if judge.startswith("google/"):
        body["temperature"] = 0
    errors = []
    for attempt in range(1, 4):
        try:
            data = call(body, timeout=240)
            msg = data["choices"][0]["message"].get("content") or ""
            verdict = parse(msg)
            return {"key": rec["key"], "judge": judge, "result": verdict,
                    "cost_usd": (data.get("usage") or {}).get("cost"), "attempts": attempt}
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError, IndexError) as e:
            errors.append(f"attempt {attempt}: {type(e).__name__} {str(e)[:200]}")
            time.sleep(5 * attempt)
    return {"key": rec["key"], "judge": judge, "failed": True, "errors": errors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="results/generations.jsonl")
    ap.add_argument("--out", default="results/judgments.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    load_env()

    gens = [json.loads(l) for l in (ROOT / args.gen).read_text().splitlines()]
    gens = [g for g in gens if not g.get("failed")]
    out = ROOT / args.out
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if not r.get("failed"):
                done.add((r["key"], r["judge"]))
    jobs = [(g, j) for g in gens for j in JUDGES if (g["key"], j) not in done]
    print(f"{len(jobs)} judgments to run ({len(done)} done)")
    lock, spent = threading.Lock(), [0.0]
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out.open("a") as fh:
        futs = [pool.submit(judge_one, g, j) for g, j in jobs]
        for f in as_completed(futs):
            r = f.result()
            with lock:
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                spent[0] += r.get("cost_usd") or 0
            status = "FAILED " + r["errors"][-1] if r.get("failed") else json.dumps(r["result"])[:110]
            print(f"{r['key']:30s} {r['judge']:26s} {status}  total=${spent[0]:.3f}")
    print(f"done. judge spend ${spent[0]:.4f}")


if __name__ == "__main__":
    main()

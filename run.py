"""Proof-of-concept token-governance measurement harness (OpenRouter).

Runs 4 cybersecurity learning tasks x 5 provider families x 3 conditions x N runs and
logs every raw response with provider-reported token usage and cost.

Conditions
  N  naive           flagship model, verbose prompt, full (padded) context, no output limits,
                     provider-default reasoning
  P  prompt-governed Tier 1: flagship model, lean prompt, trimmed context, structured output with a
                     length limit, max-token cap, minimal reasoning budget where reasoning is on by default
  F  full-governed   Tier 1 + Tier 3: same as P but routed to the provider's low-cost model

Usage: python3 run.py [--runs 3] [--providers openai,...] [--tasks phishing,...] [--conds N,P,F]
                      [--out results/generations.jsonl] [--budget 11.0] [--workers 8]
"""
import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API = "https://openrouter.ai/api/v1/chat/completions"

PROVIDERS = {
    "openai":    {"flagship": "openai/gpt-4o",               "lowcost": "openai/gpt-4o-mini"},
    "anthropic": {"flagship": "anthropic/claude-sonnet-4.6",  "lowcost": "anthropic/claude-haiku-4.5"},
    "google":    {"flagship": "google/gemini-2.5-pro",        "lowcost": "google/gemini-2.5-flash"},
    "xai":       {"flagship": "x-ai/grok-4.7",                "lowcost": "x-ai/grok-4.3"},
    "meta":      {"flagship": "meta-llama/llama-4-maverick",  "lowcost": "meta-llama/llama-4-scout"},
}

# How to bound reasoning for models that reason by default (governed conditions only).
REASONING_BOUND = {
    "google/gemini-2.5-pro":   ({"max_tokens": 512}, 1024),
    "google/gemini-2.5-flash": ({"max_tokens": 512}, 1024),
    "x-ai/grok-4.7":           ({"effort": "low"}, 2000),
    "x-ai/grok-4.3":           ({"effort": "low"}, 2000),
}

# Answer-length cap (max_tokens) per task for governed conditions.
ANSWER_CAP = {"phishing": 300, "codereview": 700, "incident": 600, "ioc": 600}
TASKS = ["phishing", "codereview", "incident", "ioc"]
CONDS = ["N", "P", "F"]


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY missing (put it in poc/.env)")


def read(task, name):
    return (ROOT / "tasks" / task / name).read_text()


def build_prompt(task, cond):
    core = read(task, "core.txt")
    if cond == "N":
        context = read(task, "extra_before.txt") + core + read(task, "extra_after.txt")
        return read(task, "naive_prompt.txt").replace("{CONTEXT}", context)
    return read(task, "governed_prompt.txt").replace("{CONTEXT}", core)


def build_request(provider, task, cond):
    model = PROVIDERS[provider]["lowcost" if cond == "F" else "flagship"]
    body = {
        "model": model,
        "messages": [{"role": "user", "content": build_prompt(task, cond)}],
        "usage": {"include": True},
    }
    if cond in ("P", "F"):
        cap = ANSWER_CAP[task]
        if model in REASONING_BOUND:
            reasoning, headroom = REASONING_BOUND[model]
            body["reasoning"] = reasoning
            cap += headroom
        body["max_tokens"] = cap
    return body


def call(body, timeout=900):
    req = urllib.request.Request(
        API,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
            "X-Title": "TALE2026 token-governance PoC",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def run_one(provider, task, cond, run):
    body = build_request(provider, task, cond)
    errors = []
    for attempt in range(1, 4):
        t0 = time.time()
        try:
            data = call(body)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
            detail = e.read().decode()[:300] if isinstance(e, urllib.error.HTTPError) else ""
            errors.append(f"attempt {attempt}: {type(e).__name__} {e} {detail}".strip())
            time.sleep(5 * attempt)
            continue
        latency = time.time() - t0
        if "error" in data or not data.get("choices"):
            errors.append(f"attempt {attempt}: API error {json.dumps(data.get('error'))[:300]}")
            time.sleep(5 * attempt)
            continue
        usage = data.get("usage") or {}
        choice = data["choices"][0]
        content = (choice.get("message") or {}).get("content") or ""
        if not usage.get("completion_tokens") and not content:
            errors.append(f"attempt {attempt}: zero-token empty response")
            time.sleep(5 * attempt)
            continue
        details = usage.get("completion_tokens_details") or {}
        return {
            "key": f"{provider}|{task}|{cond}|{run}",
            "provider": provider, "task": task, "cond": cond, "run": run,
            "model": body["model"], "served_model": data.get("model"),
            "upstream": data.get("provider"), "generation_id": data.get("id"),
            "request": {k: v for k, v in body.items() if k != "messages"},
            "prompt_chars": len(body["messages"][0]["content"]),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": details.get("reasoning_tokens", 0) or 0,
            "total_tokens": usage.get("total_tokens", 0),
            "cost_usd": usage.get("cost"),
            "finish_reason": choice.get("finish_reason"),
            "native_finish_reason": choice.get("native_finish_reason"),
            "latency_s": round(latency, 2),
            "attempts": attempt, "errors": errors,
            "content": content,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    return {"key": f"{provider}|{task}|{cond}|{run}", "provider": provider, "task": task,
            "cond": cond, "run": run, "model": body["model"], "failed": True, "errors": errors,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--providers", default=",".join(PROVIDERS))
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--out", default="results/generations.jsonl")
    ap.add_argument("--budget", type=float, default=11.0, help="stop scheduling new calls above this spend (USD)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    load_env()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done, spent = set(), 0.0
    if out.exists():
        for line in out.read_text().splitlines():
            rec = json.loads(line)
            if not rec.get("failed"):
                done.add(rec["key"])
                spent += rec.get("cost_usd") or 0

    jobs = [(p, t, c, r) for r in range(1, args.runs + 1) for p in args.providers.split(",")
            for t in args.tasks.split(",") for c in args.conds.split(",")
            if f"{p}|{t}|{c}|{r}" not in done]
    print(f"{len(jobs)} calls to run ({len(done)} already done, ${spent:.4f} spent so far)")

    lock = threading.Lock()
    state = {"spent": spent}

    def guarded(job):
        with lock:
            if state["spent"] > args.budget:
                return None
        return run_one(*job)

    with ThreadPoolExecutor(max_workers=args.workers) as pool, out.open("a") as fh:
        futures = [pool.submit(guarded, j) for j in jobs]
        for fut in as_completed(futures):
            rec = fut.result()
            if rec is None:
                continue
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                state["spent"] += rec.get("cost_usd") or 0
            if rec.get("failed"):
                print(f"FAILED {rec['key']}: {rec['errors'][-1] if rec['errors'] else ''}")
            else:
                print(f"{rec['key']:32s} {rec['model']:30s} tok={rec['total_tokens']:>6} "
                      f"(in {rec['prompt_tokens']}, out {rec['completion_tokens']}, rsn {rec['reasoning_tokens']}) "
                      f"${(rec['cost_usd'] or 0):.4f} {rec['finish_reason']} total=${state['spent']:.3f}")
    print(f"done. total spend ${state['spent']:.4f}")


if __name__ == "__main__":
    main()

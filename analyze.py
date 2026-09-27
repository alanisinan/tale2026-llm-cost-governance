"""Aggregate PoC generations + judgments into paper tables, summary stats, and the trade-off figure.

Usage: .venv/bin/python analyze.py   (needs matplotlib for the figure)
Outputs: results/summary.json, results/per_cell.csv, ../fig_tradeoff.pdf
"""
import csv
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

from run import PROVIDERS, TASKS, CONDS, ROOT
from judge import JUDGES

PROVIDER_LABEL = {"openai": "OpenAI", "anthropic": "Anthropic", "google": "Google", "xai": "xAI", "meta": "Meta"}
MODEL_LABEL = {
    "openai/gpt-4o": "GPT-4o", "openai/gpt-4o-mini": "GPT-4o-mini",
    "anthropic/claude-sonnet-4.6": "Claude Sonnet 4.6", "anthropic/claude-haiku-4.5": "Claude Haiku 4.5",
    "google/gemini-2.5-pro": "Gemini 2.5 Pro", "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "x-ai/grok-4.7": "Grok 4.7", "x-ai/grok-4.3": "Grok 4.3",
    "meta-llama/llama-4-maverick": "Llama 4 Maverick", "meta-llama/llama-4-scout": "Llama 4 Scout",
}
COND_LABEL = {"N": "Naive", "P": "Prompt-governed (Tier 1)", "F": "Full-governed (Tier 1+3)"}


def load(path):
    p = ROOT / path
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def score(task, res):
    """Quality in [0,1] from one judge's result."""
    items = [int(v) for v in res["items"].values()]
    recall = sum(items) / len(items)
    if task == "phishing":
        return recall * int(res.get("verdict_phishing", 1))
    if task == "ioc":
        tp, fp = sum(items), sum(int(v) for v in res.get("distractors", {}).values())
        if tp == 0:
            return 0.0
        precision = tp / (tp + fp)
        return 2 * precision * recall / (precision + recall)
    return recall


def refang(text):
    t = text.lower().replace("[.]", ".").replace("(.)", ".").replace("[:]", ":")
    t = re.sub(r"hxxp", "http", t)
    return t.replace("\\\\", "\\")


def ioc_recall_deterministic(content):
    key = json.loads((ROOT / "tasks" / "ioc" / "key.json").read_text())
    t = refang(content)
    hits = [any(m.lower() in t for m in it["match"]) for it in key["items"]]
    return sum(hits) / len(hits)


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    gens = [g for g in load("results/generations.jsonl") if not g.get("failed")]
    judg = defaultdict(dict)
    for j in load("results/judgments.jsonl"):
        if not j.get("failed"):
            judg[j["key"]][j["judge"]] = j["result"]

    # Per-generation quality (mean of available judges) + judge agreement.
    pairs, det_vs_judge = [], []
    for g in gens:
        res = judg.get(g["key"], {})
        scores = [score(g["task"], res[j]) for j in JUDGES if j in res]
        g["quality"] = st.mean(scores) if scores else None
        g["n_judges"] = len(scores)
        if all(j in res for j in JUDGES):
            a, b = res[JUDGES[0]], res[JUDGES[1]]
            for sect in ("items", "distractors"):
                for k in a.get(sect, {}):
                    if k in b.get(sect, {}):
                        pairs.append((int(a[sect][k]), int(b[sect][k])))
            if "verdict_phishing" in a and "verdict_phishing" in b:
                pairs.append((int(a["verdict_phishing"]), int(b["verdict_phishing"])))
        if g["task"] == "ioc" and res:
            jr = st.mean(sum(int(v) for v in res[j]["items"].values()) / 12 for j in res)
            det_vs_judge.append((ioc_recall_deterministic(g["content"]), jr))

    # Per (provider, task, cond) cell means over runs.
    cells = defaultdict(list)
    for g in gens:
        cells[(g["provider"], g["task"], g["cond"])].append(g)

    def cm(p, t, c, f):
        vals = [f(g) for g in cells.get((p, t, c), []) if f(g) is not None]
        return st.mean(vals) if vals else None

    rows = []
    for p in PROVIDERS:
        for t in TASKS:
            for c in CONDS:
                gs = cells.get((p, t, c), [])
                if not gs:
                    continue
                rows.append({
                    "provider": p, "task": t, "cond": c, "model": gs[0]["model"], "n": len(gs),
                    "tokens": cm(p, t, c, lambda g: g["total_tokens"]),
                    "tokens_sd": st.stdev([g["total_tokens"] for g in gs]) if len(gs) > 1 else 0,
                    "prompt_tokens": cm(p, t, c, lambda g: g["prompt_tokens"]),
                    "completion_tokens": cm(p, t, c, lambda g: g["completion_tokens"]),
                    "reasoning_tokens": cm(p, t, c, lambda g: g["reasoning_tokens"]),
                    "cost": cm(p, t, c, lambda g: g["cost_usd"]),
                    "quality": cm(p, t, c, lambda g: g["quality"]),
                    "truncated": sum(g["finish_reason"] == "length" for g in gs),
                })
    with (ROOT / "results" / "per_cell.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cell = {(r["provider"], r["task"], r["cond"]): r for r in rows}

    def red(a, b):
        return 1 - b / a

    # Provider x condition summary (mean over the 4 tasks of per-task cell means).
    prov = {}
    for p in PROVIDERS:
        prov[p] = {}
        for c in CONDS:
            rs = [cell[(p, t, c)] for t in TASKS if (p, t, c) in cell]
            if len(rs) < len(TASKS):
                continue
            prov[p][c] = {
                "model": MODEL_LABEL[rs[0]["model"]],
                "tokens": st.mean(r["tokens"] for r in rs),
                "cost": st.mean(r["cost"] for r in rs),
                "quality": st.mean(r["quality"] for r in rs) if all(r["quality"] is not None for r in rs) else None,
                "truncated": sum(r["truncated"] for r in rs),
            }
        if all(c in prov[p] for c in CONDS):
            for c in ("P", "F"):
                prov[p][c]["token_reduction"] = st.mean(
                    red(cell[(p, t, "N")]["tokens"], cell[(p, t, c)]["tokens"]) for t in TASKS)
                prov[p][c]["cost_reduction"] = st.mean(
                    red(cell[(p, t, "N")]["cost"], cell[(p, t, c)]["cost"]) for t in TASKS)

    complete = [p for p in PROVIDERS if all(c in prov[p] for c in CONDS)]
    overall = {}
    for c in ("P", "F"):
        tr = [red(cell[(p, t, "N")]["tokens"], cell[(p, t, c)]["tokens"]) for p in complete for t in TASKS]
        cr = [red(cell[(p, t, "N")]["cost"], cell[(p, t, c)]["cost"]) for p in complete for t in TASKS]
        dq = [cell[(p, t, c)]["quality"] - cell[(p, t, "N")]["quality"] for p in complete for t in TASKS
              if cell[(p, t, c)]["quality"] is not None and cell[(p, t, "N")]["quality"] is not None]
        overall[c] = {"token_reduction_mean": st.mean(tr) if tr else None,
                      "token_reduction_min": min(tr) if tr else None, "token_reduction_max": max(tr) if tr else None,
                      "cost_reduction_mean": st.mean(cr) if cr else None,
                      "quality_delta_mean": st.mean(dq) if dq else None,
                      "quality_delta_min": min(dq) if dq else None, "quality_delta_max": max(dq) if dq else None}
    for c in CONDS:
        qs = [cell[(p, t, c)]["quality"] for p in complete for t in TASKS if cell[(p, t, c)]["quality"] is not None]
        overall.setdefault(c, {})["quality_mean"] = st.mean(qs) if qs else None

    task_q = {t: {c: st.mean(cell[(p, t, c)]["quality"] for p in complete) for c in CONDS}
              for t in TASKS if complete and all(cell[(p, t, c)]["quality"] is not None for p in complete for c in CONDS)}

    summary = {
        "n_generations": len(gens), "n_judged": sum(1 for g in gens if g["n_judges"] == 2),
        "spend_generation_usd": sum(g["cost_usd"] or 0 for g in gens),
        "spend_judging_usd": sum((j.get("cost_usd") or 0) for j in load("results/judgments.jsonl")),
        "retried_calls": sum(1 for g in gens if g["attempts"] > 1),
        "truncated_calls": [g["key"] for g in gens if g["finish_reason"] == "length"],
        "judge_agreement": dict(zip(("percent", "kappa"), kappa(pairs))) if pairs else None,
        "judge_decisions": len(pairs),
        "ioc_det_vs_judge_recall_mad": st.mean(abs(a - b) for a, b in det_vs_judge) if det_vs_judge else None,
        "providers": prov, "overall": overall, "task_quality": task_q,
    }
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "providers"}, indent=2))
    for p in PROVIDERS:
        for c in CONDS:
            if c in prov[p]:
                d = prov[p][c]
                q = f"{d['quality'] * 100:5.1f}%" if d["quality"] is not None else "  n/a"
                print(f"{p:10s} {c} {d['model']:18s} tok={d['tokens']:8.0f} cost=${d['cost']:.4f} q={q} "
                      f"tokred={d.get('token_reduction', 0) * 100:5.1f}% trunc={d['truncated']}")
    if len(complete) == len(PROVIDERS) and all(prov[p][c]["quality"] is not None for p in PROVIDERS for c in CONDS):
        write_tables(prov, task_q, cell)
    if len(complete) == len(PROVIDERS) and all(prov[p][c]["quality"] is not None for p in PROVIDERS for c in CONDS):
        plot(prov)


TASK_LABEL = {"phishing": "Phishing analysis", "codereview": "Secure code review",
              "incident": "Incident report", "ioc": "IOC extraction (F1)"}


def write_tables(prov, task_q, cell):
    """Emit LaTeX rows for the paper so every number comes straight from the data."""
    short = {"N": "N", "P": "P", "F": "F"}
    lines = []
    for p in PROVIDERS:
        for i, c in enumerate(CONDS):
            d = prov[p][c]
            red = "--" if c == "N" else f"{d['token_reduction'] * 100:.1f}\\%"
            first = PROVIDER_LABEL[p] if i == 0 else ""
            lines.append(f"{first} & {short[c]} & {d['model']} & {d['tokens']:,.0f} & {red} & "
                         f"{d['cost'] * 100:.2f} & {d['quality'] * 100:.1f} \\\\")
        if p != list(PROVIDERS)[-1]:
            lines.append("\\addlinespace[1pt]")
    (ROOT / "results" / "table_poc_rows.tex").write_text("\n".join(lines) + "\n")
    tl = []
    for t in TASKS:
        red_p = st.mean(1 - cell[(p, t, "P")]["tokens"] / cell[(p, t, "N")]["tokens"] for p in PROVIDERS)
        red_f = st.mean(1 - cell[(p, t, "F")]["tokens"] / cell[(p, t, "N")]["tokens"] for p in PROVIDERS)
        q = task_q[t]
        tl.append(f"{TASK_LABEL[t]} & {q['N'] * 100:.1f} & {q['P'] * 100:.1f} & {q['F'] * 100:.1f} & "
                  f"{red_p * 100:.1f}\\% & {red_f * 100:.1f}\\% \\\\")
    (ROOT / "results" / "table_task_rows.tex").write_text("\n".join(tl) + "\n")
    print("tables written")


def plot(prov):
    """Trade-off frontier as small multiples: one panel per provider, shared axes.

    x = cost per task relative to naive (log), y = answer-key quality change vs. naive (points).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
        "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 7, "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    colors = {"N": "#2a78d6", "P": "#eb6834", "F": "#1baf7a"}
    markers = {"N": "o", "P": "s", "F": "^"}
    fig, axes = plt.subplots(1, len(prov), figsize=(7.16, 1.5), sharey=True)
    for ax, (p, d) in zip(axes, prov.items()):
        xs = [100.0] + [100 * (1 - d[c]["cost_reduction"]) for c in ("P", "F")]
        ys = [0.0] + [100 * (d[c]["quality"] - d["N"]["quality"]) for c in ("P", "F")]
        ax.axhline(0, color=ink2, lw=0.6, ls=(0, (3, 2)), zorder=0)
        ax.plot(xs, ys, color="#b9b8b2", lw=1.0, zorder=1)
        for c, x, y in zip(CONDS, xs, ys):
            ax.scatter(x, y, s=30, marker=markers[c], color=colors[c], edgecolors="white", linewidths=0.8, zorder=3)
        ax.set_title(f"{PROVIDER_LABEL[p]}", color=ink, pad=3)
        ax.text(0.03, 0.97, f"N: {d['N']['model']}\nF: {d['F']['model']}", transform=ax.transAxes,
                fontsize=5.8, color=ink2, va="top", ha="left", linespacing=1.1)
        ax.set_xscale("log")
        ax.set_xlim(1.0, 180)
        ax.set_ylim(-13, 8)
        ax.xaxis.set_major_locator(FixedLocator([1, 3, 10, 30, 100]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
        ax.yaxis.set_major_locator(FixedLocator([-10, -5, 0, 5]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+g}" if v else "0"))
        ax.grid(True, which="major", color=grid, lw=0.5)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color(ink2)
        ax.tick_params(colors=ink2, length=2.5, pad=1.5)
    axes[0].set_ylabel("Quality change (points)", color=ink)
    fig.supxlabel("Cost per task relative to naive (log scale)", fontsize=7.5, color=ink, y=0.02)
    handles = [plt.Line2D([], [], marker=markers[c], color=colors[c], ls="", ms=5, mec="white", mew=0.6,
                          label=COND_LABEL[c]) for c in CONDS]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, handletextpad=0.3,
               columnspacing=1.2, bbox_to_anchor=(0.995, 1.01))
    fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0.02, 1, 0.9))
    out = ROOT.parent / "fig_tradeoff.pdf"
    fig.savefig(out)
    fig.savefig(ROOT / "results" / "fig_tradeoff.png", dpi=300)
    print("figure written:", out)


if __name__ == "__main__":
    main()

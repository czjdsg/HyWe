#!/usr/bin/env python3
"""v7 Stage B: probe each O-side fact with a browser-less search agent.

For every probe-worthy fact mined in Stage A (explore-only run):
  1. QUESTION-FORM  (deepseek-v4-pro): turn the fact into a question that
     does NOT leak the answer value.
  2. S BLIND-ANSWER (qwen3.7-max @ :8000): a search-only agent (search /
     scholar / visit, NO browser) gets <=5 tool calls to answer it.
  3. JUDGE          (deepseek-v4-pro): compare the agent's answer to the
     ground-truth value observed live on the site ->
         S_REACHABLE  - search conveys the same info (semantic match;
                        live values: same offer / reasonable drift)
         O_EXCLUSIVE  - not found, or only vague aggregates vs a specific
                        live value, or wrong entity

Facts are pre-filtered in code: UI chrome and CAPTCHA logs are skipped
outright; purely visual facts are tagged O_EXCLUSIVE without probing.

Pure-Python threadpool (one thread per fact) — deliberately NOT an agent
workflow: this is data plumbing, and LLM orchestrators get lazy with it.

Usage:
  source ../env.local.sh   # serper/jina pools!
  python3 fact_prober.py --run /tmp/synth7_poc [--conc 16] [--max-probes 10]
Outputs per site: <run>/<domain>/fact_tags.json
Aggregate:        <run>/fact_tags_index.jsonl
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime
import json
import os
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))
from hybrid_search import SearchTool      # noqa: E402
from hybrid_scholar import ScholarTool    # noqa: E402
from hybrid_visit import VisitTool        # noqa: E402

# ---------------- endpoints ----------------
S_AGENT_BASE = os.environ.get("PROBE_AGENT_BASE", "http://localhost:8000/v1")
S_AGENT_MODEL = os.environ.get("PROBE_AGENT_MODEL", "kimi-k3")
FLASH_BASE = os.environ.get("PROBE_FLASH_BASE", "http://localhost:8000/v1")
FLASH_MODEL = os.environ.get("PROBE_FLASH_MODEL", "kimi-k3")
API_KEY = os.environ.get("PROBE_API_KEY", "EMPTY")
MAX_PROBE_STEPS = int(os.environ.get("PROBE_MAX_STEPS", "5"))

# ---------------- fact filters ----------------
BANNED_KEY_RE = re.compile(
    r"(_location|_visible|_button|_coords|_position|_placeholder|_tabs|"
    r"_menu|_layout|_redirect|_behavior)(_v\d+)?$")
CAPTCHA_RE = re.compile(
    r"captcha|verification_challenge|robot_check|anti_bot|press_hold|"
    r"automation_block|rate_limit", re.I)
# Walled-off / anti-bot DIAGNOSTICS recorded as facts (the explorer reframes
# banned captcha logs as "access_status" / "403_reference_id" / "wall_scope"
# keys, dodging CAPTCHA_RE). The block itself is never a task-worthy fact —
# nobody asks a research assistant "what is the 403 reference ID".
BLOCKED_RE = re.compile(
    r"access_denied|access_blocked|access_status|_403\b|403_|_blocked|"
    r"blocked_|_block\b|_wall|wall_scope|verification_wall|reference_id|"
    r"err_connection|connection_closed|browsing_activity|bot_detect|"
    r"akamai|cloudflare_challenge|page_unavailable", re.I)
# value must carry a number / currency / multi-word proper noun to be worth probing
SIGNAL_RE = re.compile(r"\d|[$€£¥]|[A-Z][a-z]+ [A-Z][a-z]")


def classify_fact(key: str, value: str, visual: bool = False) -> str:
    """-> 'probe' | 'auto_o' (skip the probe — fact came from a visual clue) |
    'skip' (not task-worthy).

    auto_o is decided SOLELY by the extractor's structured `visual` flag (the
    fact was derived from rendered imagery/layout that search cannot recover),
    NEVER by string-matching the value text. A fact that merely MENTIONS an
    image but carries searchable text (title/name/price) has visual=False and
    is probed."""
    if CAPTCHA_RE.search(key) or CAPTCHA_RE.search(value):
        return "skip"
    if BLOCKED_RE.search(key) or BLOCKED_RE.search(value):
        return "skip"
    if BANNED_KEY_RE.search(key):
        return "skip"
    if visual:
        return "auto_o"
    if SIGNAL_RE.search(value):
        return "probe"
    return "skip"


def richness(value: str) -> int:
    """Prioritization score when capping probes per site."""
    score = 0
    score += 5 * len(re.findall(r"[$€£¥]\s?[\d,]+", value))
    score += 2 * len(re.findall(r"\d[\d,]*", value))
    score += len(re.findall(r"[A-Z][a-z]+", value))
    return score


# ---------------- LLM plumbing ----------------
def call_llm(base: str, model: str, messages: list, *, max_tokens: int = 2000,
             temperature: float = 0.2, timeout: int = 900, retries: int = 3) -> str:
    payload = json.dumps({
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
    }).encode()
    last = ""
    for attempt in range(retries):
        req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                     data=payload, method="POST")
        req.add_header("Authorization", f"Bearer {API_KEY}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode("utf-8", errors="replace"))
                return d["choices"][0]["message"]["content"] or ""
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2 ** attempt * 2, 15))
    raise RuntimeError(f"LLM call failed: {last}")


def extract_json(text: str) -> dict | None:
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    depth, start, in_str, esc = 0, -1, False, False
    for i, c in enumerate(text):
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


# ---------------- S-side tools (shared, thread-safe enough: stateless calls) --
def _flash_summarize(prompt_text: str, image_data_urls=None) -> str:
    try:
        return call_llm(FLASH_BASE, FLASH_MODEL,
                        [{"role": "system",
                          "content": "You extract useful page information for a web agent."},
                         {"role": "user", "content": prompt_text}],
                        max_tokens=4000, temperature=0.0)
    except Exception as exc:
        return f"[summary-fallback {type(exc).__name__}]\n" + prompt_text[:6000]


_TOOLS_LOCK = threading.Lock()
_TOOLS: dict = {}


def get_tools():
    with _TOOLS_LOCK:
        if not _TOOLS:
            _TOOLS["search"] = SearchTool()
            _TOOLS["scholar"] = ScholarTool()
            _TOOLS["visit"] = VisitTool(
                summarize_fn=_flash_summarize,
                tokenizer_path=os.environ.get("TOKENIZER_PATH", ""),
                max_webpage_tokens=8000,
            )
        return _TOOLS


# ---------------- stage 1: question-form ----------------
Q_FORM_PROMPT = """Today's date: {today}.

Convert this website-derived fact into ONE self-contained question
a person could pose to a research assistant WITHOUT a browser. The question
must NOT reveal or hint at the answer value. Include enough context (site
name, route, dates, product, section) that the question is unambiguous.

CRITICAL — time-anchored facts:
If the fact value or key references a relative time window ("This Past
Week", "today", "current top", "weekly popular", "this month", "latest",
"now showing"), the question MUST anchor it to TODAY's date — e.g.
"as of {today}, this past week" or "for the week ending {today}". Without
the anchor, the search agent cannot reproduce the same ranking, and a
correct answer will be wrongly judged stale.

CRITICAL — single-fact resolution:
The question must isolate THIS ONE fact, not a class of facts. If the key
encodes a specific position / rank / index / slot (top1, top_6, position_3,
property_4), the question MUST ask about THAT SINGLE rank only — e.g.
"which shot is at rank 6 in this filter" NOT "what are the top 6 shots".
Otherwise the answer becomes a list and cannot be matched against this one
ground-truth row. The question's expected answer must be ONE concrete row,
not a multi-row table.

Site: {domain}  ({entry_url})
Fact key: {key}
Fact value (ground truth — do NOT leak into the question): {value}

Return ONLY JSON: {{"question": "<the question>"}}"""


def question_form(domain: str, entry_url: str, key: str, value: str) -> str:
    today = datetime.date.today().isoformat()
    raw = call_llm(FLASH_BASE, FLASH_MODEL,
                   [{"role": "user", "content": Q_FORM_PROMPT.format(
                       domain=domain, entry_url=entry_url, key=key, value=value,
                       today=today)}],
                   max_tokens=400, temperature=0.0)
    obj = extract_json(raw) or {}
    q = (obj.get("question") or "").strip()
    if not q:
        raise RuntimeError("question_form: empty question")
    return q


# ---------------- stage 2: S blind-answer agent ----------------
S_AGENT_SYSTEM = """You are a research agent answering ONE question using ONLY these
tools (no browser, no clicking):
  search  - web search; args {"query": ["q1", "q2"]}
  scholar - academic search; args {"query": ["q"]}
  visit   - read a page via extractor; args {"url": "...", "goal": "..."}

You have AT MOST %d tool calls total. Respond each turn with EXACTLY ONE JSON object:
  {"action":"search","arguments":{"query":["..."]}}
  {"action":"scholar","arguments":{"query":["..."]}}
  {"action":"visit","arguments":{"url":"...","goal":"..."}}
  {"action":"answer","answer":"<concise answer with concrete values>","found":true|false}

Rules:
- Prefer 2 query variants per search call.
- Answer as soon as you have the information; do not burn calls.
- found=false if the specific information cannot be located via search.
No prose outside the JSON.""" % MAX_PROBE_STEPS


def s_probe(question: str) -> dict:
    """Run the <=5-step search agent. Returns {answer, found, steps, transcript}."""
    tools = get_tools()
    messages = [{"role": "system", "content": S_AGENT_SYSTEM},
                {"role": "user", "content": f"QUESTION: {question}"}]
    transcript: list[dict] = []
    steps = 0
    while True:
        raw = call_llm(S_AGENT_BASE, S_AGENT_MODEL, messages,
                       max_tokens=1200, temperature=0.2)
        obj = extract_json(raw) or {}
        action = obj.get("action")
        if action == "answer" or steps >= MAX_PROBE_STEPS:
            return {
                "answer": (obj.get("answer") or "").strip(),
                "found": bool(obj.get("found")) if action == "answer" else False,
                "steps": steps,
                "transcript": transcript,
            }
        args = obj.get("arguments") or {}
        try:
            if action == "search":
                q = args.get("query") or []
                if isinstance(q, str):
                    q = [q]
                result = tools["search"].call(q)
            elif action == "scholar":
                q = args.get("query") or []
                if isinstance(q, str):
                    q = [q]
                result = tools["scholar"].call(q)
            elif action == "visit":
                result = tools["visit"].call(args.get("url") or "",
                                             args.get("goal") or "answer the question")
            else:
                result = f"[invalid action {action!r} — use search/scholar/visit/answer]"
        except Exception as exc:
            result = f"[tool error {type(exc).__name__}: {exc}]"
        steps += 1
        result = str(result)[:2500]
        transcript.append({"step": steps, "action": action,
                           "args": args, "result_head": result[:400]})
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user",
                         "content": f"TOOL RESULT (call {steps}/{MAX_PROBE_STEPS}):\n{result}\n\n"
                                    "Answer now if you can; otherwise next tool call."})


# ---------------- stage 3: judge ----------------
JUDGE_PROMPT = """A fact was observed LIVE on a website using a real browser. A search
agent (no browser, only web search / page reader, {steps} tool calls) then tried
to answer the equivalent question. Decide whether search alone could deliver
this information.

QUESTION: {question}
GROUND TRUTH (live on {domain}): {value}
SEARCH AGENT ANSWER (found={found}): {answer}

Verdict rules:
- "S_REACHABLE": the search answer conveys the SAME information — semantic
  match. For volatile values (prices, live rates): same offer or same
  type at plausibly-drifted value still counts as reachable.
- Multi-attribute facts (e.g. "designer X — N likes, M views" or
  "hotel H — $P, R rating"): match by AGREEMENT across the discriminating
  attributes, NOT by literal name alignment. If the search answer names a
  different surface label (e.g. the SHOT TITLE instead of the DESIGNER
  NAME) but ALL the numeric/identifying attributes match exactly
  (same likes, same views; or same price, same rating), it IS the same
  underlying fact — return S_REACHABLE. The ground-truth value string may
  omit fields the search answer includes (and vice versa); fields that
  appear in BOTH must agree, fields present in only one are neutral.
- "O_EXCLUSIVE": search failed, or produced only vague aggregates ("fares
  from $577") when ground truth is a specific live datum (specific
  itinerary/config price), or named a DIFFERENT entity (mismatched on the
  discriminating attributes, not just on a label the truth value happened
  to use). Information that only exists in the site's interactive state
  (filter counts, configurator prices, live seat maps) is typically
  O_EXCLUSIVE.

Return ONLY JSON:
{{"tag":"S_REACHABLE"|"O_EXCLUSIVE",
  "reason":"<1-2 sentences citing the comparison>"}}"""


def judge(domain: str, question: str, value: str, probe: dict) -> dict:
    raw = call_llm(FLASH_BASE, FLASH_MODEL,
                   [{"role": "user", "content": JUDGE_PROMPT.format(
                       steps=probe["steps"], question=question, domain=domain,
                       value=value, found=probe["found"],
                       answer=probe["answer"] or "(no answer)")}],
                   max_tokens=800, temperature=0.0)
    obj = extract_json(raw) or {}
    tag = obj.get("tag")
    if tag not in ("S_REACHABLE", "O_EXCLUSIVE"):
        raise ValueError("Probe judge did not return a valid capability label")
    return {"tag": tag, "reason": (obj.get("reason") or "")[:400]}


# ---------------- per-fact pipeline ----------------
def probe_one(item: dict) -> dict:
    """item: {domain, entry_url, key, value, step, mode}"""
    t0 = time.time()
    out = dict(item)
    try:
        q = question_form(item["domain"], item["entry_url"], item["key"], item["value"])
        out["question"] = q
        pr = s_probe(q)
        out["s_answer"] = pr["answer"]
        out["s_found"] = pr["found"]
        out["s_steps"] = pr["steps"]
        out["s_transcript"] = pr["transcript"]
        verdict = judge(item["domain"], q, item["value"], pr)
        out.update(verdict)
        out["probe"] = "probed"
    except Exception as exc:
        out["probe"] = "error"
        out["tag"] = None  # Failed probing cannot establish a capability label.
        out["reason"] = f"probe error: {type(exc).__name__}: {exc}"
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Stage A output dir")
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--max-probes", type=int, default=10,
                    help="cap probed facts per site (by richness)")
    ap.add_argument("--limit-sites", type=int, default=0)
    args = ap.parse_args()

    run = Path(args.run)
    site_dirs = sorted(d for d in run.iterdir()
                       if d.is_dir() and (d / "result.json").is_file())
    if args.limit_sites:
        site_dirs = site_dirs[: args.limit_sites]

    # Build the flat probe worklist + per-site skeletons
    site_meta: dict[str, dict] = {}
    work: list[dict] = []
    for sd in site_dirs:
        res = json.loads((sd / "result.json").read_text())
        facts = ((res.get("kb") or {}).get("facts")) or {}
        domain = sd.name
        entry_url = (res.get("site") or {}).get("entry_url") or ""
        rows = []
        for k, v in facts.items():
            cls = classify_fact(k, v.get("value") or "", bool(v.get("visual")))
            rows.append({"domain": domain, "entry_url": entry_url,
                         "key": k, "value": v.get("value") or "",
                         "step": v.get("step"), "mode": v.get("mode"),
                         "probe": cls, "tag": None})
        # cap probes per site by richness
        probe_rows = [r for r in rows if r["probe"] == "probe"]
        probe_rows.sort(key=lambda r: -richness(r["value"]))
        for r in probe_rows[args.max_probes:]:
            r["probe"] = "skip_capped"
        for r in rows:
            if r["probe"] == "auto_o":
                r["tag"] = "O_EXCLUSIVE"
                r["reason"] = "visual fact — search cannot retrieve rendered imagery"
        site_meta[domain] = {"rows": rows, "entry_url": entry_url}
        work.extend(r for r in rows if r["probe"] == "probe")

    print(f"sites={len(site_dirs)}  total_facts="
          f"{sum(len(m['rows']) for m in site_meta.values())}  to_probe={len(work)}",
          flush=True)

    t0 = time.time()
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.conc) as pool:
        futs = {pool.submit(probe_one, dict(w)): w for w in work}
        for fut in cf.as_completed(futs):
            res = fut.result()
            w = futs[fut]
            w.update(res)
            done += 1
            if done % 10 == 0 or done == len(work):
                el = time.time() - t0
                print(f"[probe] {done}/{len(work)}  rate={done/max(el,0.01):.2f}/s  "
                      f"eta={(len(work)-done)/max(done/max(el,0.01),0.01):.0f}s", flush=True)

    # Write per-site outputs
    idx_path = run / "fact_tags_index.jsonl"
    with idx_path.open("w") as idx_f:
        for domain, meta in site_meta.items():
            rows = meta["rows"]
            counts = {
                "total": len(rows),
                "probed": sum(1 for r in rows if r.get("probe") == "probed"),
                "s_reachable": sum(1 for r in rows if r.get("tag") == "S_REACHABLE"),
                "o_exclusive": sum(1 for r in rows if r.get("tag") == "O_EXCLUSIVE"),
                "auto_o": sum(1 for r in rows if r.get("probe") == "auto_o"),
                "skipped": sum(1 for r in rows if (r.get("probe") or "").startswith("skip")),
                "errors": sum(1 for r in rows if r.get("probe") == "error"),
            }
            out = {"domain": domain, "facts": rows, "counts": counts}
            (run / domain / "fact_tags.json").write_text(
                json.dumps(out, ensure_ascii=False, indent=1))
            idx_f.write(json.dumps({"domain": domain, **counts},
                                   ensure_ascii=False) + "\n")
            print(f"  {domain}: {counts}", flush=True)
    print(f"DONE in {time.time()-t0:.0f}s -> {idx_path}", flush=True)


if __name__ == "__main__":
    main()

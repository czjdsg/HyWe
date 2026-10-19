#!/usr/bin/env python
"""screen_insta_sites.py — screen InSTA-150k websites for ones likely RICH in
O_EXCLUSIVE facts (facts obtainable ONLY by operating the browser, not by
search/visit). Each InSTA row (website / instruction / steps / criteria) carries a
prototypical task; we treat that task's characteristics as a proxy for the site's
O-richness and score 4 signals (the cases you named):

  A. MANY FILTER CONDITIONS  — faceted/complex filtering UIs; multi-constraint asks
     (e.g. "under $200 with pool and free parking near downtown").
  B. URL DOESN'T CAPTURE STATE — result appears after in-place / dynamic (AJAX/SPA)
     interaction, so a visit(url) can't reproduce it; or the URL is opaque.
  C. NO CONSTRUCTIBLE search/visit QUERY — the answer is behind clicks/selects/
     scrolls; it is a RESULT of interaction, not a static searchable fact.
  D. HEAVY VISUAL / REAL-TIME RENDER — charts, curves, maps, animations, live data;
     text extraction can't capture it, it must be seen/rendered.

Static info-lookup asks (definitions, encyclopedia, "what is …") are DOWN-weighted
(those are S-reachable, not O-rich).

Stage 1 (fast, all rows): heuristic signal score -> rank -> dedup by domain.
Stage 2 (optional --llm-judge): an LLM re-scores the top-K on the same 4-case rubric
to refine the final selection.

Output:
  <out>.pool.json   {"all_usable":[{domain,entry_url,category,source,web_name,o_score,signals}], "total"}
                    (input of collector/stage1_probe.py)
  <out>.detail.jsonl  every kept candidate with full signal breakdown (for inspection/tuning)

Requires pandas + pyarrow.
"""
from __future__ import annotations
import argparse, ast, json, os, re, sys, urllib.parse, urllib.request
from collections import Counter

# ---------------------------------------------------------------- signal lexicons
FILTER = ["filter", "filters", "filtering", "sort by", "sort", "sorted", "refine", "narrow down",
    "narrow", "facet", "faceted", "dropdown", "drop-down", "checkbox", "radio button", "slider",
    "price range", "date range", "range", "toggle", "tabs", "tab", "select", "selection", "options",
    "option", "availability", "in stock", "apply filter", "apply", "advanced search", "filter by",
    "group by", "between", "calendar", "date picker", "min ", "max ", "rating", "brand", "size", "color"]
INTERACT = ["click", "clicking", "expand", "collapse", "scroll", "hover", "drag", "toggle",
    "add to cart", "checkout", "submit", "fill in", "fill out", "enter the", "upload", "download",
    "play", "pause", "zoom", "pan", "load more", "pagination", "next page", "autocomplete",
    "switch to", "open the", "reveal", "interact", "select the", "choose the", "pick the"]
DYNAMIC = ["dynamically", "dynamic", "without navigating", "without changing the url", "same page",
    "in place", "in-place", "live update", "real-time", "real time", "realtime", "ajax",
    "updates automatically", "results update", "infinite scroll", "lazy load", "single page",
    "single-page", "interactive", "spa", "loads without", "no url change"]
VISUAL = ["chart", "charts", "graph", "graphs", "curve", "curves", "plot", "plots", "heatmap",
    "map", "maps", "visualization", "visualisation", "animation", "animated", "gif", "carousel",
    "gallery", "tooltip", "canvas", "svg", "diagram", "timeline", "gauge", "dashboard", "trend",
    "price history", "stock chart", "candlestick", "interactive map", "3d", "ticker", "scoreboard",
    "live score", "weather map", "radar", "seat map", "floor plan", "slider", "widget", "heat map",
    "bar chart", "line chart", "pie chart", "real-time data", "live data"]
NEGATIVE = ["definition of", "definition", "what is the meaning", "the meaning of", "look up the meaning",
    "read the article", "wikipedia", "learn about", "summary of", "biography", "the history of",
    "encyclopedia", "dictionary", "translate", "who was", "when was", "where is located", "capital of",
    "how do you spell", "synonyms of", "pronunciation"]
CONSTRAINT = r"\b(and|with|under|over|below|above|between|near|featuring|priced|rated|within|" \
             r"that (?:has|have|is|are|include|offer)|less than|more than|at least|no more than)\b"
NAV_STEP = r"\b(navigate|go to|open the (?:site|page|url)|visit|homepage|load the (?:page|site))\b"
ACT_STEP = r"\b(click|select|filter|sort|scroll|toggle|choose|apply|expand|collapse|hover|drag|" \
           r"set the|enter|type|check the|uncheck|switch|zoom|slide|pick|adjust)\b"

def _rx(words):
    return re.compile(r"(?<![a-z])(?:" + "|".join(re.escape(w) for w in words) + r")", re.I)
RE_FILTER, RE_INTERACT, RE_DYNAMIC, RE_VISUAL, RE_NEG = (_rx(x) for x in (FILTER, INTERACT, DYNAMIC, VISUAL, NEGATIVE))
RE_CONSTRAINT, RE_NAV, RE_ACT = (re.compile(p, re.I) for p in (CONSTRAINT, NAV_STEP, ACT_STEP))
RE_CREATE = re.compile(r"\b(create|generate|draw|build|make a|design|customi[sz]e|construct|produce a|render a)\b", re.I)
RE_READ = re.compile(r"\b(find|identify|determine|what (?:is|are)|how (?:many|much)|which|read|look up|the price|the value|the result|the status|compare|list all|state the|the name of|the number of|check (?:if|whether)|verify)\b", re.I)

CATEGORIES = {  # coarse topic tag (best-effort; not used for scoring)
    "shopping": ["buy", "cart", "product", "price", "shop", "store", "order", "checkout", "deal", "sale"],
    "travel": ["flight", "hotel", "book", "trip", "train", "ticket", "airport", "reservation", "itinerary"],
    "finance": ["stock", "price", "invest", "market", "crypto", "bank", "loan", "portfolio", "chart"],
    "food": ["recipe", "restaurant", "menu", "food", "dish", "cuisine", "order food"],
    "media": ["video", "movie", "music", "song", "album", "watch", "stream", "episode", "playlist"],
    "sports": ["score", "game", "team", "match", "player", "standings", "schedule", "league"],
    "jobs": ["job", "career", "hiring", "salary", "resume", "apply for", "position", "employer"],
    "realestate": ["property", "rent", "apartment", "house", "listing", "real estate", "bedroom"],
    "maps": ["map", "route", "directions", "location", "nearby", "distance", "navigation"],
    "weather": ["weather", "forecast", "temperature", "rain", "climate", "radar"],
    "reference": ["definition", "wiki", "article", "encyclopedia", "dictionary", "learn"],
}

def parse_list(s):
    if s is None: return []
    if hasattr(s, "tolist"):                 # numpy array (parquet list column)
        s = s.tolist()
    if isinstance(s, (list, tuple)):
        return [str(x) for x in s]
    s = str(s)
    try:
        v = ast.literal_eval(s)
        return [str(x) for x in v] if isinstance(v, (list, tuple)) else [s]
    except Exception:
        return [s]

def score(instr, steps, criteria):
    instr = str(instr or "")
    steps_l = parse_list(steps); crit_l = parse_list(criteria)
    text = " ".join([instr] + steps_l + crit_l)
    f = min(len(RE_FILTER.findall(text)), 8)
    i = min(len(RE_INTERACT.findall(text)), 10)
    d = min(len(RE_DYNAMIC.findall(text)), 5)
    v = min(len(RE_VISUAL.findall(text)), 6)
    neg = len(RE_NEG.findall(text))
    cc = min(len(RE_CONSTRAINT.findall(instr)), 6)                      # multi-condition (case A)
    nav = sum(1 for st in steps_l if RE_NAV.search(st))
    act = sum(1 for st in steps_l if RE_ACT.search(st))
    ratio = act / (nav + act + 1e-6)                                   # interaction-heaviness (case C)
    # A FACT task READS an existing value; a pure GENERATOR/TOOL only CREATES output (no fact to harvest).
    is_create = bool(RE_CREATE.search(instr))
    is_read = bool(RE_READ.search(instr))
    tool_pen = 8.0 if (is_create and not is_read) else 0.0
    s = (2.4 * f + 1.6 * i + 1.9 * d + 2.6 * v + 1.2 * cc
         + 3.0 * ratio * (1 if act >= 2 else 0) - 1.6 * min(neg, 4) - tool_pen)
    sig = dict(filter=f, interact=i, dynamic=d, visual=v, constraints=cc, neg=neg,
               act_steps=act, nav_steps=nav, act_ratio=round(ratio, 2), fact=int(is_read), tool=int(tool_pen > 0))
    return round(s, 2), sig

def norm_domain(website):
    w = str(website or "").strip().lower()
    if not w: return None, None
    url = w if w.startswith("http") else "https://" + w
    net = urllib.parse.urlparse(url).netloc or w
    dom = net[4:] if net.startswith("www.") else net
    return dom, ("https://" + net + "/") if not w.startswith("http") else w

def categorize(text):
    t = text.lower(); best, bn = "general", 0
    for cat, kws in CATEGORIES.items():
        n = sum(t.count(k) for k in kws)
        if n > bn: best, bn = cat, n
    return best

# ---------------------------------------------------------------- optional LLM judge
def llm_judge(base, model, key, instr, steps, sig):
    sysm = ("You judge whether a website is likely RICH in OPERATION-EXCLUSIVE facts — information that "
        "can ONLY be obtained by interacting with the live page (clicking, filtering, sorting, scrolling, "
        "reading rendered charts/maps/animations), and CANNOT be obtained by a web search or by reading the "
        "page text via a reader. Score 0-5 on these signals: (A) many filter/facet conditions; (B) result "
        "state the URL can't reproduce (dynamic/in-place/AJAX/SPA); (C) the answer is behind interaction so no "
        "search/visit query can fetch it; (D) heavy visual / real-time-rendered content (charts, curves, maps, "
        "animations, live data). A pure static info-lookup scores 0-1. Output STRICT JSON.")
    usr = (f"Website task instruction: {instr}\nSteps: {steps}\n\n"
        'JSON: {"o_score": <0-5 float>, "why": "<one short phrase naming the strongest signal(s)>"}')
    body = {"model": model, "temperature": 0.2, "max_tokens": 120,
            "messages": [{"role": "system", "content": sysm}, {"role": "user", "content": usr}]}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        txt = json.load(urllib.request.urlopen(req, timeout=60))["choices"][0]["message"]["content"] or ""
        m = re.search(r"\{.*\}", txt, re.S)
        o = json.loads(m.group(0)) if m else {}
        return float(o.get("o_score", 0)), str(o.get("why", ""))
    except Exception:
        return None, ""

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True,
                    help="InSTA-150k-v3 parquet (huggingface.co/datasets/data-for-agents/insta-150k-v3)")
    ap.add_argument("--out", default="collector/data/insta_screened")
    ap.add_argument("--top-n", type=int, default=2000)
    ap.add_argument("--min-score", type=float, default=6.0, help="drop candidates below this heuristic score")
    ap.add_argument("--llm-judge", action="store_true", help="re-score the top (top-n * judge-mult) with an LLM")
    ap.add_argument("--judge-mult", type=float, default=2.5, help="judge this many x top-n heuristic leaders")
    ap.add_argument("--judge-workers", type=int, default=24)
    # 第二次从 InSTA 扩池时必须排除已用域名 —— 否则启发式排序会原样再选出上一轮
    # 那 2000 个(它们本来就是分数最高的),等于什么都没加。
    ap.add_argument("--exclude-domains", default="",
                    help="JSON 文件:已用域名清单(list 或 dict 的键),这些域名不参与候选")
    args = ap.parse_args()

    exclude = set()
    if args.exclude_domains:
        _e = json.load(open(args.exclude_domains, encoding="utf-8"))
        exclude = set(_e if isinstance(_e, list) else _e.keys())
        print(f"[exclude] 排除已用域名 {len(exclude)}", flush=True)

    import pandas as pd
    df = pd.read_parquet(args.parquet)
    print(f"[load] {len(df)} rows | cols={list(df.columns)}", flush=True)

    scored = []
    seen = set()
    for r in df.itertuples(index=False):
        website = getattr(r, "website", None)
        dom, entry = norm_domain(website)
        if not dom or dom in seen:      # dedup by domain (keep first / highest later)
            pass
        instr = getattr(r, "instruction", "") or ""
        s, sig = score(instr, getattr(r, "steps", ""), getattr(r, "criteria", ""))
        scored.append({"domain": dom, "entry_url": entry, "web_name": str(website),
                       "instruction": instr, "o_score": s, "signals": sig})
    # dedup by domain keeping the highest-scoring task per domain
    best = {}
    for c in scored:
        d = c["domain"]
        if d and (d not in best or c["o_score"] > best[d]["o_score"]):
            best[d] = c
    cand = sorted(best.values(), key=lambda c: c["o_score"], reverse=True)
    if exclude:
        _b = len(cand)
        cand = [c for c in cand if c["domain"] not in exclude]
        print(f"[exclude] 候选 {_b} -> {len(cand)}(剔已用 {_b-len(cand)})", flush=True)
    cand = [c for c in cand if c["o_score"] >= args.min_score]
    print(f"[heuristic] domains={len(best)} | >= min_score({args.min_score})={len(cand)}", flush=True)

    if args.llm_judge:
        from concurrent.futures import ThreadPoolExecutor
        base = os.environ.get("SYNTH_BASE", "http://localhost:8000/v1")
        model = os.environ.get("SYNTH_MODEL", "kimi-k3"); key = os.environ.get("SYNTH_API_KEY", "EMPTY")
        pool = cand[: int(args.top_n * args.judge_mult)]
        print(f"[llm-judge] scoring top {len(pool)} with {model} …", flush=True)
        def one(c):
            js, why = llm_judge(base, model, key, c["instruction"], c["signals"], c["signals"])
            c["llm_o_score"], c["llm_why"] = js, why
            return c
        with ThreadPoolExecutor(args.judge_workers) as ex:
            done = list(ex.map(one, pool))
        # final rank: LLM score primary (fall back to heuristic), tie-break heuristic
        for c in done:
            c["rank_score"] = (c.get("llm_o_score") if c.get("llm_o_score") is not None else 0) * 2 + c["o_score"] / 10
        cand = sorted(done, key=lambda c: c["rank_score"], reverse=True)

    top = cand[: args.top_n]
    for c in top:
        c["category"] = categorize(c["instruction"])
        c["source"] = "insta"

    # ---- write pool (pipeline_driver shape) + detail ----
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pool_items = [{"domain": c["domain"], "entry_url": c["entry_url"], "category": c["category"],
                   "source": "insta", "web_name": c["web_name"], "o_score": c["o_score"],
                   "insta_task": str(c.get("instruction") or "")[:400],   # goal-seeding for the explorer
                   **({"llm_o_score": c["llm_o_score"]} if "llm_o_score" in c else {})} for c in top]
    json.dump({"all_usable": pool_items, "total": len(pool_items)},
              open(args.out + ".pool.json", "w"), ensure_ascii=False, indent=1)
    with open(args.out + ".detail.jsonl", "w") as fh:
        for c in top:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    # ---- summary ----
    print(f"\n[done] wrote {len(top)} sites -> {args.out}.pool.json", flush=True)
    print("category mix:", dict(Counter(c["category"] for c in top)))
    ss = [c["o_score"] for c in top]
    print(f"heuristic score: min={min(ss)} max={max(ss)} median={sorted(ss)[len(ss)//2]}")
    print("signal presence in selection:",
          {k: sum(1 for c in top if c["signals"][k] > 0) for k in ("filter", "interact", "dynamic", "visual")})
    print("\n=== top 12 examples ===")
    for c in top[:12]:
        tag = f"llm={c.get('llm_o_score')}" if "llm_o_score" in c else ""
        print(f"  [{c['o_score']:.1f} {tag}] {c['domain']:28s} {c['signals']}  | {c['instruction'][:70]}")

if __name__ == "__main__":
    main()

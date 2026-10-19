#!/usr/bin/env python
"""User-value filter for extracted facts.

Keeps ONLY facts a real user would care about as an ANSWER (concrete values:
prices, SKUs, specs, data/chart/map readings, named results + attributes, the
"N items match X" outcome, computed/aggregated numbers). Drops scaffolding:
filter-option enumerations, nav / menu / sidebar structure, pagination,
marketing / hero copy, UI-state narration, and near-duplicate frames (keeping
the single most complete one).

Runs one cheap glm pass per site over ALL its facts at once (so it can judge
near-duplicates and completeness globally, which the per-turn extractor cannot).

Usage:
    SYNTH_BASE=http://localhost:8000/v1 SYNTH_API_KEY=... \
      python collector/value_filter.py runs/insta_ab_v3/progresslighting.com [...]
  -> writes <site>/facts_filtered.json  and prints kept/dropped stats.
"""
import os, sys, json, urllib.request

BASE = (os.environ.get("SYNTH_BASE") or os.environ.get("PROBE_FLASH_BASE")
        or "http://localhost:8000/v1").rstrip("/")
KEY = (os.environ.get("SYNTH_API_KEY") or os.environ.get("EXPLORER_API_KEY")
       or os.environ.get("PROBE_AGENT_KEY") or "EMPTY")
MODEL = os.environ.get("VALUE_FILTER_MODEL", os.environ.get("SYNTH_MODEL", "kimi-k3"))

SYS = (
    "You are a strict curator deciding which extracted web-exploration facts a REAL USER would keep "
    "as a genuine ANSWER, versus page scaffolding to discard. The bar is: 'would a real user, having "
    "a concrete task on this site, care about THIS as an answer or a key piece of the answer?'\n"
    "KEEP (keep=true) — concrete answer content:\n"
    "  - a specific value: price, SKU, count, measurement, date, coordinate, rate, percentage, aggregated/"
    "computed number, a chart/map/legend reading\n"
    "  - a named result WITH its attributes (product name + price + finish; a listing row with its fields)\n"
    "  - a constraint-satisfied outcome ('2 products match Dimmable=Yes AND Graphite'; 'cheapest is $148')\n"
    "  - a concrete data reading off the page (a pivot cell, a region's value, a spec sheet line)\n"
    "DROP (keep=false) — scaffolding a user never asks about:\n"
    "  - filter/facet OPTION enumerations ('Color: Antique Bronze (35), Antique Copper (20), ...')\n"
    "  - navigation / menu / sidebar / category-list structure; pagination ('page 1 of 4')\n"
    "  - marketing / hero / banner copy; generic section or layout descriptions\n"
    "  - UI-state narration ('filter panel open', 'images still loading', 'sort set to Best Selling')\n"
    "  - NEAR-DUPLICATES: if several facts describe the same thing, keep ONLY the single most complete "
    "one (keep=true) and drop the rest (keep=false)\n"
    "Be strict: when a fact is mostly scaffolding with no concrete user-answer value, drop it. Output STRICT JSON only."
)

def judge(domain, facts):
    lines = "\n".join(f"{i}. {str(f)[:300]}" for i, f in enumerate(facts))
    usr = (f"Site: {domain}\nFacts (index. value):\n{lines}\n\n"
           'Return JSON: {"decisions":[{"i":<index>,"keep":<true|false>,"cat":"<one word: value|result|'
           'reading|listing | enum|nav|marketing|uistate|dup>"} ... one per fact]}')
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": 4000,
                       "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": usr}]}).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    txt = json.load(urllib.request.urlopen(req, timeout=180))["choices"][0]["message"]["content"] or ""
    s = txt[txt.find("{"): txt.rfind("}") + 1]
    dec = json.loads(s).get("decisions") or []
    keep = {d["i"]: (bool(d.get("keep")), d.get("cat", "")) for d in dec if isinstance(d, dict) and "i" in d}
    # default-keep any fact the model forgot (conservative), so we never silently lose data
    return [(keep.get(i, (True, "unjudged"))) for i in range(len(facts))]

def apply_to_kb(site_dir):
    """Pipeline hook (A->B): filter result.json's kb.facts down to user-value facts IN PLACE,
    stashing the dropped ones under kb['_facts_dropped'] for audit. Idempotent (marks done).
    Returns (n_in, n_kept). Never raises — on any failure leaves facts untouched (returns (-1,-1))."""
    try:
        rp = os.path.join(site_dir, "result.json")
        res = json.loads(open(rp).read())
        kb = res.get("kb") or {}
        facts = kb.get("facts") or {}
        if kb.get("_value_filtered") or not facts:
            return (len(facts), len(facts))
        items = list(facts.items())
        dec = judge(os.path.basename(site_dir.rstrip("/")), [v.get("value") if isinstance(v, dict) else str(v) for _, v in items])
        kept, dropped = {}, {}
        for (k, v), (keepit, cat) in zip(items, dec):
            (kept if keepit else dropped)[k] = v
        kb["facts"] = kept
        kb["_facts_dropped"] = dropped
        kb["_value_filtered"] = True
        res["kb"] = kb
        open(rp, "w").write(json.dumps(res, ensure_ascii=False))
        return (len(items), len(kept))
    except Exception:
        return (-1, -1)

def load_facts(site_dir):
    """Prefer fact_tags.json (tagged) else result.json kb. Returns (facts_list, container, mode)."""
    ftp = os.path.join(site_dir, "fact_tags.json")
    if os.path.exists(ftp):
        obj = json.load(open(ftp)); facts = obj.get("facts") or []
        return facts, obj, "fact_tags"
    rp = os.path.join(site_dir, "result.json")
    kb = (json.load(open(rp)).get("kb") or {}).get("facts") or {}
    facts = [{"key": k, **(v if isinstance(v, dict) else {"value": v})} for k, v in kb.items()]
    return facts, None, "kb"

def main():
    for site_dir in sys.argv[1:]:
        site_dir = site_dir.rstrip("/")
        dom = os.path.basename(site_dir)
        try:
            facts, obj, mode = load_facts(site_dir)
        except Exception as e:
            print(f"{dom}: load fail {e}"); continue
        vals = [f.get("value") if isinstance(f, dict) else str(f) for f in facts]
        if not vals:
            print(f"{dom}: 0 facts"); continue
        dec = judge(dom, vals)
        kept = [f for f, (k, c) in zip(facts, dec) if k]
        out = {"domain": dom, "n_in": len(facts), "n_kept": len(kept),
               "facts": kept,
               "dropped": [{"value": (f.get("value") if isinstance(f, dict) else str(f))[:120], "cat": c}
                           for f, (k, c) in zip(facts, dec) if not k]}
        json.dump(out, open(os.path.join(site_dir, "facts_filtered.json"), "w"), ensure_ascii=False, indent=1)
        from collections import Counter
        cats = Counter(c for (k, c) in dec if not k)
        print(f"{dom}: {len(facts)} -> {len(kept)} kept ({100*len(kept)//max(len(facts),1)}%) | dropped by: {dict(cats)}")

if __name__ == "__main__":
    main()

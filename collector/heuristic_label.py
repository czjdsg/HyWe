#!/usr/bin/env python
"""Heuristic per-fact TYPE labeling over the final pool — no re-probe needed.

For every kept fact (facts_refiltered.json), glm reads the fact value + the
judge's original O-reason + the search answer/outcome that are ALREADY on disk,
and assigns a type label. This does NOT change the O/S tag; it adds an
annotation layer so the pool can be filtered by fact quality/type.

Labels:
  genuine_O      real structural barrier — only obtainable by operating the site
  agent_limited_O reachable in principle by a strong scraper/API, but the probe
                  agent structurally couldn't (deep filter, pagination, parse row);
                  STABLE across re-probes
  transient_O    probe failed for a one-off reason (HTTP 4xx/5xx, timeout, anti-bot,
                  DNS/SSL, redirect/migration, site retired) — would likely flip on retry
  volatile       inherently time-varying value (live stock/count, sale price, countdown,
                  ranking/position, "N results", "N days ago", view counts)
  reachable      (mostly S) plainly reachable static content the search got right
  bad_fact       content internally inconsistent / corrupted / attributes mismatched to its id

Non-destructive + resumable: writes <site>/facts_labeled.json, skips done sites.
Usage: python collector/heuristic_label.py --all [--conc 10]
       python collector/heuristic_label.py --calib site1 site2   # print, write nothing
"""
import os, sys, json, time, glob, urllib.request, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

# Stage-1 run directory (one sub-directory per site, written by stage1_probe.py).
RUN_DIR = os.environ.get("STAGE1_RUN_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "stage1"))
GLM = (os.environ.get("SYNTH_BASE") or "http://localhost:8000/v1").rstrip("/") + "/chat/completions"
KEY = os.environ.get("SYNTH_API_KEY") or "EMPTY"
MODEL = os.environ.get("VALUE_FILTER_MODEL", "kimi-k3")

SYS = (
    "You annotate extracted web facts with a TYPE label. Each fact already carries an O_EXCLUSIVE or "
    "S_REACHABLE tag from a prior probe (a search agent tried to reproduce it via web search; if it "
    "succeeded -> S_REACHABLE, else -> O_EXCLUSIVE). You are given the fact, its current tag, the judge's "
    "reason, and whether the search found it. DO NOT change the O/S tag — assign a TYPE label describing "
    "WHY this fact is (un)reachable, judged on the evidence:\n"
    "- \"genuine_O\": a REAL structural barrier — the value sits behind interactive form/configurator/"
    "date-picker/filter submission, is dynamically JS-rendered content a crawler won't get, is a precise "
    "chart/map/tooltip reading at an exact point, or is otherwise never exposed to a search/fetch. Only "
    "obtainable by operating the site.\n"
    "- \"agent_limited_O\": reachable IN PRINCIPLE by a strong scraper / public API / direct page fetch, but "
    "the probe agent failed for a CAPABILITY reason — it couldn't apply a deep filter, paginate to a later "
    "page, parse a specific row/rank, or render an SPA. This would fail on EVERY re-probe (stable), it is "
    "not a one-off error.\n"
    "- \"transient_O\": the probe failed for a ONE-OFF reason — HTTP 403/404/500/503, timeout, anti-bot / "
    "bot-detection block, DNS/SSL error, redirect/domain-migration, or the site being retired/archived. A "
    "retry would likely succeed.\n"
    "- \"volatile\": querying the SAME thing later would return a DIFFERENT value — live stock/inventory, a "
    "current sale price/discount, a countdown, a live ranking/listing position, a live result/job/item "
    "count, view counts, or a relative date ('N days ago'). The fact is not durable. IMPORTANT: a reading "
    "PINNED to a fixed past date/timestamp (e.g. 'grid output on 05/21/2025 at 10:45', 'closing price on "
    "Jan 6') is NOT volatile — that value is durable/historical; label it by its reachability instead "
    "(genuine_O / agent_limited_O / transient_O / reachable).\n"
    "- \"reachable\": (typically S_REACHABLE) plainly reachable static content that the search agent got "
    "right; no barrier.\n"
    "- \"bad_fact\": the fact's own content looks wrong / internally inconsistent — attributes that don't "
    "match its own identifier, values apparently copied from a neighboring record, or garbled text.\n"
    "Prefer volatile/bad_fact when they clearly apply (they override the O sub-types). For an S_REACHABLE "
    "fact use 'reachable' unless it is volatile or a bad_fact. Output STRICT JSON only."
)

def label_site(domain, facts, *, retries=4, timeout=180):
    lines = []
    for i, f in enumerate(facts):
        tag = "O_EXCLUSIVE" if str(f.get("tag", "")).startswith("O") else "S_REACHABLE"
        lines.append(f"{i}. [{tag}] value: {str(f.get('value',''))[:230]}\n"
                     f"   found={f.get('s_found')} reason: {str(f.get('reason',''))[:230]}")
    usr = (f"Site: {domain}\nFacts:\n" + "\n".join(lines) +
           '\n\nReturn JSON: {"labels":[{"i":<index>,"label":"genuine_O|agent_limited_O|transient_O|'
           'volatile|reachable|bad_fact","why":"<=10 words"} ... one per fact]}')
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": 4000,
                       "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": usr}]}).encode()
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(GLM, data=body,
                                         headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            txt = json.load(urllib.request.urlopen(req, timeout=timeout))["choices"][0]["message"]["content"] or ""
            s = txt[txt.find("{"): txt.rfind("}") + 1]
            labs = json.loads(s).get("labels") or []
            m = {d["i"]: (d.get("label", "?"), d.get("why", "")) for d in labs if isinstance(d, dict) and "i" in d}
            return [m.get(i, ("unlabeled", "")) for i in range(len(facts))], None
        except Exception as e:
            last = e; time.sleep(2 * (a + 1))
    return None, f"{type(last).__name__}: {last}"

def process(site_dir, write=True):
    dom = os.path.basename(site_dir.rstrip("/"))
    src = os.path.join(site_dir, "facts_refiltered.json")
    outp = os.path.join(site_dir, "facts_labeled.json")
    if write and os.path.exists(outp):
        try:
            o = json.load(open(outp)); return dom, Counter(x.get("hlabel", "?") for x in o.get("facts", [])), "skip"
        except Exception:
            pass
    try:
        obj = json.load(open(src)); facts = obj.get("facts") or []
    except Exception as e:
        return dom, Counter(), f"loadfail:{e}"
    if not facts:
        return dom, Counter(), "empty"
    labs, err = label_site(dom, facts)
    if labs is None:
        return dom, Counter(), f"glmfail:{err}"
    for f, (lab, why) in zip(facts, labs):
        f["hlabel"] = lab; f["hlabel_why"] = why
    if write:
        json.dump({"domain": dom, "n": len(facts), "facts": facts}, open(outp, "w"), ensure_ascii=False, indent=1)
    return dom, Counter(l for l, _ in labs), "ok"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="", help="Stage-1 run dir (default: $STAGE1_RUN_DIR or runs/stage1)")
    ap.add_argument("--calib", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--conc", type=int, default=10)
    a = ap.parse_args()
    if a.run_dir:
        global RUN_DIR
        RUN_DIR = a.run_dir

    if a.calib:
        for name in a.calib:
            sd = os.path.join(RUN_DIR, name)
            obj = json.load(open(os.path.join(sd, "facts_refiltered.json"))); facts = obj.get("facts") or []
            labs, err = label_site(name, facts)
            print(f"\n== {name} ({len(facts)}) err={err} ==")
            if labs:
                for f, (lab, why) in zip(facts, labs):
                    t = "O" if str(f.get("tag", "")).startswith("O") else "S"
                    print(f"  [{t}] {lab:16s} {str(f.get('value'))[:80]}  <- {why}")
        return

    sites = [os.path.dirname(s) for s in sorted(glob.glob(os.path.join(RUN_DIR, "*/facts_refiltered.json")))]
    print(f"labeling {len(sites)} sites, conc={a.conc}", flush=True)
    total = Counter(); fails = []; done = 0
    with ThreadPoolExecutor(max_workers=a.conc) as ex:
        futs = {ex.submit(process, sd, True): sd for sd in sites}
        for fut in as_completed(futs):
            dom, cnt, status = fut.result(); done += 1
            if status.startswith(("glmfail", "loadfail")):
                fails.append((dom, status))
            else:
                total += cnt
            if done % 50 == 0:
                print(f"  [{done}/{len(sites)}] {dict(total)} fails={len(fails)}", flush=True)
    print(f"\nDONE {done} sites | label distribution:")
    tot = sum(total.values())
    for k, c in total.most_common():
        print(f"  {k}: {c}  ({100*c//max(tot,1)}%)")
    if fails:
        print(f"FAILED {len(fails)} (rerun to retry): {fails[:15]}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Second-pass, STRICTER user-value re-filter over the final tagged pool.

Runs one glm-5.2 pass PER SITE over all its tagged facts, emitting a per-fact
keep/drop decision (逐条). Sharper than value_filter.py: it also kills the
residual noise seen in the pool — ordinal/positional UI enumeration
("Fifth and sixth rows of videos", "step 3 of 4", "the 2nd listing is..."),
pure layout/position description, and process narration — while preserving each
kept fact's O/S tag. Non-destructive: writes <site>/facts_refiltered.json and
leaves fact_tags.json untouched. Idempotent + resumable (skips done sites).

Usage:
  python collector/refilter_pool.py --calib site1 site2 ...   # print decisions, write nothing
  python collector/refilter_pool.py --all [--conc 8]          # full pool
"""
import os, sys, json, time, glob, urllib.request, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

# Stage-1 run directory (one sub-directory per site, written by stage1_probe.py).
RUN_DIR = os.environ.get("STAGE1_RUN_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "stage1"))
BASE = (os.environ.get("SYNTH_BASE") or "http://localhost:8000/v1").rstrip("/")
KEY = os.environ.get("SYNTH_API_KEY") or "EMPTY"
MODEL = os.environ.get("VALUE_FILTER_MODEL", "kimi-k3")

SYS = (
    "You are a STRICT curator deciding which extracted web-exploration facts a REAL USER would keep as a "
    "genuine ANSWER to a concrete task on this site, versus page scaffolding / UI narration to discard. "
    "The bar: 'would a real user, having a concrete task on this site, care about THIS as the answer or a "
    "key piece of the answer?'\n\n"
    "KEEP (keep=true) — concrete answer content:\n"
    "  - a specific value: price, SKU/model, count, measurement/reading, date, coordinate, rate, %, "
    "aggregated/computed number, a chart/map/legend/table-cell reading\n"
    "  - a named result WITH attributes (product name + price + spec; a listing row with its real fields: "
    "flight number+times, yacht name+length+price, park name+address+amenities)\n"
    "  - a constraint-satisfied outcome ('4 Disney host agencies in California', 'cheapest is $148', "
    "'1,108 jobs match default filters')\n"
    "  - a durable reference datum a user would look up (an address, an official contact email, an opening-"
    "hours table, a named policy with its concrete terms)\n\n"
    "DROP (keep=false) — scaffolding / narration a user never asks about:\n"
    "  - ORDINAL / POSITIONAL UI ENUMERATION: any fact whose substance is *where things sit on screen* rather "
    "than their value — 'Fifth and sixth rows of videos are...', 'the 3rd listing is...', 'row 5 shows...', "
    "'the second card...'. If stripping the position leaves no concrete user-answer value, DROP it.\n"
    "  - PROCESS / STEP NARRATION: 'the upload workflow has 4 steps: 1) select photo 2)...', 'after clicking X "
    "the panel opened', 'sort set to Best Selling', 'images still loading', 'successfully navigated to...'\n"
    "  - filter/facet OPTION enumerations ('Color: Antique Bronze (35), Antique Copper (20)...'); "
    "navigation / menu / sidebar / category-list structure; pagination ('page 1 of 4')\n"
    "  - marketing / hero / banner copy; generic section or layout description ('the page has a header and a "
    "grid of tiles')\n"
    "  - NEAR-DUPLICATES: if several facts describe the same thing, keep ONLY the single most complete one "
    "(keep=true) and drop the rest (keep=false). Distinct values (two different prices/rows of DATA) are NOT dups.\n\n"
    "Be strict: when a fact is mostly scaffolding, position description, or process narration with no concrete "
    "user-answer value, DROP it. Keep genuine data even if it names a position, AS LONG AS a concrete value "
    "rides along ('405 North at Getty Center Dr: 71 MPH' KEEP; 'the fifth row of videos' DROP). "
    "Output STRICT JSON only."
)

def judge(domain, vals, *, retries=4, timeout=180):
    lines = "\n".join(f"{i}. {str(v)[:300]}" for i, v in enumerate(vals))
    usr = (f"Site: {domain}\nFacts (index. value):\n{lines}\n\n"
           'Return JSON: {"decisions":[{"i":<index>,"keep":<true|false>,"cat":"<one word: value|result|'
           'reading|listing|reference | position|process|enum|nav|marketing|dup>"} ... one per fact]}')
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": 4000,
                       "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": usr}]}).encode()
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                         headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            txt = json.load(urllib.request.urlopen(req, timeout=timeout))["choices"][0]["message"]["content"] or ""
            s = txt[txt.find("{"): txt.rfind("}") + 1]
            dec = json.loads(s).get("decisions") or []
            keep = {d["i"]: (bool(d.get("keep")), d.get("cat", "")) for d in dec if isinstance(d, dict) and "i" in d}
            # default-keep any fact the model forgot (conservative — never silently lose data)
            return [keep.get(i, (True, "unjudged")) for i in range(len(vals))], None
        except Exception as e:
            last = e
            time.sleep(2 * (a + 1))
    return None, f"{type(last).__name__}: {last}"

def load_tagged(site_dir):
    obj = json.load(open(os.path.join(site_dir, "fact_tags.json")))
    # Keep failed probes in fact_tags.json for inspection, but never promote
    # them to training capability labels (including legacy default-O errors).
    return [f for f in obj.get("facts") or []
            if f.get("probe") != "error" and f.get("tag") in ("S_REACHABLE", "O_EXCLUSIVE")]

def process(site_dir, write=True):
    dom = os.path.basename(site_dir.rstrip("/"))
    outp = os.path.join(site_dir, "facts_refiltered.json")
    if write and os.path.exists(outp):
        try:
            o = json.load(open(outp)); return dom, o.get("n_in", 0), o.get("n_kept", 0), Counter(), "skip"
        except Exception:
            pass
    try:
        facts = load_tagged(site_dir)
    except Exception as e:
        return dom, 0, 0, Counter(), f"loadfail:{e}"
    vals = [f.get("value") if isinstance(f, dict) else str(f) for f in facts]
    if not vals:
        return dom, 0, 0, Counter(), "empty"
    dec, err = judge(dom, vals)
    if dec is None:
        return dom, len(vals), -1, Counter(), f"glmfail:{err}"
    kept, dropped = [], []
    for f, (k, c) in zip(facts, dec):
        (kept if k else dropped).append((f, c))
    if write:
        out = {"domain": dom, "n_in": len(facts), "n_kept": len(kept),
               "facts": [f for f, _ in kept],   # full fact dict incl. O/S tag preserved
               "dropped": [{"tag": (f.get("tag") if isinstance(f, dict) else "?"),
                            "value": (f.get("value") if isinstance(f, dict) else str(f))[:140], "cat": c}
                           for f, c in dropped]}
        json.dump(out, open(outp, "w"), ensure_ascii=False, indent=1)
    return dom, len(facts), len(kept), Counter(c for _, c in dropped), ("calib" if not write else "ok")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="", help="Stage-1 run dir (default: $STAGE1_RUN_DIR or runs/stage1)")
    ap.add_argument("--calib", nargs="*", help="site names under the run dir; print decisions, write nothing")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--conc", type=int, default=8)
    a = ap.parse_args()
    if a.run_dir:
        global RUN_DIR
        RUN_DIR = a.run_dir

    if a.calib:
        for name in a.calib:
            sd = os.path.join(RUN_DIR, name)
            facts = load_tagged(sd)
            vals = [f.get("value") if isinstance(f, dict) else str(f) for f in facts]
            dec, err = judge(name, vals)
            print(f"\n===== {name}  ({len(vals)} facts)  err={err} =====")
            if dec is None: continue
            for f, (k, c) in zip(facts, dec):
                tag = "O" if str(f.get("tag", "")).startswith("O") else ("S" if str(f.get("tag", "")).startswith("S") else "?")
                mark = "KEEP" if k else f"DROP({c})"
                print(f"  [{tag}] {mark:11s} {str(f.get('value'))[:120]}")
        return

    sites = sorted(glob.glob(os.path.join(RUN_DIR, "*/fact_tags.json")))
    sites = [os.path.dirname(s) for s in sites]
    print(f"re-filtering {len(sites)} sites, conc={a.conc}")
    tot_in = tot_kept = 0; drop_cats = Counter(); fails = []; done = 0
    with ThreadPoolExecutor(max_workers=a.conc) as ex:
        futs = {ex.submit(process, sd, True): sd for sd in sites}
        for fut in as_completed(futs):
            dom, nin, nkept, cats, status = fut.result()
            done += 1
            if status.startswith(("glmfail", "loadfail")):
                fails.append((dom, status))
            elif nkept >= 0:
                tot_in += nin; tot_kept += nkept; drop_cats += cats
            if done % 50 == 0:
                print(f"  [{done}/{len(sites)}] kept {tot_kept}/{tot_in}  fails={len(fails)}")
    print(f"\nDONE {done} sites | facts {tot_in} -> {tot_kept} kept ({100*tot_kept//max(tot_in,1)}%)")
    print(f"dropped by cat: {dict(drop_cats.most_common())}")
    if fails:
        print(f"FAILED {len(fails)} sites (rerun to retry): {fails[:15]}")

if __name__ == "__main__":
    main()

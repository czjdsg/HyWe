#!/usr/bin/env python3
"""v7 Stage C: synthesize one hybrid task per site from the TAGGED fact table.

Input  (per site): result.json (Stage A KB) + fact_tags.json (Stage B tags)
Output (per site): task.json  — plus an aggregate synth7_tasks.jsonl in the
                   rollout-watcher input row format.

Key property vs the old RECORD stage: the S/O division of labor is now
EMPIRICAL. Each fact carries a measured tag:
  O_EXCLUSIVE  - a search agent could NOT retrieve it (browser required)
  S_REACHABLE  - a search agent retrieved it in <=5 steps
  S_HARVESTED  - external fact surfaced DURING the probe, anchored to the
                 O fact it was probing (verified findable by construction)

The synthesis model (Claude) must build a fact CHAIN that crosses the
S/O boundary; the route_plan is DERIVED from the tags, not narrated.
Validation is code-side and rejects:
  - phases whose cited facts don't match the phase mode
  - chains lacking >=1 O_EXCLUSIVE or >=1 S-side fact
  - questions that don't embed >=2 concrete anchors from cited fact values

Usage:
  python3 task_synth.py --run /tmp/synth7_poc [--conc 8] [--retries 1]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import time
import urllib.request
from pathlib import Path

SYNTH_BASE = os.environ.get("SYNTH_BASE", "http://localhost:8000/v1")
SYNTH_MODEL = os.environ.get("SYNTH_MODEL", "qwen3.7-max")
API_KEY = os.environ.get("SYNTH_API_KEY", "EMPTY")


SYNTH_MAX_RETRIES = int(os.environ.get("SYNTH_MAX_RETRIES", "12"))


def call_llm(messages, *, max_tokens=4000, temperature=0.3,
             timeout=900) -> str:
    """Retry transient transport / gateway errors with exponential backoff.
    Caps at SYNTH_MAX_RETRIES attempts (default 12) so a single site cannot
    wedge the pipeline when the upstream goes into a sustained rate-limit /
    quota-exhausted state (e.g. HTTP 418 from the synth gateway). Backoff
    caps at 60s, so worst-case wait per site ≈ 12 minutes before raising."""
    payload = json.dumps({"model": SYNTH_MODEL, "messages": messages,
                          "max_tokens": max_tokens,
                          "temperature": temperature}).encode()
    attempt = 0
    last_exc = None
    while attempt < SYNTH_MAX_RETRIES:
        req = urllib.request.Request(SYNTH_BASE.rstrip("/") + "/chat/completions",
                                     data=payload, method="POST")
        req.add_header("Authorization", f"Bearer {API_KEY}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode("utf-8", errors="replace"))
                return d["choices"][0]["message"]["content"] or ""
        except Exception as exc:
            last_exc = exc
            attempt += 1
            wait = min(2 ** min(attempt, 6) * 2, 60)
            print(f"  [llm-retry #{attempt}/{SYNTH_MAX_RETRIES}] "
                  f"{type(exc).__name__}: {str(exc)[:120]} — retrying in {wait}s",
                  flush=True)
            time.sleep(wait)
    raise RuntimeError(
        f"call_llm exhausted {SYNTH_MAX_RETRIES} retries; last error: "
        f"{type(last_exc).__name__}: {str(last_exc)[:200]}")


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


# ---------------- fact table assembly ----------------
def load_fact_table(site_dir: Path) -> tuple[dict, dict]:
    """Returns (facts_by_key, site_info). Harvested S facts get key
    `h__<anchor>__<key>` with tag S_HARVESTED."""
    res = json.loads((site_dir / "result.json").read_text())
    tags = json.loads((site_dir / "fact_tags.json").read_text())
    site = res.get("site") or {}
    table: dict[str, dict] = {}
    for row in tags.get("facts") or []:
        tag = row.get("tag")
        probe = row.get("probe") or ""
        if probe.startswith("skip"):
            continue  # chrome/banned — not available for tasks
        if tag not in ("S_REACHABLE", "O_EXCLUSIVE"):
            continue
        table[row["key"]] = {
            "value": row["value"],
            "tag": tag,
            "side": "S" if tag == "S_REACHABLE" else "O",
            "reason": row.get("reason") or "",
            "question": row.get("question") or "",
            "s_answer": row.get("s_answer") or "",
        }
        for h in row.get("harvested") or []:
            hkey = f"h__{row['key'][:24]}__{h['key']}"[:80]
            table[hkey] = {
                "value": h["value"],
                "tag": "S_HARVESTED",
                "side": "S",
                "anchor": row["key"],
                "source": h.get("source") or "",
            }
    return table, site


def render_table(table: dict) -> str:
    lines = []
    for k, v in table.items():
        extra = ""
        if v["tag"] == "S_HARVESTED":
            extra = f"  [anchored to: {v.get('anchor')}; source: {v.get('source')}]"
        elif v["tag"] == "S_REACHABLE":
            extra = f"  [search found: {v.get('s_answer','')[:120]}]"
        lines.append(f"- {k}  [{v['tag']}]: {v['value']}{extra}")
    return "\n".join(lines)


# ---------------- synthesis prompt ----------------
SYNTH_PROMPT = """You are designing UP TO {n_tasks} DISTINCT realistic hybrid web tasks
for the site below. Use as many of the available facts as you can across
the set — each task should be different in scope/intent, not paraphrases.

Every fact in the table carries an EMPIRICAL tag from a live experiment:
  O_EXCLUSIVE  - a search agent could NOT find this; it only exists in the
                 site's interactive state. Reaching it REQUIRES driving the
                 browser on the site (O-side work).
  S_REACHABLE  - a search agent DID find this in <=5 search/visit calls.
                 An efficient agent should get it via search (S-side work).
  S_HARVESTED  - external fact found while probing the anchored O fact
                 (already verified searchable; naturally related to it).

Site: {domain}   entry_url: {entry_url}   category: {category}

FACT TABLE:
{table}

DIFFICULTY TARGETS — based on TOTAL WORK = N_facts + N_switches.
A SWITCH (mode hand-off) counts as one unit of work because the runtime
must recognize the wrong mode and emit a `switch` action. SHAPE-1 always
contributes one leading switch even if only one fact is collected.

  easy   - score 2-3   (e.g. 1 switch + 1 fact, or 2-3 facts no switch)
  medium - score 4-5   (e.g. 1 switch + 3 facts, or 2 switches + 2-3 facts)
  hard   - score 6+    (e.g. 2 switches + 4+ facts, multi-constraint chains)

Compute score as: len(cited_fact_keys) + count_of_switches_in_the_plan.
For SHAPE-1, count_of_switches=1 (the mandatory leading switch).
For SHAPE-2 (two_side_single_switch), count_of_switches=1.
For SHAPE-3 (multi_crossing), count_of_switches=number of S<->O boundaries.

Try to produce a MIX across difficulties when the fact table supports it
(e.g. 1 easy + 1 medium + 1 hard for {n_tasks}=3). A SHAPE-1 with 3+ facts
SHOULD be medium, not easy. A SHAPE-2 with 5+ facts SHOULD be hard.

NO PADDING — difficulty must come from a GENUINE need, never from stapling
extra facts onto a question to inflate the score:
- Every cited fact must be something the ANSWER actually requires. Read
  your question back: if a fact could be deleted and the question still
  reads complete, DELETE it and accept the lower difficulty.
- BAD (padded): "How many courses are in each of three STEM categories,
  what do third-party sources report for each, and how big is the whole
  catalog?" — no real user stacks 7 number-lookups in one breath.
- GOOD (genuine): "VSNT's latest annual EPS vs the consensus estimate —
  did it beat or miss, and what's the stock at now?" — three facts, but
  one coherent investing question.
- If the table cannot support a NATURAL hard task, output easy/medium
  tasks only. A missing hard slot is fine; a Frankenstein hard is not.

THREE TASK SHAPES — design across all that the fact table supports.

The core skill we are training is the SWITCH decision — recognizing when
the current mode is wrong for the next sub-goal and handing off. We want a
MIX of easy switch signals and harder multi-crossing chains.

(SHAPE 1) MISMATCHED-START single-side task (PURE switch trainer):
  The task is genuinely answerable entirely from ONE side (only O facts
  needed, OR only S facts needed), but the agent's runtime starts on the
  OPPOSITE side. The correct first action will therefore be a `switch` —
  but YOU DO NOT WRITE THE SWITCH into the phases list. The hybrid runtime
  inserts switches automatically at every S<->O boundary it detects.

  Use this when the fact table has a high-quality O-only OR S-only chain
  that yields a clean user-intent answer (e.g. "what's the top hotel in Las
  Vegas right now" — pure O; or "who won the 2024 NBA MVP" — pure S).

  phases: [{{mode=Y, subgoal=..., fact_keys=[...the actual work...]}}]
  - phases has EXACTLY ONE entry — the working phase. NO phantom starter,
    NO "switch" subgoal, NO empty fact_keys phase.
  - Y is the side where the facts live.
  - start_mode = the OPPOSITE of Y (this tells the runtime to start the
    agent on the wrong side so it must switch immediately).
  - difficulty: easy when fact_keys=1, medium when fact_keys=2-3, hard
    when fact_keys=4+. Computed via score = N_facts + 1 (mandatory switch).

(SHAPE 2) GENUINE TWO-SIDE single-switch task:
  Solution actually needs BOTH sides — search establishes context AND the
  site provides the exclusive answer (or vice versa). Exactly ONE switch.

  Two sub-variants by starting mode:
  (2a) O→S: visit the site first to surface the O_EXCLUSIVE entity (a
       top-N listing, an in-image artist, a filtered count), then switch
       to S to enrich/verify it.
  (2b) S→O: start with a search to establish context or a candidate set,
       then switch to O to find the site-exclusive value.

  Chain MUST include >=1 O_EXCLUSIVE AND >=1 S_REACHABLE/S_HARVESTED.

(SHAPE 3) MULTI-CROSSING task (advanced):
  Two or more crossings (OSO, SOS, OSOS). Use ONLY when the solution
  genuinely requires returning to a side after leaving it — e.g. O surfaces
  an entity, S enriches it, then O must verify something NEW that depends
  on what S found. If the same facts can be gathered with one switch, use
  SHAPE-2 and let the score determine difficulty — do NOT slice a 1-switch
  solution into 3 phases just to claim multi_crossing. A high-score SHAPE-2
  (score 6+) is a perfectly good hard task.

EVERY TASK MUST CONTAIN >=1 SWITCH. There is no shape without a switch —
a single-mode no-switch task is INVALID. The minimum task is a SHAPE-1 with
1 leading switch + 1 working phase.

ONE COHERENT SCENARIO — every fact in the chain must belong to the same
user story:
- All cited facts must revolve around ONE entity, ONE comparison, or ONE
  decision the user is making. A fact about the Golden Knights does not
  belong in a question about the Canadiens; a fact about chair A does not
  belong in a question about lamp B unless the user is explicitly
  cross-shopping them for the same purpose.
- Test: state the user's underlying goal in one sentence ("deciding if
  VSNT is a buy", "picking a hotel near Bryce Canyon"). Every fact must
  serve THAT goal. Facts that fail the test → cut them.

UNTRUSTWORTHY S VALUES — handle search noise honestly:
- Some S_HARVESTED/S_REACHABLE values in the table are search-agent noise:
  wildly different from the O-side value for the same attribute (wrong
  ticker, stale snapshot, different entity with a similar name).
- Do NOT build a task whose answer celebrates such a discrepancy as a
  finding ("CNBC says 6.94 but search says -232!"). If the two sides
  disagree by an implausible margin, treat the S value as NOISE and avoid
  citing it. A genuine cross-check task needs two trustworthy values that
  plausibly refer to the same thing.

SET-LEVEL DIVERSITY:
- A task_set with >=2 tasks SHOULD include at least one SHAPE-1 (the pure
  switch trainer) when the table supports it. SHAPE-1 is high-value
  training signal — do not skip it just because the chain is "too simple".
- Across the set, vary starting mode (some O-start, some S-start).
- Difficulty mix: easy SHAPE-1 + medium SHAPE-2 + hard SHAPE-3 is ideal.

USER-INTENT TEST — every task MUST pass this:
A realistic person who has NOT yet visited the site must plausibly want this
answer. Imagine them about to open a search engine or message a research
assistant. Would they reasonably ask THIS? If the question only makes sense
to someone already on the page (asking about which tabs, filter sections,
toolbar buttons, dropdown options, UI states, or "what categories are
available"), it is REJECTED.

Good intents (information a user wants):
- "I'm planning a trip — what's the cheapest highly-rated hotel near Bryce
  Canyon, and how does it compare to a hotel in Sedona?"
- "Who's the top fantasy basketball player in 9-cat for 2026-27?"
- "What's the most popular Dribbble shot this week and who designed it?"

Bad intents (the user shouldn't have to ask these):
- "What filter sections are visible on the page?"
- "Which tabs appear in the main navigation?"
- "What sort options are available in the dropdown?"
- "What buttons are in the chart toolbar?"
- "Which input fields appear in the advanced search form?"

The facts in the table may include some UI-shaped O facts; you do NOT have
to use them. Prefer facts whose VALUE is a concrete answer (a price, a
ranking, a name, a count) over facts whose value just describes UI options.
If the only available facts are UI-shaped (e.g. one site has nothing but
filter/option/tab dumps), output {{"kind": "no_task", "reason": "..."}}.

ANTI-OVERLAP — tasks within the set MUST differ substantively:
- No two tasks may share more than half of their cited fact_keys.
- Different angles: e.g. one about pricing, another about features, another
  about comparison/validation. Not "Q1 in present tense, Q2 in past tense".

Output EXACTLY ONE JSON object:

{{"kind": "task_set",
  "tasks": [
    {{
      "question": "<what a real user would TYPE.
                   Two halves: (1) INTENT — what they want; (2) CONSTRAINTS —
                   the setup (location, dates, party size, category, etc.)
                   that scopes the answer.
                   THE QUESTION ASKS FOR THE ANSWER. It MUST NOT contain the
                   answer. Specifically, NO substring from any cited fact's
                   value may appear in the question text. The user does not
                   know what 'Hôtel Indigo', '€245', '4.2★', '8,443 courses',
                   etc. are — they are going to FIND OUT by doing the task.
                   NO steps, NO 'search for'/'navigate to', NO tool names.
                   NO meta-method phrasing: never 'does a web search agree',
                   'compare with search results', 'based on the most recent
                   snapshot' — express cross-checking as a natural concern
                   ('do third-party sources quote the same price?', 'does the
                   official site match what review sites say?').>",
      "shape": "single_side_switch" | "two_side_single_switch" | "multi_crossing",
      "start_mode": "S" | "O",
      "chain": [
        {{"fact_key": "<key from table>", "role": "<why the task needs it>"}}
      ],
      "phases": [
        {{"mode": "S"|"O",
          "subgoal": "<concrete intent of this WORKING segment. Plain
                      English description of what to do, NEVER mention
                      'switch' or 'hand off'.>",
          "done_when": "<observable exit condition>",
          "fact_keys": ["<keys this phase obtains/uses — tags MUST match mode>"]}}
      ],
      "difficulty": "easy"|"medium"|"hard",
      "answer_sketch": "<1-2 sentences: what the correct final answer looks like,
                        grounded in the fact values>"
    }}
  ]
}}

HARD RULES (validated by code — violations are rejected per-task):
1. Every fact_key must exist in the table.
2. O-phase fact_keys must all be O_EXCLUSIVE; S-phase fact_keys must all be
   S_REACHABLE or S_HARVESTED.
3. Phases must alternate modes (no two consecutive same-mode phases).
4. CROSS-SIDE FACT REQUIREMENT — depends on shape:
   - single_side_switch: chain has facts from EXACTLY ONE side (all O OR
     all S). phases has EXACTLY ONE working entry (NO phantom starter).
     start_mode MUST be the OPPOSITE side from phases[0].mode — this
     forces the runtime to insert a switch at turn 1.
   - two_side_single_switch or multi_crossing: chain MUST include >=1
     O_EXCLUSIVE AND >=1 S-side fact (S_REACHABLE or S_HARVESTED).
     start_mode MUST equal phases[0].mode (the agent starts where the
     first phase lives, no leading switch).
5. question must NOT contain ANY substring of length >=4 (case-insensitive,
   non-trivial — i.e. not "the"/"and"/year digits alone) drawn from any cited
   fact's VALUE. The question states constraints; the facts are the answer.
   Practical paraphrasing pattern — replace specific value words with their
   GENERIC DESCRIPTION + the user's actual constraint:
     LEAK:    "What's the price of the Harmonie Whisky Decanter (Clear)?"
              (leaks 'Harmonie Whisky Decanter', 'Clear' — those ARE the answer)
     CLEAN:   "I'm looking at Baccarat's clear-glass whisky decanters. What
               do they list for on the international site?"
              (says 'clear-glass whisky decanters' as a constraint, not
               the specific product line)
     LEAK:    "How much does it cost to stay at the Belmond Hotel Caruso for
               4 nights in late June?"
              (leaks 'Belmond Hotel Caruso' — picking the right hotel IS
               part of the agent's job)
     CLEAN:   "For Belmond's flagship Amalfi-Coast property, what's the
               nightly rate for a 4-night late-June stay in a suite?"
              (constraint: 'Amalfi-Coast property' / 'suite'; the specific
               hotel name is what the agent must discover)
   If you find yourself writing the answer's brand/model/specific tier in
   the question, you are TELLING the user the answer. Re-state as a CONSTRAINT
   on what they're shopping for, never the line they're shopping FROM.
6. start_mode determines the agent's runtime starting side. The runtime
   automatically inserts a `switch` action at every S<->O boundary where
   start_mode → phases[0].mode → phases[i].mode → phases[i+1].mode differ;
   YOU MUST NOT write any phase whose subgoal is "switch" or "hand off".
   A multi-task set SHOULD include both O-start and S-start variants when
   the table supports it.
7. (set-level) No two tasks share >50% of fact_keys.
8. ENTITY PROVENANCE — never write an entity full name the fact table does
   not show. If a fact key/value only carries a ticker or abbreviation
   (e.g. "vsnt_current_price"), the question may use the ticker itself
   ("VSNT") or a generic description ("the stock with ticker VSNT"), but
   MUST NOT invent an expansion ("Versant Health") — expansions you
   "remember" are often wrong for this specific site snapshot, and the
   agent will chase a non-existent entity. Every multi-word proper-noun
   phrase in the question must appear in the fact keys/values or be the
   site's own name.

If the table cannot support ANY genuine boundary-crossing task (e.g. no
O_EXCLUSIVE facts, or all O and S facts are about unrelated topics), output:
  {{"kind": "no_task", "reason": "<1 sentence>"}}

If the table only supports fewer than {n_tasks} non-overlapping tasks, output
fewer tasks rather than padding with redundant variants.

No code fences. No prose around the JSON."""


# ---------------- validation ----------------
def anchors_from_value(v: str) -> set[str]:
    out = set()
    for m in re.finditer(r"(?:HK\$|US\$|\$|€|£|¥)\s?[\d,]+(?:\.\d+)?", v):
        out.add(m.group(0).replace(" ", ""))
    for m in re.finditer(r"\b\d[\d,]*(?:\.\d+)?\b", v):
        if len(m.group(0).replace(",", "")) >= 2:
            out.add(m.group(0))
    for m in re.finditer(r"\b[A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+){0,3}\b", v):
        if len(m.group(0)) >= 4:
            out.add(m.group(0))
    return out


def _set_overlap_violations(tasks: list[dict]) -> str | None:
    """Return None if every pair shares <=50% of fact_keys, else a reason."""
    if len(tasks) <= 1:
        return None
    keysets = []
    for t in tasks:
        ks = set()
        for c in t.get("chain") or []:
            k = (c or {}).get("fact_key")
            if k:
                ks.add(k)
        for p in t.get("phases") or []:
            for k in p.get("fact_keys") or []:
                ks.add(k)
        keysets.append(ks)
    for i in range(len(keysets)):
        for j in range(i + 1, len(keysets)):
            a, b = keysets[i], keysets[j]
            if not a or not b:
                continue
            overlap = len(a & b) / max(len(a), len(b))
            if overlap > 0.5:
                return (f"tasks[{i}] and tasks[{j}] share "
                        f"{int(overlap*100)}% of fact_keys "
                        f"({sorted(a & b)[:5]}...) — overlap must be <=50%")
    return None


def validate(task: dict, table: dict, domain: str = "") -> tuple[bool, str]:
    if task.get("kind") == "no_task":
        return True, ""
    if task.get("kind") != "task":
        return False, f"kind must be 'task' or 'no_task', got {task.get('kind')!r}"
    question = (task.get("question") or "").strip()
    if len(question) < 30:
        return False, "question too short"
    chain = task.get("chain") or []
    phases = task.get("phases") or []
    if not chain or not phases:
        return False, "chain and phases are required"

    # rule 1: keys exist
    cited = []
    for c in chain:
        k = (c or {}).get("fact_key")
        if k not in table:
            return False, f"chain cites unknown fact_key {k!r}"
        cited.append(k)
    for p in phases:
        for k in (p.get("fact_keys") or []):
            if k not in table:
                return False, f"phase cites unknown fact_key {k!r}"

    # rule 2: tag/mode consistency
    for i, p in enumerate(phases):
        mode = p.get("mode")
        if mode not in ("S", "O"):
            return False, f"phases[{i}].mode must be S or O"
        if not (p.get("subgoal") or "").strip() or not (p.get("done_when") or "").strip():
            return False, f"phases[{i}] needs non-empty subgoal and done_when"
        for k in (p.get("fact_keys") or []):
            side = table[k]["side"]
            if side != mode:
                return False, (f"phases[{i}] is {mode}-mode but cites {k!r} "
                               f"which is {table[k]['tag']} ({side}-side)")

    # rule 3: alternation
    for i in range(1, len(phases)):
        if phases[i]["mode"] == phases[i - 1]["mode"]:
            return False, f"phases {i-1} and {i} are both {phases[i]['mode']}"

    # rule 4: cross-side fact requirement depends on shape.
    # phases now contain ONLY working phases (no phantom starters);
    # start_mode is a top-level field — runtime inserts switches at every
    # boundary where consecutive modes differ.
    shape = task.get("shape") or "two_side_single_switch"
    start_mode = task.get("start_mode")
    if start_mode not in ("S", "O"):
        return False, f"start_mode must be 'S' or 'O', got {start_mode!r}"
    sides = {table[k]["side"] for k in cited}
    for p in phases:
        sides.update(table[k]["side"] for k in (p.get("fact_keys") or []))
    # subgoal must not mention switch wording (runtime inserts switch)
    for i, p in enumerate(phases):
        sg = (p.get("subgoal") or "").lower()
        if "switch" in sg or "hand off" in sg or "handoff" in sg:
            return False, (f"phases[{i}].subgoal must not mention switch / "
                           f"hand off — the runtime inserts switches "
                           f"automatically at boundaries")
    if shape == "single_side_switch":
        # Single-side facts; exactly ONE working phase; start_mode is OPPOSITE.
        if len(sides) != 1:
            return False, (f"single_side_switch must have facts on EXACTLY "
                           f"one side; got {sides}")
        work_side = next(iter(sides))
        if len(phases) != 1:
            return False, (f"single_side_switch must have EXACTLY 1 working "
                           f"phase (no phantom starter); got {len(phases)}")
        if phases[0]["mode"] != work_side:
            return False, (f"single_side_switch phases[0].mode must equal the "
                           f"side where facts live ({work_side}); got "
                           f"{phases[0]['mode']}")
        if start_mode == work_side:
            return False, (f"single_side_switch start_mode must be the OPPOSITE "
                           f"of where facts live; facts are {work_side}-side "
                           f"but start_mode is {start_mode} — without the "
                           f"mismatch the agent has no switch to perform")
    else:
        if not {"S", "O"}.issubset(sides):
            return False, f"{shape} task must span both sides; got {sides}"
        if start_mode != phases[0]["mode"]:
            return False, (f"{shape} start_mode must equal phases[0].mode "
                           f"(no leading switch for two_side/multi tasks); got "
                           f"start_mode={start_mode}, phases[0].mode={phases[0]['mode']}")

    # SWITCH REQUIREMENT — every task must contain >=1 mode switch (leading or
    # inter-phase). Belt-and-suspenders: the shape rules above already imply
    # this, but enforce it directly so a future prompt change can't slip a
    # single-mode no-switch task through.
    n_switches = sum(1 for i in range(1, len(phases))
                     if phases[i]["mode"] != phases[i - 1]["mode"])
    if start_mode != phases[0]["mode"]:
        n_switches += 1
    if n_switches < 1:
        return False, ("task must contain >=1 switch (leading or "
                       "inter-phase); got 0 — this is not a hybrid task")
    # shape/crossing consistency: multi_crossing means what it says.
    if shape == "multi_crossing" and n_switches < 2:
        return False, (f"shape=multi_crossing requires >=2 switches; plan has "
                       f"{n_switches} — relabel as two_side_single_switch")

    # S-START ANCHOR — an S-start task ships with web="" (no URL), so the
    # question text is the agent's ONLY context. It must name the site, or
    # the agent cannot know whose EPS / which catalog / what database the
    # question refers to ("did the company beat expectations?" → which?).
    if start_mode == "S" and domain:
        root = domain.split(".")[0].lower()
        if root and root not in question.lower():
            return False, (f"S-start task ships with web=\"\" — the question "
                           f"must name the site ('{root}' / {domain}) so the "
                           f"agent has an anchor; rewrite the question to "
                           f"mention it naturally (e.g. 'on {domain}, ...')")

    # DIFFICULTY SCORE — score = N_facts + N_switches. Difficulty label must
    # match the score band (easy 2-3, medium 4-5, hard 6+). A SHAPE-1 with
    # one fact + one leading switch = score 2 = easy.
    score = len(cited) + n_switches
    dif = (task.get("difficulty") or "").lower()
    if dif == "easy" and score > 3:
        return False, (f"difficulty=easy but score={score} (facts={len(cited)}, "
                       f"switches={n_switches}); easy requires score 2-3")
    if dif == "medium" and not (4 <= score <= 5):
        return False, (f"difficulty=medium but score={score} (facts={len(cited)}, "
                       f"switches={n_switches}); medium requires score 4-5")
    if dif == "hard" and score < 6:
        return False, (f"difficulty=hard but score={score} (facts={len(cited)}, "
                       f"switches={n_switches}); hard requires score >=6")

    # rule 5: question must NOT contain answer — no cited fact value substring
    # may appear in the question text. Tokens are checked at length >=4 and
    # exclude bare year digits / common stopwords / site name / generic
    # category attribute words (which describe the answer's SHAPE, not its
    # specific value).
    qnorm = " " + question.replace(",", "").lower() + " "
    STOP = {"the", "and", "for", "with", "from", "this", "that", "what",
            "have", "your", "into", "over", "than", "are", "you", "is"}
    # The site is by definition the subject of the task — its name MUST be
    # allowed in the question. Strip the TLD; lowercase.
    SITE_NAMES: set[str] = set()
    if domain:
        # 'belmond.com' → {'belmond'}; 'lowermybills.com' → {'lowermybills'}
        # also keep multiword versions ('lower my bills' → 'lowermybills')
        root = domain.split(".")[0].lower()
        if root:
            SITE_NAMES.add(root)
        # also allow the title-cased form to match (e.g. 'LowerMyBills' as
        # one token from anchors_from_value)
    # Generic attribute / category words: shape of the answer, not its
    # content. e.g. 'price', 'total', 'duration', 'distance' are the kind of
    # thing a real user WOULD say in the question ('what's the total price?')
    # and shouldn't be treated as a value leak.
    GENERIC = {
        # money / amount shape
        "price", "prices", "cost", "costs", "total", "totals", "amount",
        "amounts", "fee", "fees", "charge", "charges", "value", "tax",
        "taxes", "subtotal", "discount", "deposit", "credit", "credits",
        # quantity / metric shape
        "distance", "duration", "difficulty", "weight", "height", "length",
        "width", "size", "sizes", "rating", "score",
        # generic dimensions found everywhere
        "name", "names", "title", "category", "type", "color", "options",
        "details", "description", "page", "pages", "site", "link", "links",
        "section", "sections", "list", "feature", "features", "available",
        # checkout / commerce UI labels
        "checkout", "cart", "basket", "order", "shipping", "delivery",
        "billing", "address", "payment", "summary",
        "recommended", "items", "results",
        # locale / time
        "today", "tomorrow", "tonight", "morning", "evening", "weekend",
        # action verbs that appear in both Q and value
        "click", "scroll", "open", "close", "select", "search", "find",
        # spaces / rooms / common nouns
        "garden", "room", "rooms", "suite", "suites", "hotel", "hotels",
        # geographic granularity (when a user says 'in Italy' or 'Wisconsin'
        # in the question, they're CONSTRAINING — the specific city/landmark
        # is the answer, not the country/state)
        "italy", "france", "spain", "japan", "china", "england", "europe",
        "africa", "america", "asia", "australia", "wisconsin", "california",
        "texas", "florida", "newyork", "google", "amazon", "facebook",
        "microsoft", "apple",
    }
    leaks = []
    for k in cited:
        for tok in anchors_from_value(table[k]["value"]):
            t = tok.replace(",", "").lower()
            if len(t) < 4:
                continue
            if t in STOP:
                continue
            if t in GENERIC:
                continue
            if t in SITE_NAMES:
                continue
            # Also strip whitespace from multi-word tokens before checking
            t_compact = t.replace(" ", "")
            if t_compact in SITE_NAMES:
                continue
            if re.fullmatch(r"20\d{2}", t):  # bare year is allowed
                continue
            if f" {t} " in qnorm or f" {t}." in qnorm or f" {t}?" in qnorm:
                leaks.append((k, tok))
    if leaks:
        sample = ", ".join(f"{k}→'{v}'" for k, v in leaks[:3])
        return False, (f"question leaks cited fact value(s): {sample}. "
                       f"Restate as constraints; the answer must stay in "
                       f"answer_sketch / chain.")

    # rule 8: ENTITY PROVENANCE — every multi-word proper-noun phrase in the
    # question must be traceable to the fact table (keys, values), the domain,
    # or the category. Guards against the synth model expanding a ticker /
    # abbreviation into an invented full name (e.g. VSNT → "Versant Health"
    # when the site actually shows Versant Media Group): the agent then chases
    # a non-existent entity. Single capitalized words are not checked (too
    # many false positives); leading/trailing stopwords are stripped.
    _PROV_STOP = {
        "i'm", "i", "what", "which", "who", "whose", "how", "when", "where",
        "why", "does", "do", "did", "is", "are", "was", "were", "can", "could",
        "should", "would", "the", "a", "an", "on", "in", "at", "to", "for",
        "from", "of", "and", "or", "but", "also", "if", "then", "their",
        "its", "his", "her", "my", "your", "our", "this", "that", "these",
        "those", "with", "by", "as", "after", "before", "include", "including",
    }
    corpus_parts = [domain, domain.split(".")[0]]
    for k in table:
        corpus_parts.append(k.replace("_", " "))
        corpus_parts.append(str(table[k].get("value") or ""))
    corpus_raw = " ".join(corpus_parts).lower()
    # Build a token bag from the corpus so phrasings differ in punctuation,
    # possessives, or joiner words but still trace back. "Belmond Hotel
    # Caruso" in corpus → tokens {belmond, hotel, caruso}; question phrase
    # "Belmond's Caruso" → tokens {belmond, caruso} ⊂ corpus_tokens → OK.
    def _toks(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z][a-z0-9]+", text.lower())
                if len(w) >= 3}
    corpus_tokens = _toks(corpus_raw)
    unsourced = []
    for m in re.finditer(
            r"\b[A-Z][\w&.'-]*(?:\s+[A-Z][\w&.'-]*)+\b", question):
        words = m.group(0).split()
        while words and words[0].lower().strip(".,'") in _PROV_STOP:
            words = words[1:]
        while words and words[-1].lower().strip(".,'") in _PROV_STOP:
            words = words[:-1]
        if len(words) < 2:
            continue
        phrase = " ".join(words)
        # First try the original substring match (most permissive).
        if phrase.lower() in corpus_raw:
            continue
        # Fallback: per-token coverage. Every content word in the phrase
        # must appear somewhere in the corpus. Strips possessives (`'s`),
        # hyphens, ampersands, etc.
        phrase_tokens = _toks(phrase)
        if phrase_tokens and phrase_tokens <= corpus_tokens:
            continue
        unsourced.append(phrase)
    if unsourced:
        return False, (f"question names entity phrase(s) with NO source in the "
                       f"fact table: {unsourced[:3]}. Never expand a ticker or "
                       f"abbreviation into a full name the facts don't show — "
                       f"use the exact name from the facts, the bare "
                       f"ticker/abbreviation, or a generic description.")
    return True, ""


def derive_meta(task: dict, table: dict) -> dict:
    """Compute meta from validated task. start_mode comes from the LLM
    (validated by shape rules); we don't overwrite it from phases[0].mode
    because SHAPE-1 deliberately has start_mode ≠ phases[0].mode.

    Also expand `phases` (working-only) into a typed `route_plan` that
    explicitly lists every switch — easier for hybrid_guided_routed router
    and for the viewer to render.
    """
    phases = task.get("phases") or []
    start_mode = task.get("start_mode") or (phases[0]["mode"] if phases else "O")
    crossings = sum(1 for i in range(1, len(phases))
                    if phases[i]["mode"] != phases[i - 1]["mode"])
    # Add a leading crossing if start_mode differs from phases[0].mode
    # (the runtime's first action must be a switch).
    if phases and start_mode != phases[0]["mode"]:
        crossings += 1
    o_keys = [k for p in phases if p["mode"] == "O" for k in (p.get("fact_keys") or [])]
    s_keys = [k for p in phases if p["mode"] == "S" for k in (p.get("fact_keys") or [])]
    # Build typed route_plan with explicit switch entries.
    route_plan = []
    prev_mode = start_mode
    for p in phases:
        if p["mode"] != prev_mode:
            route_plan.append({
                "type": "switch",
                "to": p["mode"],
                "reason": (f"hand off to {p['mode']}-side: "
                           f"{p.get('subgoal', '')[:100]}"),
            })
        route_plan.append({
            "type": "task",
            "mode": p["mode"],
            "subgoal": p.get("subgoal", ""),
            "done_when": p.get("done_when", ""),
        })
        prev_mode = p["mode"]
    score = len(o_keys) + len(s_keys) + crossings
    return {"start_mode": start_mode, "n_phases": len(phases),
            "n_crossings": crossings,
            "score": score,
            "shape": task.get("shape") or "two_side_single_switch",
            "o_fact_keys": o_keys, "s_fact_keys": s_keys,
            "route_plan_typed": route_plan}


# ---------------- per-site synthesis ----------------
def _validate_task_set(obj: dict, table: dict, domain: str = "") -> tuple[list[dict], list[str]]:
    """Returns (accepted_tasks, per_index_reasons). Each task is validated
    independently; then set-level overlap is checked over accepted ones."""
    if obj.get("kind") == "no_task":
        return [], ["set:no_task"]
    if obj.get("kind") != "task_set":
        return [], [f"set:wrong kind {obj.get('kind')!r} (need task_set)"]
    raw_tasks = obj.get("tasks") or []
    if not isinstance(raw_tasks, list) or not raw_tasks:
        return [], ["set:tasks is empty or not a list"]
    accepted: list[dict] = []
    reasons: list[str] = []
    for i, t in enumerate(raw_tasks):
        if not isinstance(t, dict):
            reasons.append(f"task[{i}]: not an object")
            continue
        wrapped = dict(t); wrapped["kind"] = "task"
        ok, why = validate(wrapped, table, domain)
        if not ok:
            reasons.append(f"task[{i}]: {why}")
            continue
        accepted.append(wrapped)
    # set-level overlap on accepted tasks
    if len(accepted) >= 2:
        ov = _set_overlap_violations(accepted)
        if ov:
            # drop the LAST task involved in the first overlap to make progress
            # (model can re-aim it on retry); flag for retry feedback
            reasons.append(f"set:{ov}")
            return accepted, reasons
    return accepted, reasons


def synth_one(site_dir: Path, retries: int, n_tasks: int) -> dict:
    domain = site_dir.name
    try:
        table, site = load_fact_table(site_dir)
    except Exception as exc:
        return {"domain": domain, "kind": "error",
                "reason": f"load: {type(exc).__name__}: {exc}",
                "tasks": []}
    n_o = sum(1 for v in table.values() if v["side"] == "O")
    n_s = sum(1 for v in table.values() if v["side"] == "S")
    counts = {"O": n_o, "S": n_s,
              "harvested": sum(1 for v in table.values()
                               if v["tag"] == "S_HARVESTED")}
    # NOTE: a one-sided table (n_o==0 OR n_s==0) is NOT a no_task case anymore.
    # SHAPE-1 (single_side_switch) explicitly uses single-side fact chains —
    # the agent must switch on turn 1 because its starting mode has no work
    # to do. Only bail when BOTH sides are empty.
    if n_o == 0 and n_s == 0:
        out = {"domain": domain, "site": site, "kind": "no_task",
               "reason": "empty fact table — explorer collected nothing",
               "table_counts": counts, "tasks": []}
        (site_dir / "task.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        return out

    prompt = SYNTH_PROMPT.format(
        n_tasks=n_tasks,
        domain=domain,
        entry_url=site.get("entry_url") or "",
        category=site.get("category") or "?",
        table=render_table(table),
    )
    messages = [{"role": "user", "content": prompt}]
    accepted: list[dict] = []
    final_reasons: list[str] = []
    for attempt in range(retries + 1):
        try:
            raw = call_llm(messages, max_tokens=6000)
        except Exception as exc:
            final_reasons = [f"llm: {exc}"]
            break
        obj = extract_json(raw)
        if not obj:
            final_reasons = ["unparseable JSON"]
        else:
            accepted, reasons = _validate_task_set(obj, table, domain)
            final_reasons = reasons
            if accepted and not any(r.startswith("set:") and "overlap" in r for r in reasons):
                break  # done: have at least 1 valid task and no overlap issue
        # retry with feedback
        messages.append({"role": "assistant", "content": raw})
        feedback = ("VALIDATION FEEDBACK:\n  " + "\n  ".join(final_reasons or ["unknown"])
                    + f"\n\nReturn a corrected task_set JSON with up to {n_tasks} tasks. "
                      "If you can only produce fewer valid tasks, return fewer.")
        messages.append({"role": "user", "content": feedback})

    if not accepted:
        out = {"domain": domain, "site": site, "kind": "rejected",
               "reason": "; ".join(final_reasons)[:600], "table_counts": counts,
               "tasks": []}
        (site_dir / "task.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        return out

    # decorate accepted tasks with derived meta
    for t in accepted:
        t.update(derive_meta(t, table))
    out = {"domain": domain, "site": site, "kind": "task_set",
           "table_counts": counts, "n_tasks": len(accepted),
           "tasks": accepted,
           "retry_reasons": final_reasons[:6]}
    (site_dir / "task.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--n-tasks", type=int, default=3,
                    help="target number of distinct tasks per site")
    args = ap.parse_args()
    run = Path(args.run)
    site_dirs = sorted(d for d in run.iterdir()
                       if d.is_dir() and (d / "fact_tags.json").is_file())
    print(f"synthesizing for {len(site_dirs)} sites", flush=True)

    rows = []
    with cf.ThreadPoolExecutor(max_workers=args.conc) as pool:
        futs = {pool.submit(synth_one, sd, args.retries, args.n_tasks): sd
                for sd in site_dirs}
        for fut in cf.as_completed(futs):
            r = fut.result()
            rows.append(r)
            kind = r.get("kind")
            if kind == "task_set":
                n = len(r.get("tasks") or [])
                diffs = ",".join(t.get("difficulty", "?") for t in r["tasks"])
                print(f"  [task_set] {r['domain']} — {n} tasks ({diffs})", flush=True)
            else:
                print(f"  [{kind}] {r['domain']} — {r.get('reason','')[:80]}", flush=True)

    # Aggregate rollout-ready rows — one row per task (multi-task per site)
    agg = run / "synth7_tasks.jsonl"
    n_task = 0
    with agg.open("w") as f:
        for r in rows:
            if r.get("kind") != "task_set":
                continue
            for i, t in enumerate(r["tasks"]):
                n_task += 1
                # start_mode drives the `web` field: S-start → web="" so the
                # runtime begins on the search side; O-start → web=entry_url
                # so the runtime begins on the operation side. Same encoding
                # hybrid_guided_routed already uses (has_initial_url ⇒ O).
                start_mode = t["start_mode"]
                entry_url = (r.get("site") or {}).get("entry_url", "")
                web = entry_url if start_mode == "O" else ""
                f.write(json.dumps({
                    "task_id": f"synth7-{r['domain']}-{i}",
                    "subset": (r.get("site") or {}).get("category", "synth7"),
                    "question": t["question"],
                    "web": web,
                    "dataset": "synth7",
                    "split": "test",
                    "record_id": f"synth7-{r['domain']}-{i}",
                    "answer": "",
                    "start_mode": start_mode,
                    # route_plan now uses explicit typed entries so a
                    # `switch` step is materialized for every S<->O boundary
                    # (including the leading mismatch in SHAPE-1 single-side
                    # tasks). Each entry is one of:
                    #   {"type":"task","mode":"S/O","subgoal":..,"done_when":..}
                    #   {"type":"switch","to":"S/O","reason":..}
                    "route_plan": json.dumps(
                        t.get("route_plan_typed") or [],
                        ensure_ascii=False),
                    "_signal": json.dumps({
                        "difficulty": t.get("difficulty"),
                        "shape": t.get("shape"),
                        "n_crossings": t.get("n_crossings"),
                        "score": t.get("score"),
                        "o_fact_keys": t.get("o_fact_keys"),
                        "s_fact_keys": t.get("s_fact_keys"),
                        "answer_sketch": t.get("answer_sketch"),
                    }, ensure_ascii=False),
                }, ensure_ascii=False) + "\n")
    kinds = {}
    n_per_set = []
    for r in rows:
        kinds[r.get("kind")] = kinds.get(r.get("kind"), 0) + 1
        if r.get("kind") == "task_set":
            n_per_set.append(len(r.get("tasks") or []))
    mean_tasks = sum(n_per_set) / max(len(n_per_set), 1) if n_per_set else 0
    print(f"DONE: {kinds}  mean_tasks/site={mean_tasks:.1f}  "
          f"total_tasks={n_task} -> {agg}", flush=True)


if __name__ == "__main__":
    main()

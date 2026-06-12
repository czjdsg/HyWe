"""v8 explorer: decoupled OPERATOR + FACT-EXTRACTOR agents.

v7 had a single agent doing 4 jobs per turn (see / plan / act / extract).
This left facts always rushed and shallow. v8 splits the job:

  OPERATOR (kimi-k2.6, vision)
    - sees current screenshot + browser state + recent action history
    - outputs ONLY a tool_call (the next browser action) + a 1-line thought
    - never writes facts_learned; that field is removed from its schema

  FACT EXTRACTOR (claude-sonnet-4-6, vision)
    - runs AFTER every OPERATOR turn whose action succeeded
    - sees the full recent trajectory: every prior step's action +
      observation_head + post-action screenshot, plus current screenshot
      and KB-so-far + open_questions
    - outputs zero or more facts_learned + optional facts_to_pursue
    - extractor MAY decide there is nothing new and emit [] — this is fine
    - extractor is encouraged to recognise multi-step constraint chains
      ("user filtered by X then sorted by Y then clicked row 3 → THIS is
       the constrained result")

CAPTCHA hard-stop
    - Before each OPERATOR turn we check the current page's text. If it
      matches the CAPTCHA fingerprint, the run STOPS immediately and the
      result is marked walled. No more "click Accept All Cookies 10 times".

v7 file (explorer_agent.py) is untouched and remains the rollback target.
"""
from __future__ import annotations

import asyncio
import base64
import copy
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# Reuse everything we built in v7 — prompts, router, runtime hooks, KB.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from explorer_agent import (   # noqa: E402
    ExplorerAgent as _V7,
    ExplorerRouter,
    SYS_EXPLORE_OPS, SYS_EXPLORE_STAR,
    SYSTEM_PROMPT_BY_MODE,
    TOOL_WHITELIST_BY_MODE,
    JSON_OBJ_RE,
    API_BASE, API_KEY, MODEL,
    LLM_TIMEOUT_S,
    _data_url_from_png,
    _truncate,
    validate_route_plan,
)

# Fact-extractor endpoint (separate model, separate base if user wants)
FACT_API_BASE = os.environ.get("FACT_API_BASE", API_BASE)
FACT_API_KEY = os.environ.get("FACT_API_KEY", API_KEY)
FACT_MODEL = os.environ.get("FACT_MODEL", "aws.claude-sonnet-4-6")
FACT_MAX_HISTORY = int(os.environ.get("FACT_MAX_HISTORY", "8"))
FACT_TRAJ_SHOTS = int(os.environ.get("FACT_TRAJ_SHOTS", "2"))  # how many prior screenshots to pass

# CAPTCHA fingerprints in the rendered page text. If any of these phrases
# appears in current_title / current_url / observed page text right before a
# turn, we STOP the run — repeated clicking won't get us past this. Better
# to spend the budget on a different site than burn 30 turns dismissing a
# wall that won't move.
_CAPTCHA_PATTERNS = re.compile(
    r"are you a (?:human|person|robot)"
    r"|verify(?:ing)? you are (?:human|not a robot)"
    r"|press(?:ing)? and hold"
    r"|complete (?:the )?(?:captcha|security check)"
    r"|enable (?:javascript|cookies) and reload"
    r"|cloudflare.*(?:challenge|checking your browser)"
    r"|akamai (?:bot manager|reference)"
    r"|access denied"
    r"|forbidden"
    r"|reference (?:id|#)\s*[a-f0-9-]{8,}"
    r"|err_(?:http2_protocol_error|too_many_redirects|connection_)",
    re.I,
)


def looks_walled(state: dict, last_obs: dict | None) -> str | None:
    """Returns a short reason string if the page is unrecoverably walled,
    else None. Conservative: prefers letting the agent try over false
    positives that would kill legitimate runs."""
    title = (state.get("current_title") or "").strip()
    url = (state.get("current_url") or "").strip()
    text = (state.get("page_text") or "")[:4000]
    blob = f"{title}\n{url}\n{text}"
    if _CAPTCHA_PATTERNS.search(blob):
        return f"captcha/wall fingerprint matched at {url}"
    # walled status carried from the runtime
    if last_obs and isinstance(last_obs, dict):
        diag = last_obs.get("diagnostics") or {}
        if isinstance(diag, dict) and diag.get("walled"):
            return f"runtime reported walled: {diag.get('walled_reason','')}"
    return None


# ----------------------------- OPERATOR prompt -----------------------------
# We take v7's SYS_EXPLORE_OPS as base context (it contains the whole tool
# catalogue, ACTION strategy, EXTRA O-only handles, trigger/result rules,
# banned key suffixes etc. — all still applicable to the OPERATOR), and
# append a v8 override that:
#   1. removes facts_learned / facts_to_pursue from the operator's schema,
#   2. tells it that a separate fact-extractor handles extraction,
#   3. tells it to PLAN AHEAD across turns (depth of work matters).
# The override appears LAST in the system prompt — LLMs treat trailing
# instructions as authoritative, so this safely supersedes the v7 schema.
_OPERATOR_OVERRIDE = """
=== v8 OPERATOR OVERRIDE — read this LAST, it supersedes the schema above ===

You are the OPERATOR agent. Your single job is to drive the browser one
action at a time. A separate FACT-EXTRACTOR agent reads the observation
AFTER your action runs and writes any facts that resulted. You MUST NOT
output facts_learned or facts_to_pursue.

Your output schema is REDUCED to:
  {"kind":"act",
   "thought":"<1-3 short lines: what you observe, the multi-step plan you
              are advancing, and why THIS action is the next step>",
   "tool_call":{"name":"<tool>","arguments":{...},
                "action_description":"<5-15 words: visible target + outcome>"}}

ANY facts_learned / facts_to_pursue / visual_note field you emit will be
silently discarded. Visual cues are NOT your responsibility — the FACT
extractor sees every screenshot you reach and records visual facts
(layout, color, image content, pin position, chart shape) as facts in
its own KB. You focus on driving the browser; it focuses on recording.

EXPLORATION DEPTH — this is where v7 fell short:
- The point of being on the live site is to TRIGGER constrained results
  (filter applied, search submitted, item drilled into, configurator
  configured). A run that stays on the homepage forever produces nothing
  worth extracting.
- Treat each session as a real user with a goal: pick a goal in step 1
  ("find the cheapest hotel in Paris for Jun 9-11, 8+ rating"), then drive
  toward it across many turns. Don't bounce around.
- Dismissing cookie popups, opening filter panels, etc. are SETUP — they
  produce no fact and that's fine. Just push through to a result state.

CAPTCHA HARD-STOP:
- If you see ANY captcha / "are you human" / press-and-hold / 403 / access
  denied screen, do NOT try to dismiss it. The driver will detect this and
  stop the run automatically. Wasting turns on it is forbidden.

COOKIE / CONSENT BANNERS — anti-loop rule:
- A cookie/consent banner is a one-shot dismissal. Click "Accept" or
  "Reject" or close-X exactly ONCE. If the SAME banner is still visible
  after your dismissal, the click did not register and clicking the same
  button again will keep failing. Switch strategies on the second attempt:
  try the alternate button (Accept↔Reject), or `goto` to a deep URL
  (search/results/category page) that often bypasses the consent overlay,
  or use `press_key` with key="Escape", or `scroll` past the overlay and
  click a content link instead.
- Reading your own prior `action_description` text in history matters:
  if your last 2 actions were both "Click ACCEPT ALL on cookie banner",
  STOP doing that — pick a different tool (`goto` deep URL /
  `press_key` Escape / `scroll` then click a content link).
- The fact extractor will not record anything until you reach a real
  RESULT state, so every wasted cookie-banner click is a fact you'll
  never get to write.

No code fences, no prose outside the JSON, exactly ONE tool_call.
"""

OPERATOR_SYS = SYS_EXPLORE_OPS + _OPERATOR_OVERRIDE


# ----------------------------- FACT prompt -----------------------------
# We inherit v7's `facts_learned` knowledge body wholesale — it carries the
# 12-hour-tuned rules (TRIGGER vs RESULT pairs, IMAGE-EMBEDDED facts whitelist,
# 9 BANNED categories with renamed-CAPTCHA examples, banned key suffixes,
# multi-attribute completeness with the Bryce Canyon / VSNT / cart-total
# samples, "real user search box" self-check). We then add v8-only deltas:
# the extractor sees a TRAJECTORY rather than a single turn, output is
# optional (it can return empty when the just-executed turn was setup), and
# the key is encouraged to encode the multi-step constraint chain.
import re as _re

# pull the canonical `facts_learned` + IMAGE-EMBEDDED + trigger/result +
# BANNED sections out of v7's OPS prompt. Boundaries are stable since v7.
_V7_OPS = __import__("explorer_agent").SYS_EXPLORE_OPS
_m_start = _V7_OPS.find("`facts_learned` — CRITICAL.")
_m_end = _V7_OPS.find("`facts_to_pursue`")
if _m_start == -1 or _m_end == -1:
    raise RuntimeError("Could not slice v7 facts_learned section — prompt drifted")
_V7_FACTS_BLOCK = _V7_OPS[_m_start:_m_end].strip()

FACT_SYSTEM = (
    "You are the FACT-EXTRACTOR for a hybrid web exploration run. A separate\n"
    "OPERATOR agent drives the browser one turn at a time. After every\n"
    "successful operator action you are invoked. You look at the entire\n"
    "trajectory so far (every action + observation + screenshot in the\n"
    "operator's history, plus the CURRENT post-action screenshot and KB\n"
    "already collected), and decide whether the just-executed turn produced\n"
    "any new user-interest fact worth recording.\n"
    "\n"
    "FACT RULES — these are inherited verbatim from v7 and they ARE the\n"
    "definition of a fact for this whole pipeline. Read them carefully — the\n"
    "TRIGGER/RESULT pairs, the BANNED categories (including CAPTCHA renames),\n"
    "the IMAGE-EMBEDDED whitelist, and the multi-attribute completeness\n"
    "standard all apply unchanged:\n"
    "\n"
    + _V7_FACTS_BLOCK +
    "\n\n"
    "=== v8 TRAJECTORY-AWARE deltas — read this LAST, it supersedes details ===\n"
    "\n"
    "1. KEY THE FULL CHAIN. The best v8 facts are the END-STATE of a\n"
    "   multi-step constraint chain. The operator may have filtered city,\n"
    "   picked dates, sorted by price, then clicked into row 3 — the FACT is\n"
    "   the resulting fully-constrained item. Encode the chain in the key:\n"
    "     GOOD: 'paris_jun9_jun11_2adult_8plus_rating_lowest_hotel'\n"
    "           'logistics_app_search_top1_designer_profile_location'\n"
    "           'tokyo_jun11_13_under_$200_freecancel_top_property'\n"
    "     BAD: 'hotel_name' (no chain context — search agent can't reproduce)\n"
    "   And pack EVERY constraint from the chain plus EVERY attribute visible\n"
    "   on the result row into the value, per the multi-attribute rule above.\n"
    "\n"
    "2. OUTPUT IS OPTIONAL. Returning facts_learned=[] is the correct answer\n"
    "   when the just-executed turn was housekeeping: dismissed a cookie\n"
    "   modal, opened a filter panel that hasn't been applied yet, scrolled\n"
    "   without revealing new constrained data, hit an error/timeout. Do NOT\n"
    "   pad with anything ban-listed above. Quality > quantity.\n"
    "\n"
    "3. NEVER REPEAT KB. The KB-so-far block in the user message lists every\n"
    "   fact already recorded. If the latest action only redisplayed something\n"
    "   already there, return facts_learned=[]. Do not rephrase a known fact\n"
    "   into a new key — the explorer's absorb step normalizes keys, so dupes\n"
    "   are silently dropped anyway and you waste a turn.\n"
    "\n"
    "4. USE THE WHOLE TRAJECTORY, not just this turn. You see the last N\n"
    "   operator turns with their actions, observation summaries, and prior\n"
    "   screenshots, plus the current screenshot. The current page may only\n"
    "   make sense in light of what was applied 4 steps ago. That's exactly\n"
    "   the value-add you have over the operator — use it.\n"
    "\n"
    "5. facts_to_pursue is HOW YOU TALK BACK to the operator. List 0-2 open\n"
    "   questions the operator should chase next (e.g. 'expand the dimmed\n"
    "   product to read its price', 'open the 'see all' panel to count\n"
    "   results'). The operator sees these as the next-step prompt.\n"
    "\n"
    "6. VISUAL FACTS ARE YOURS TOO. The operator no longer writes any visual\n"
    "   note — every visual cue lands here. Things like:\n"
    "     - 'Top result is a mustard-yellow accent chair against slate-blue wall'\n"
    "     - '3rd map pin sits inside a triangular plaza next to a curved river'\n"
    "     - 'Movie poster: red text on a black silhouette of a city skyline'\n"
    "     - 'Product carousel image #2 is a top-down flatlay of 5 ceramic mugs'\n"
    "   are PROPER facts — emit them as facts_learned rows when the visual\n"
    "   detail is what a downstream agent would need to relocate the item or\n"
    "   verify the claim. Key them by what's depicted, e.g.\n"
    "   'paris_jun9_jun11_top1_hotel_listing_photo_visual', not 'visual_note'.\n"
    "   Generic visual chrome (button colors, header layouts, font weights) is\n"
    "   still banned — same standard as text facts: a real user would ask\n"
    "   about this concrete depicted thing, not about the design system.\n"
    "\n"
    "7. UPDATE-IN-PLACE WHEN A FACT GROWS. If the latest turn revealed MORE\n"
    "   about a fact you already wrote (e.g. you wrote 'controls_components'\n"
    "   with 6 items visible, the operator scrolled and you now see 4 more),\n"
    "   DO NOT write a new key like '_full' or '_v2' — that bloats KB. Emit a\n"
    "   facts_updated row instead and the absorb step will replace the old\n"
    "   value end-to-end. Use updates when:\n"
    "     - the same entity got a more complete attribute list (added price,\n"
    "       added dimensions, added end-of-list confirmation)\n"
    "     - a partial list became complete\n"
    "     - a visual fact gained a sharper anchor\n"
    "   Use facts_learned (new key) ONLY when this is a genuinely different\n"
    "   entity / page / result row from anything in KB.\n"
    "\n"
    "Output EXACTLY ONE JSON object — no code fences, no prose around it:\n"
    "  {\"facts_learned\":[\n"
    "      {\"key\":\"<snake_case slot, encode the constraint chain>\",\n"
    "       \"value\":\"<multi-attribute complete sentence(s), one fact per row>\"}\n"
    "   ],\n"
    "   \"facts_updated\":[\n"
    "      {\"key\":\"<EXISTING key from KB so far — must match exactly>\",\n"
    "       \"value\":\"<new full value that REPLACES the old one end-to-end>\",\n"
    "       \"reason\":\"<one short clause, e.g. 'list now complete' or\n"
    "                  'added price + dimensions'>\"}\n"
    "   ],\n"
    "   \"facts_to_pursue\":[\"<short open question for the operator>\"],\n"
    "   \"walled\":{\"is_walled\": <true|false>, \"reason\": \"<short>\"},\n"
    "   \"note\":\"<OPTIONAL one sentence; e.g. 'housekeeping turn, no fact'>\"}\n"
    "\n"
    "facts_learned MAY be []. facts_updated MAY be []. facts_to_pursue MAY\n"
    "be []. All three empty is fine for a pure-housekeeping turn.\n"
    "\n"
    "WALLED JUDGMENT — set walled.is_walled=true when the current page is\n"
    "unrecoverable for any agent: bot-detection wall (captcha, 'press and\n"
    "hold', Cloudflare/Akamai 'checking your browser'), access-denied\n"
    "(403/forbidden, 'reference id <hex>'), or a fatal network error page\n"
    "(ERR_HTTP2_PROTOCOL_ERROR, ERR_CONNECTION_RESET) — i.e. continuing on\n"
    "this URL would be wasted budget. DO NOT mark walled for:\n"
    "  - cookie banners (those are dismissable)\n"
    "  - login modals that allow guest browsing on other pages\n"
    "  - empty/slow-loading pages (just yield empty facts)\n"
    "  - pages with content that just happens to mention 'access denied'\n"
    "    as a search result\n"
    "Be conservative — when in doubt, set is_walled=false. The driver will\n"
    "break out of the explore loop only when you actively flag it.\n"
)



def _fmt_facts_block(facts: dict) -> str:
    if not facts:
        return "  (KB empty)"
    lines = []
    for k, v in facts.items():
        side = v.get("mode") or "?"
        step = v.get("step", "?")
        val = (v.get("value") or "")[:200]
        lines.append(f"  [{side}@step{step}] {k} = {val}")
    return "\n".join(lines)


class ExplorerAgentV8(_V7):
    """Decoupled operator / fact-extractor variant of the v7 ExplorerAgent.

    We inherit v7 to reuse: __init__, runtime hookup, router state machine,
    KB, _checkpoint, _snapshot, _parse_response, _call_llm (for operator),
    notes/history persistence. We OVERRIDE:
      - _build_messages: drop facts_learned/facts_to_pursue from operator
                         schema (use OPERATOR_SYS), keep STAR & RECORD as-is.
      - run: per-turn loop now does (a) captcha check (b) operator step
             (c) fact-extractor step, with proper checkpointing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Walled flag — set by the fact extractor when it judges the page
        # unrecoverable; consumed by the main loop on the NEXT operator turn.
        self._walled_flag: str | None = None
        # Optional override path so the v8 sites_meta gets v8-specific notes.
        self.notes.append({"agent": "v8", "operator_model": MODEL,
                           "fact_model": FACT_MODEL})

    # ---------- operator prompt (no facts schema) ----------
    def _build_messages(self, state: dict, mode: str) -> list[dict]:
        if mode == ExplorerRouter.ACT_OPS:
            # Swap system prompt to operator-only schema; keep the user
            # message construction logic from v7 unchanged.
            sys_text = OPERATOR_SYS
        else:
            sys_text = SYSTEM_PROMPT_BY_MODE.get(mode, SYS_EXPLORE_STAR)
        msgs = super()._build_messages(state, mode)
        msgs[0] = {"role": "system", "content": sys_text}
        return msgs

    # ---------- fact extractor ----------
    async def _call_fact_model(self, messages: list[dict]) -> str:
        body = {
            "model": FACT_MODEL,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 16000,
        }
        last = None
        for attempt in range(4):
            try:
                async with httpx.AsyncClient(timeout=LLM_TIMEOUT_S) as c:
                    r = await c.post(
                        f"{FACT_API_BASE}/chat/completions",
                        json=body,
                        headers={"Authorization": f"Bearer {FACT_API_KEY}",
                                 "Content-Type": "application/json"})
                    if r.status_code == 200:
                        return r.json()["choices"][0]["message"]["content"] or ""
                    if r.status_code == 429 or 500 <= r.status_code < 600:
                        last = httpx.HTTPStatusError(
                            f"{r.status_code}", request=r.request, response=r)
                    else:
                        # 4xx fatal — don't loop
                        raise httpx.HTTPStatusError(
                            f"fatal {r.status_code}: {r.text[:200]}",
                            request=r.request, response=r)
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429,) or 500 <= e.response.status_code < 600:
                    last = e
                else:
                    raise
            except Exception as e:
                last = e
            await asyncio.sleep(min(30.0, 5.0 * (2 ** attempt)))
        raise RuntimeError(f"fact model failed: {last}")

    def _build_fact_messages(self, state: dict) -> list[dict]:
        """Trajectory-aware prompt for the fact extractor.

        Carries: site frame, KB so far, open questions, the recent operator
        actions + observation_head, the post-action screenshot for each of
        the last FACT_TRAJ_SHOTS turns, then the CURRENT screenshot.
        """
        site = self.site
        site_frame = (
            f"Target site: {site.get('domain','?')}  ({site.get('category','?')})\n"
            f"Entry URL: {site.get('entry_url','')}\n"
        )
        kb = self.router.kb_snapshot() if hasattr(self, "router") else {"facts": {}}
        facts_block = _fmt_facts_block(kb.get("facts") or {})
        open_qs = kb.get("open_questions") or []
        oq_block = ("\n".join(f"  - {q}" for q in open_qs)
                    if open_qs else "  (none yet)")

        recent = self.history[-FACT_MAX_HISTORY:]
        # narrate trajectory (text-only diff of actions/observations)
        traj_lines = []
        for h in recent:
            resp = h.get("response_json", {}) or {}
            tc = resp.get("tool_call") or {}
            ad = tc.get("action_description") or ""
            obs = h.get("observation", {}) or {}
            if isinstance(obs, dict):
                # show a compact summary
                obs_head = (
                    f"status={obs.get('status')} "
                    f"url={(obs.get('ended_url') or '')[:60]} "
                    f"err={(obs.get('error') or '')[:80]}"
                )
            else:
                obs_head = _truncate(str(obs), 280)
            traj_lines.append(
                f"--- step {h['step']} ---\n"
                f"OPERATOR did: {tc.get('name','?')}  {ad}\n"
                f"OBSERVATION: {obs_head}"
            )
        traj_blob = "\n\n".join(traj_lines) if traj_lines else "(no prior turns)"

        cur_url = state.get("current_url", "")
        cur_title = state.get("current_title", "")
        text_top = (
            f"{site_frame}\n"
            f"## KB so far (do NOT duplicate)\n{facts_block}\n\n"
            f"## Open questions to chase\n{oq_block}\n\n"
            f"## Recent operator trajectory (action + observation)\n"
            f"{traj_blob}\n\n"
            f"## Current state after the latest action\n"
            f"URL: {cur_url}\nTitle: {cur_title}\n\n"
            "Now look at the prior screenshots (for context) and the CURRENT "
            "screenshot (the result of the latest operator action). Decide "
            "what NEW user-interest fact(s) the operator just produced. If "
            "the action was housekeeping (closed a popup, opened a filter "
            "panel that hasn't been applied yet), return facts_learned=[]."
        )

        # vision content: include up to FACT_TRAJ_SHOTS prior screenshots
        # then the current one. Operator agent already only attaches the
        # current shot, so the extractor having multi-step visual context is
        # its main advantage.
        content_parts: list[dict] = [{"type": "text", "text": text_top}]
        prior_shots: list[str] = []
        for h in recent[-FACT_TRAJ_SHOTS:]:
            sp = (h.get("response_json") or {}).get("visit_screenshot_path") or ""
            # Note: we don't have great per-step shots — fall back to the
            # current shot if a prior frame isn't on disk.
            if sp and os.path.exists(sp):
                prior_shots.append(sp)
        cur_shot = state.get("img_path") or ""
        if cur_shot and os.path.exists(cur_shot):
            prior_shots.append(cur_shot)
        for p in prior_shots[-(FACT_TRAJ_SHOTS + 1):]:
            try:
                content_parts.append(
                    {"type": "image_url",
                     "image_url": {"url": _data_url_from_png(p)}})
            except Exception:
                pass

        return [
            {"role": "system", "content": FACT_SYSTEM},
            {"role": "user", "content": content_parts},
        ]

    @staticmethod
    def _parse_fact_response(raw: str) -> dict | None:
        if not raw:
            return None
        s = raw.strip()
        if s.startswith("```"):
            s = s.strip("`")
            if s.lower().startswith("json"):
                s = s[4:]
            s = s.strip()
        try:
            return json.loads(s)
        except Exception:
            m = JSON_OBJ_RE.search(s)
            if not m:
                return None
            try:
                return json.loads(m.group(0))
            except Exception:
                return None

    async def _run_fact_extractor(self, mode: str, state: dict, step_idx: int):
        """Call the extractor; on success absorb into KB. Errors are logged,
        never fatal — extraction is OPTIONAL by design."""
        try:
            messages = self._build_fact_messages(state)
            raw = await self._call_fact_model(messages)
        except Exception as exc:
            self.notes.append({"step": step_idx, "fact_error": str(exc)[:200]})
            return
        parsed = self._parse_fact_response(raw)
        # Record the EXACT post-action screenshot the extractor saw, so the
        # viewer can render the right image alongside the EXTRACTOR turn.
        ext_shot = state.get("img_path") if isinstance(state, dict) else None
        self.notes.append({"step": step_idx, "fact_raw": raw[:1200],
                           "fact_parsed": parsed,
                           "shot_path": ext_shot})
        if not parsed:
            return
        # Walled judgment: the extractor decides whether the page is
        # unrecoverable. We stash the verdict on self._walled_flag so the
        # main loop can break on the NEXT operator turn (before another
        # round of clicks burns budget on a wall the agent can't break).
        walled = parsed.get("walled")
        if isinstance(walled, dict) and walled.get("is_walled"):
            reason = (walled.get("reason") or "extractor flagged walled").strip()
            self._walled_flag = f"extractor: {reason[:200]}"
        raw_facts = parsed.get("facts_learned") or []
        raw_updates = parsed.get("facts_updated") or []
        raw_qs = parsed.get("facts_to_pursue") or []
        before = len(self.router.facts)
        self.router.absorb_facts(mode, raw_facts)
        n_updated = self.router.update_facts(raw_updates)
        self.router.retire_answered_questions(raw_facts)
        self.router.absorb_questions(raw_qs)
        added = len(self.router.facts) - before
        if n_updated:
            self.notes.append({"step": step_idx,
                               "fact_updates_applied": n_updated})
        # An update is "productive" the same way a new fact is — keep the
        # router's dry-streak from prematurely tripping when the extractor
        # is busy polishing existing entries.
        productive = added + n_updated
        # Don't notify_turn from here — the dynamic-budget signal is owned
        # by the operator turn. But DO update facts_added counter so the
        # router's "dry streak" reflects the post-extraction state.
        self.router.update_dynamic_signals(productive)

    # ---------- main loop ----------
    async def run(self) -> dict:
        site = self.site
        try:
            await self.runtime.start()
        except Exception as exc:
            res = {"site": site, "fatal": True,
                   "error": f"runtime.start: {type(exc).__name__}: {exc}",
                   "candidates": [], "history": []}
            try:
                with open(os.path.join(self.runtime.task_dir, "result.json"), "w") as f:
                    json.dump(res, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return res

        step_idx = 0
        walled_at = None
        last_obs = None

        while True:
            step_idx += 1
            mode = self.router.decide()
            if mode == ExplorerRouter.STOP:
                self.notes.append({"step": step_idx, "router_stop": True})
                break
            if step_idx > self.max_steps + 4:
                self.notes.append({"step": step_idx, "router_stop": "safety_cap"})
                break

            try:
                state = await self.runtime.state(role="op")
            except Exception as exc:
                self.notes.append({"step": step_idx, "error": f"state: {exc}"})
                break

            # ---- CAPTCHA / walled hard-stop ----
            # We no longer pattern-match the page text. The fact extractor
            # decides per-turn whether the current page is unrecoverable
            # (sets walled.is_walled=true in its JSON output). The flag is
            # read from self._walled_flag after each extractor call.
            if mode == ExplorerRouter.ACT_OPS and getattr(self, "_walled_flag", None):
                wall_reason = self._walled_flag
                self._walled_flag = None
                self.notes.append({"step": step_idx, "walled_stop": wall_reason,
                                   "url": state.get("current_url")})
                walled_at = wall_reason
                break

            messages = self._build_messages(state, mode)
            await asyncio.sleep(1.5)
            try:
                raw = await self._call_llm(messages, mode=mode)
            except Exception as exc:
                err_str = str(exc)
                self.notes.append({"step": step_idx, "mode": mode,
                                   "error": f"op_llm: {err_str}"})
                if any(t in err_str for t in ("fatal 4", "fatal 401", "fatal 403")):
                    break
                await asyncio.sleep(15)
                self.router.notify_turn(mode, success=False, kind="unknown")
                continue

            parsed = self._parse_response(raw)
            self.notes.append({"step": step_idx, "mode": mode,
                               "raw": raw[:1500], "parsed": parsed})
            self._checkpoint()
            if not parsed:
                self.history.append({"step": step_idx, "mode": mode,
                                     "response_json": {},
                                     "observation": {"status": "format_error",
                                                     "msg": "non-JSON"}})
                self.router.notify_turn(mode, success=False, kind="unknown")
                continue

            kind = parsed.get("kind")
            allowed_kinds = {
                ExplorerRouter.ACT_OPS: {"act"},
                ExplorerRouter.ACT_STAR: {"act"},
                ExplorerRouter.RECORD: {"record_task"},
            }.get(mode, set())
            if kind not in allowed_kinds:
                obs = {"status": "router_violation", "mode": mode,
                       "got_kind": kind,
                       "msg": f"router required {allowed_kinds}, you gave {kind}"}
                self.history.append({"step": step_idx, "mode": mode,
                                     "response_json": parsed, "observation": obs})
                self.router.notify_turn(mode, success=False, kind=kind or "unknown")
                continue

            if mode in (ExplorerRouter.ACT_OPS, ExplorerRouter.ACT_STAR):
                tool_whitelist = TOOL_WHITELIST_BY_MODE.get(mode, set())
                tc_name = (parsed.get("tool_call") or {}).get("name", "")
                if tc_name not in tool_whitelist:
                    obs = {"status": "tool_violation", "mode": mode,
                           "got_tool": tc_name,
                           "msg": f"router required tool in {sorted(tool_whitelist)}, "
                                  f"you used {tc_name!r}"}
                    self.history.append({"step": step_idx, "mode": mode,
                                         "response_json": parsed, "observation": obs})
                    self.router.notify_turn(mode, success=False, kind="act")
                    continue

            if kind == "act":
                tc = parsed.get("tool_call") or {}
                rec = await self.runtime.act(tc)
                obs = {"status": rec.get("status"),
                       "ended_url": rec.get("ended_url"),
                       "error": rec.get("error"),
                       "diagnostics": rec.get("diagnostics")}
                last_obs = obs
                # Record the EXACT pre-action screenshot the operator saw,
                # so the viewer can render the correct image alongside this
                # action (runtime.step auto-increments on every state() call
                # so file-numbered mapping is otherwise unreliable).
                pre_shot = state.get("img_path") if isinstance(state, dict) else None
                self.history.append({"step": step_idx, "mode": mode,
                                     "response_json": parsed, "observation": obs,
                                     "shot_path": pre_shot})
                ok = rec.get("status") == "success"

                # ---- NEW: post-action fact extraction (optional) ----
                if ok:
                    # Refresh state to see the post-action page
                    try:
                        post_state = await self.runtime.state(role="ext")
                    except Exception:
                        post_state = state
                    await self._run_fact_extractor(mode, post_state, step_idx)
                else:
                    # failed action — still notify router but no extraction
                    self.router.update_dynamic_signals(0)

                self.router.notify_turn(mode, success=ok, kind="act")
            elif kind == "record_task":
                # RECORD path unchanged from v7 — it writes a candidate task,
                # not a fact, so we don't gate it on the extractor.
                task = parsed.get("task") or {}
                question_ok = bool((task.get("question") or "").strip())
                kb_snap = self.router.kb_snapshot()
                plan_ok, plan_reason = validate_route_plan(task, kb=kb_snap)
                if question_ok and plan_ok:
                    self.candidate_tasks.append(task)
                    obs = {"status": "recorded",
                           "tasks_so_far": len(self.candidate_tasks)}
                    success = True
                else:
                    obs = {"status": "record_invalid",
                           "msg": plan_reason or "missing question"}
                    success = False
                self.history.append({"step": step_idx, "mode": mode,
                                     "response_json": parsed, "observation": obs})
                self.router.notify_turn(mode, success=success, kind="record")

        # ----- finalize -----
        snap = self._snapshot()
        snap["walled_stop"] = walled_at
        snap["candidates"] = self.candidate_tasks
        snap["elapsed_s"] = getattr(self, "_elapsed_s", None)
        try:
            with open(os.path.join(self.runtime.task_dir, "result.json"), "w") as f:
                json.dump(snap, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        try:
            await self.runtime.stop()
        except Exception:
            pass
        return snap

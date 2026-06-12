"""Explorer Agent: a self-directed ReAct loop that uses ExplorerRuntime to
explore a website and emit candidate hybrid tasks as a side-effect.

This is the synth3-B path. It is NOT trying to solve a task; instead it is told
"explore this site, and any time you discover a concrete user-facing task that
genuinely needs interaction, record it."

LLM: qwen3.7-plus on :8002 with vision enabled (screenshots are inlined as
data URLs in the message).
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# Reuse hybrid_prompts tool schemas (single source of truth).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))
try:
    from hybrid_prompts import get_tools_for_mode  # type: ignore
except Exception:
    get_tools_for_mode = None  # graceful fallback; the agent will still run


API_BASE = os.environ.get("EXPLORER_API_BASE", "http://localhost:8000/v1")
API_KEY = os.environ.get("EXPLORER_API_KEY", "EMPTY")
MODEL = os.environ.get("EXPLORER_MODEL", "kimi-k2.6")

# RECORD-stage override: use a stronger model for the terminal task-summary turn.
# Defaults fall back to the explore endpoint/model when unset.
RECORD_API_BASE = os.environ.get("RECORD_API_BASE", API_BASE)
RECORD_API_KEY = os.environ.get("RECORD_API_KEY", API_KEY)
RECORD_MODEL = os.environ.get("RECORD_MODEL", MODEL)
LLM_TIMEOUT_S = int(os.environ.get("EXPLORER_LLM_TIMEOUT", "900"))


SYS_EXPLORE_OPS_TEMPLATE = """You are an exploration worker driving a real
browser. Your only job this turn is to take ONE concrete browser action ON
THE TARGET SITE.

You are NOT trying to solve any task end-to-end. You are scouting the site's
interactive capabilities. The phase you're in lasts several turns — keep
moving the same on-site sub-objective forward (filling a form, applying a
filter, drilling into a result, etc.). Do not split attention across many
unrelated UI areas in one phase.

Available tools and JSON schemas (this is the COMPLETE catalogue for this turn):
{tool_str}

Output EXACTLY ONE JSON object:
  {{"kind":"act",
    "thought":"<1-3 short lines: what you observe and why this action>",
    "tool_call":{{"name":"<tool>","arguments":{{...}},
                  "action_description":"<5-15 words: visible target + outcome>"}},
    "visual_note":"<OPTIONAL one short sentence — see rules below>",
    "facts_learned":[
      {{"key":"<short id, lowercase_underscore>","value":"<concrete value, 1 short line>"}}
    ],
    "facts_to_pursue":["<question to chase NEXT, one short line>"]}}

Action descriptions: 5-15 words, name the visible target element and the
expected outcome (helps future-you avoid loops).

ACTION strategy — what to DO this turn:
- The point of being on the live O-side site is to TRIGGER result-states a
  real user would care about. Spend your acts on:
    * Filtering / sorting / faceting the result space
    * Submitting search forms with specific constraints (dates, locations,
      categories, price ranges)
    * Browsing image carousels / galleries / map pins to surface item details
    * Drilling into a top result to read its full attributes
    * Opening configurators / option pickers and reading what gets produced

- EXTRA O-only handles — search engines CANNOT reproduce these; prefer them
  when the site supports them. Each one is a high-value fact slot:
    * Real-time / right-now snapshots — "now-streaming top channel viewer
      count", "current stock intraday change %", "today's home page hero
      banner", "live game score this quarter", "wait-time this minute".
      Search returns cached/yesterday snapshots; only the live site has
      the current-moment value.
    * Personalized / A-B / 'today's pick' feeds — "this site's row-1 trending
      now", "what the front-page recommendation panel shows this visit",
      "today's featured course / today's deal". Each visit can differ; the
      value visible right now is the fact.
    * Derived / cross-field computations the UI surfaces — "filter set to X
      AND Y, top 3 average price = ?", "cart of 3 items: subtotal + tax +
      shipping = final total". Search can find one cell, not the aggregate
      the UI just computed for you.
    * Hover / drag / zoom-revealed widget values — chart tooltips on hover,
      date-picker per-day price hints, map pins that only show their tooltip
      when hovered, timeline slider showing snapshot at the dragged point.
      Trigger the gesture, then read the revealed number.
    * Autocomplete / typeahead ordering — "type 'tok' into the search box,
      what are the top 3 suggestions and in what order". The list is
      query-dependent and server-personalized; search engines can't see it.
    * Checkout / cart / quote final summary — proceed to the last pre-pay
      step (without submitting) and capture tax, shipping, carrier ETA,
      final total. Pre-pay totals are session-only and never indexed.
    * Behavior-consequence facts — click a CTA and capture what HAPPENS
      next: "Try Free button leads to login wall vs paywall", "Cancel button
      shows refund within 24h vs 5 business days". The wording on the
      consequence page is the fact, not the button.
    * Image-embedded OCR — promo codes / deadlines / size charts / spec
      stamps / cast-credit rank baked into a banner or poster image. These
      are pixels, not text the crawler indexed.

- Each act should plausibly produce a NEW result-state worth recording. If
  the next act would only "see what controls exist" without changing the
  result-state, skip it.

`facts_learned` — CRITICAL. A fact is a RESULT-STATE value a real user would
ask about: the output of a search/filter/sort, the attributes of a Top-N
item, an aggregate count under specific constraints, a comparison across
items, a price/rating/availability/date carried by a SPECIFIC named entity,
OR a NAMED ENTITY visible inside an embedded image / hero / poster / album
art (artist name, athlete on a banner, designer credit in a screenshot,
landmark labeled inside a map tile, etc.). The image-embedded variant is
exactly the kind of fact that ONLY the live O-side can supply — searching
for the image's URL won't reveal the name inside the picture. Only record
facts that EXIST BECAUSE of the constraints you just applied or because
you're reading content INSIDE a specific image/figure.
- KEY is a short snake_case id describing the fact slot. Examples:
  "cheapest_nyc_lon_fare", "paris_jun9_top_hotel_name",
  "drama_adventure_count", "shot_designer_name", "course_monthly_cost",
  "courses_1to3_months_count",
  "hero_banner_artist_name", "featured_playlist_cover_athlete",
  "vsnt_intraday_change_now", "homepage_row1_recommendation_today",
  "lax_to_jfk_jun15_chart_tooltip_price",
  "tok_autocomplete_top3", "cart3items_final_total_with_tax",
  "try_free_cta_landing_page", "promo_banner_code_image".
- VALUE is the concrete observed value, written as COMPLETELY as the on-page
  evidence allows — pack every distinguishing attribute you can see (name,
  number, price, rating, count, duration, location, status, time stamp,
  rank, etc.) so the fact survives stand-alone, without the screenshot:
  - GOOD (complete, anchored):
    "Quality Inn Panguitch near Bryce Canyon — $77/night, 7.6 Good
     (1,977 reviews), 0.5 mi from town, free wifi/parking/breakfast,
     pet-friendly"
    "VSNT -2.46% intraday (40.44 USD at 14:32 EST), volume 1.2M, market
     cap $5.7B, P/E TTM 6.94"
    "Subtotal $147.20 + tax $11.78 + shipping $9.99 = total $168.97
     (2-day delivery, Mon Jun 16 by 8pm)"
    "Top 3 'tok' suggestions on Booking: Tokyo (Japan, 5,431 properties),
     Tokyo Disneyland Area (Chiba, 142), Tokyo Skytree Town (87)"
    "Try Free CTA → login wall requiring Google/Apple sign-in (no email
     option), 14-day trial advertised at the bottom"
  - BAD (sparse, requires screenshot to interpret):
    "Quality Inn Panguitch" (no price/rating/why-it-matters)
    "$77/night" (no name/location)
  Write 1-3 sentences when needed — completeness beats brevity. If a row
  has multiple useful columns (price + rating + reviews + amenities), list
  them all. The fact is going to be probed by a separate search agent that
  CANNOT see the page; every attribute you skip makes the probe harder.

The TRIGGER vs RESULT distinction — record the result, NOT the trigger:
- BAD (trigger / pre-state):       GOOD (result that the trigger produced):
- "checkin_date_selected =          "paris_jun9_jun11_top_hotel =
   Tue, Jun 9"                       Hôtel Indigo — €245/night, 4.2★"
- "guest_dropdown_default =         "paris_2adult_lowest_jun9 =
   Adults: 2, Rooms: 1"              Generator Paris — €98/night"
- "duration_filter_options =        "courses_1to3_months_under_$50 =
   <2h, 1-4w, 1-3m, ..."             1,284 (filtered from 8,443)"
- "calendar_months_displayed =      "trending_jun_dest_top1 =
   June 2026, July 2026"             Kansas City (no rank shown for #2-5)"
A fact whose value is the literal setting you JUST chose is almost always
a trigger, not a result — skip it.

BANNED fact types (they will be discarded downstream — do NOT write them):
- UI structure / chrome: element positions, button locations, menu or tab
  lists, search-bar existence, placeholder text, default form values, toggle
  states, dropdown enumerations WITHOUT data attached to a named entity.
  BAD: "filter_button_location", "main_nav_tabs", "search_placeholder",
       "pets_toggle_state = enabled", "assistance_animals_modal_type",
       "minimum_adult_count_allowed = 1".
- The filter/sort criterion you JUST set — record the result that came out,
  not the criterion itself. BAD: "sort_set_to = price_low_to_high".
- Calendar/picker/dropdown navigation state — record what the picker LED to,
  not what it shows. BAD: "calendar_months_displayed = June 2026, July 2026".
- CAPTCHA / anti-bot / 403 / paywall / access-denied / connection-refused
  diagnostics in any form. BAD examples (ALL banned, no matter how you
  rename them):
    "captcha_hold_10s_fails = ..."
    "<site>_blocked = press-hold wall"
    "tools_page_access_blocked = Authorization error SWAF-..."
    "insurance_calculator_403_reference_id = 0.bd82...x"
    "verification_wall_scope = site-wide block confirmed"
    "akamai_block_id = ...", "cloudflare_challenge_token = ...",
    "err_connection_closed", "get_quote_access_status = Access Denied"
  If blocked, put "try a different entry point / direct URL / archive
  mirror" in facts_to_pursue and move on — the block itself is NEVER a
  fact. Downstream filters drop these anyway; recording them only wastes
  explore budget and creates broken probe questions.
- Page-state narration. BAD: "page_loaded = successfully", "modal_appeared".
- NEVER end a key with: _location, _visible, _button, _coords, _position,
  _placeholder, _tabs, _menu, _layout, _selected, _default, _enabled,
  _displayed, _allowed, _state, _type.
- A filter list is only a fact when it carries DATA tied to a specific
  filtered slice:
  GOOD: "courses_1to3_months_count = 8,443"
        "courses_1to3_months_top1 = Google IT Support (4.8★, 47k enrolled)"
  BAD:  "duration_filter_options = Less Than 2 Hours, 1-4 Weeks, ..."

IMAGE-EMBEDDED facts (only the live site can supply these):
- DO record a named entity / number / text that appears INSIDE an image:
  artist face on a music banner, athlete on a sports hero shot, model name
  on a product render, designer credit baked into a screenshot, landmark
  labeled within a map tile, jersey number/team on a roster photo.
  GOOD: "hero_banner_artist = Taylor Swift (on the main rotator carousel)"
        "today_featured_player_name = Stephen Curry (#30, banner slot 1)"
        "store_window_designer_credit = Issey Miyake (caption inside frame)"
- DO NOT record generic visual mood / colors / textures as facts (those
  belong in visual_note, not facts_learned):
  BAD:  "banner_color = blue gradient with white sans-serif type"
        "carousel_aesthetic = minimalist flatlay"
- The KEY should name the slot the entity occupies; the VALUE should be the
  identifying string a user could later assert or look up.

Other rules:
- Add a NEW fact only when the observation produced novel info (not already in
  the Known Facts block at the top of the user message).
- 0-3 facts per step is normal. Don't pad. Don't record what was given.
- Self-check before emitting each fact: "Would a real user, before visiting
  this site, type this question into a search box?" If no — drop it.
- These facts accumulate across the run into the KB the next step sees, and
  are later probed one-by-one by a search agent — write each one so that
  "could a search engine find this?" is a meaningful question about it.

`facts_to_pursue` — what would be the next valuable question to answer?
- 0-2 short open questions. Use these to plan the next act or hint the next
  phase. Examples: "verify $828 fare directly on Norse Atlantic site",
  "find UK ETA requirement and cost", "compare 3rd-party reviews of Studio Moara".
- These get added to the Open Questions list visible to future steps.
- Drop ones you've now answered.

`visual_note` field — OPTIONAL, only when worth keeping:
- This is a one-sentence note about a CONCRETE VISUAL DETAIL on the current
  page that text alone (URL, alt-text, ARIA, OCR-like labels) would NOT
  expose. Anything that requires looking at the rendered pixels.
- Good examples (KEEP these):
  - "Top result shows a mustard-yellow accent chair against a slate-blue wall."
  - "The 3rd map pin sits inside a triangular plaza next to a curved river bend."
  - "Movie poster is dominated by red text on a black silhouette of a city skyline."
  - "Listing photo shows a Victorian-style facade with bay windows and turret."
  - "Product carousel image #2 is a top-down flatlay of 5 ceramic mugs."
- Bad examples (DO NOT write these):
  - "Page loaded successfully." (no visual detail)
  - "The button says 'Submit'." (text, not visual)
  - "There is a search bar at the top." (UI chrome, generic)
- Omit the field entirely (or set null) when nothing visually distinctive is
  worth recording this turn. Most turns will have no visual_note. Better to
  skip than write filler.
- These notes will later be used as anchors for the final hybrid task, so
  prefer notes that pin down a SPECIFIC item/scene the agent could relocate
  ("the chair in pin #3", "the poster in row 2"), not generic mood.

Guidelines:
- Be SHORT. Do not narrate plans for future turns.
- Focus on what the on-site UI uniquely supports (filters, configurators,
  multi-step forms, result panels).
- If the site is broken / captcha-walled, still emit one valid act (e.g.
  goto a fallback URL); a controller may switch you to summary mode soon.

No code fences. No prose around the JSON. Just one act.
"""


SYS_EXPLORE_STAR_TEMPLATE = """You are an exploration worker probing what
EXTERNAL information would complement the target site. Your only job this
turn is to take ONE concrete external-side action: a web search or page read.

You are NOT trying to solve a task. You are testing what kinds of facts are
available off-site that the target site itself cannot give a user. The phase
you're in lasts several turns — keep accumulating evidence around the same
external angle (visa rules, weather, third-party pricing, reputation, etc.).

Available tools and JSON schemas (this is the COMPLETE catalogue for this turn):
{tool_str}

Output EXACTLY ONE JSON object:
  {{"kind":"act",
    "thought":"<1-3 short lines: what external fact you are testing and why>",
    "tool_call":{{"name":"<tool>","arguments":{{...}},
                  "action_description":"<5-15 words>"}},
    "facts_learned":[
      {{"key":"<short id>","value":"<concrete external fact>"}}
    ],
    "facts_to_pursue":["<next question, one line>"]}}

`facts_learned` — MANDATORY when the previous step's tool returned content.
External-side facts are the whole reason the S phase exists; if you don't
extract them now, they're lost and the final task will not have hybrid
evidence.

WHEN you MUST emit ≥1 fact:
- The PREVIOUS step in history was a `search` call that returned ≥1 organic
  result → extract ≥1 concrete value (a price, a count, an airline name, a
  date, an official rule) from one of the top results. Don't just acknowledge
  "got results"; pull out a specific number or named entity.
- The previous step was a `visit` call AND the result_text does NOT contain
  "CAPTCHA" / "robot check" / "Are you a person" → extract ≥1 fact from
  the page summary (price, schedule, rule, reputation note, etc.).
- The previous step was a `scholar` call with ≥1 paper result → extract
  one concrete finding or stat.

WHEN you may emit 0 facts:
- The previous step was a `visit` that hit a CAPTCHA / 403 / robots wall —
  in that case emit ZERO facts and move on (try a different source / query
  next turn). Do NOT record the block itself: things like
  "<site>_blocked", "access_denied_reference_id", "verification_wall_scope",
  "press_hold_challenge", "403_reference", "akamai_block_id",
  "err_connection_closed" are NEVER valid facts. No user asks "what is the
  reference id on the 403 page"; downstream filters drop these and they
  only waste explore budget. If you're walled off from a source, write a
  follow-up in facts_to_pursue ("try a different scraper / direct URL /
  archive.org mirror") and continue.
- The current step is your FIRST act in the STAR phase (no tool result yet
  to extract from).

KEY naming — CRITICAL to avoid colliding with O-side facts:
- Prefix S-side keys with the source. Examples:
    "skyscanner_one_way_hkg_lon"   (S — Skyscanner showed this price)
    "expedia_round_trip_hkg_lon"   (S — Expedia showed this price)
    "govuk_eta_cost"               (S — official gov.uk page)
    "norse_destinations_list"      (S — Norse Atlantic site's route map)
- DO NOT reuse O-side key names like "cheapest_fare" or "selected_filter";
  those exist on the target site and your S-side value would overwrite them.

VALUE rules — extract a CONCRETE TOKEN, not a summary:
- Good: "$262 one-way / $577 round-trip, ZIPAIR cheapest"
- Good: "Norse Atlantic does NOT serve Hong Kong (Athens/London/NYC only)"
- Good: "average HKG-LHR fare $391 historically (faredetective)"
- Bad: "found search results about flights"  ← not a fact
- Bad: "Skyscanner has many options"  ← not specific

`facts_to_pursue` — 0-2 short open questions. These should chain off the
facts you just learned, OR off facts in the Known Facts block. If you're
at the END of a STAR phase, prioritize questions that the upcoming O-phase
could answer ON the target site.

Guidelines:
- Be SHORT.
- Aim for facts the target site cannot supply: pricing comparisons, official
  rules/regulations, weather, reviews, real-time advisories, third-party
  reputation.
- A bad STAR query is one that just looks up the target site itself
  ("site:google.com/flights ..."); prefer truly external sources.
- When the user message tells you "PHASE HANDOFF NOTE: ...", treat that as
  the highest-priority question to answer this phase.
- When the Known Facts block shows the "From external sources (S-side)"
  section is empty or has very few entries, your top priority THIS turn is
  to add an S-side fact, not to refine an O-side one.

No code fences. No prose around the JSON. Just one act.
"""


# (SYS_EXPLORE_OPS / SYS_EXPLORE_STAR are baked further below, after the
# tool-schema helper is defined.)


SYS_RECORD = """You are a task-summary worker. Exploration is finished.

Your PRIMARY input is the **Known Facts** block (KB) the explorer built up
across all phases — every concrete value, comparison, and identifier the
agent collected. The full action history is also provided for context, but
the KB is the ground truth for what the run actually established.

Your job: TIE the facts in the KB together into ONE coherent, logically-
flowing hybrid task that a real user might ask an AI agent to do. The
COMPLEXITY of the task should match the SIZE and DEPTH of the KB:
  - Few facts (≤3) and one source angle → simple lookup task
  - Several facts (4-8) spanning the site + one external source → 2-3 hop
  - Many facts (9+) crossing 2-3 sources → multi-constraint, multi-hop task
Don't pad — match difficulty to the evidence collected, but DO use as many
of the facts as plausibly fit a single user goal.

A *hybrid* task uses BOTH paths to be done well:
  S = leaving the site to query an external search engine / read external pages
  O = interacting with the target site via the browser
Pure S (text Q&A) or pure O (everything in-site) tasks are NOT hybrid.

start_mode rule (FIXED by the task itself, code-enforced downstream):
- task starts with a known URL on the target site → start_mode=O,
  switch_kind=O_needs_S_then_O   (start O, hop S for external fact, return O)
- task starts with no URL → start_mode=S, switch_kind=S_then_O
  (start S to discover the site, then switch O to operate)

route_plan rules (BINDING — violations are auto-rejected and force a retry):
- route_plan is a list of phases [{mode, subgoal, done_when}, ...]
- route_plan[0].mode MUST equal start_mode (no exception)
- AT LEAST 3 phases. Examples: O→S→O (for O-start), S→O→S (for S-start),
  or longer alternations like O→S→O→S→O.
- Phases MUST strictly alternate S↔O. Two consecutive phases with the same
  mode are rejected — fold them into a single phase if they belong together.
- DO NOT add explicit "switch" entries; the alternation IS the switch.
- Each subgoal is the concrete intent of that phase (what the agent will do).
- Each done_when is an observable exit condition for that phase.

Output EXACTLY ONE JSON object — EITHER `record_task` OR `done`. Choose
`done` when there is no genuine hybrid dependency in the KB (see below).

Option 1 (preferred when KB supports it):
  {"kind":"record_task",
   "task":{
     "question":"<see THE QUESTION FIELD rules below — intent only, no recipe>",
     "start_mode":"S"|"O",
     "switch_kind":"S_then_O"|"O_needs_S_then_O",
     "evidence":"<observations from this run that prove the task is achievable
                  AND genuinely hybrid (must mention BOTH the S contribution
                  AND the O contribution AND name the dependency)>",
     "image_required":<true|false>,
     "hybrid_dependency":{
       "type":"O_blocked_by_S"|"S_validates_O"|"S_disambiguates_O",
       "narrative":"<2-3 sentences naming SPECIFIC KB fact keys and explaining
                     why one phase's answer is REQUIRED for another. See rules below.>",
       "blocking_facts":["<S-side fact key from KB>", "..."],
       "dependent_facts":["<O-side fact key from KB>", "..."]
     },
     "route_plan":[
       {"mode":"S"|"O","subgoal":"...","done_when":"..."},
       {"mode":"S"|"O","subgoal":"...","done_when":"..."}
     ]
   }}

Option 2 (use when no genuine hybrid dependency exists in the KB):
  {"kind":"done","summary":"<one sentence: why this site's KB cannot support a
                              hybrid task (e.g., 'S phase produced no facts
                              that gate or validate O facts — the run found
                              only parallel info, not a chain').>"}

THE HYBRID DEPENDENCY FIELD — this is the gate between record_task and done.

A REAL hybrid task has ONE of these three dependency shapes:

  type="O_blocked_by_S"  — Without the S fact, the agent cannot DEFINE the
                            O-side filter or constraint. Example:
                            "Find a London flight after the UK ETA scheme
                            starts" — without the S fact (ETA launch date),
                            the O agent doesn't know what date range to filter.
                            blocking_facts: S keys that set the constraint
                            dependent_facts: O keys that USE the constraint

  type="S_validates_O"   — O extracts a value (price / rating / availability).
                            S confirms it against an authoritative external
                            source. Without S, the O answer is unverified.
                            blocking_facts: O key being checked
                            dependent_facts: S key holding the reference value

  type="S_disambiguates_O" — O produces MULTIPLE candidates. S provides the
                              tiebreaker info that picks one. Without S,
                              the agent cannot choose.
                              blocking_facts: O keys (list of candidates)
                              dependent_facts: S keys (tiebreaker criteria)

`blocking_facts` and `dependent_facts` MUST be ACTUAL KB fact keys you can
read in the "Known facts" block above. If you cannot point to specific keys
from BOTH sides of the KB that form one of the three shapes, emit
`kind:done` instead — do NOT invent a dependency to satisfy the schema.

EXAMPLES of valid hybrid_dependency:

  GOOD (O_blocked_by_S):
    {"type":"O_blocked_by_S",
     "narrative":"S fact `govuk_eta_launch_date` says ETAs become mandatory
                  on 2026-04-02. O step must filter for departures AFTER this
                  date (uses O fact `selected_departure_jul_13` to apply the
                  date filter). Without the S fact, the user has no basis to
                  pick a date constraint.",
     "blocking_facts":["govuk_eta_launch_date"],
     "dependent_facts":["selected_departure_jul_13","cheapest_hkg_lon_fare"]}

  GOOD (S_validates_O):
    {"type":"S_validates_O",
     "narrative":"O fact `cheapest_hkg_lon_fare` = HK$8,602 was shown on
                  Google Flights. S fact `skyscanner_hkg_lon_one_way` shows
                  a competing $262 price. The user's intent is to know whether
                  Google's number is competitive — answer requires comparing
                  the two facts.",
     "blocking_facts":["cheapest_hkg_lon_fare"],
     "dependent_facts":["skyscanner_hkg_lon_one_way"]}

BAD (parallel info, NOT a dependency — emit `kind:done` instead):
    "narrative": "Agent searches for flights AND verifies Done button
                   behavior."  ← these are independent tasks
    "narrative": "O finds the fare. S checks Reddit for general opinions."
                   ← S is decorative not gating
    "narrative": "Both O and S sides have information about pricing."
                   ← no specific dependency named

WHEN TO EMIT `kind:done` INSTEAD:
- KB has 0 S facts → emit done (cannot form hybrid_dependency without S keys).
- All S facts are "site_blocked" / "search_irrelevant" — they describe the
  S run's failure, not actual facts that gate O → emit done.
- O and S facts cover unrelated topics (O about prices, S about UI behavior).
  → emit done.
- You catch yourself writing "AND" to join O and S goals → emit done.

THE QUESTION FIELD — most important rule when you DO emit record_task. The `question` is what a REAL USER
would TYPE. It expresses INTENT and CONSTRAINTS, not a workflow.

`question` MUST:
- Read like a single natural sentence (or two) a person could plausibly
  send to an AI assistant — first-person ("I want...", "Can you find...",
  "Help me...") or imperative ("Find X with constraint Y").
- State the GOAL and the CONCRETE CONSTRAINTS (specific dates, prices,
  brands, locations, visual attributes pulled from the exploration).
- Stop there. The agent (downstream) figures out how to get the answer.

`question` MUST NOT:
- Prescribe steps. No "first do X, then do Y, then return to Z".
- Name S vs O paths. No "search externally for...", "navigate on the site
  to...", "switch to filter section", "open the configurator and pick...".
- Reference tools, filters, sections, buttons, or selectors. No "use the
  filter sidebar", "click the search button", "the date picker on the right".
- Mention "route_plan", "phase", "mode", or any of this framework's terms.
- Read like a checklist or numbered procedure.
- Use connector words that imply ordering between info sources: "then
  cross-reference", "after that look up", "before booking, check".

Allowed: a single specific website name in the question IF the task is
clearly about that site's offering (e.g. "I want to book a hotel in Paris
on Booking.com under €200/night that has a balcony"). But DON'T enumerate
how to use that site — just state the goal.

GOOD question examples (intent only):
- "Find a Dribbble shot of a pastel-pink mood-tracker app with a smiling
  face, and tell me the design's tags."
- "What's a fair NYC→LAX nonstop economy fare right now, and how does
  Google Maps' suggestion compare to other aggregators?"
- "Configure a Porsche 911 Carrera in Carmine Red with a dark multi-spoke
  20"/21" wheel — what should I realistically expect to pay above MSRP?"
- "I need a Behance branding project featuring a lime-green palette with a
  cute animal mascot — who designed it, and what's their Instagram reach?"

BAD question examples (steps/recipe — DO NOT WRITE THESE):
- "On Behance, search 'branding identity', filter by Branding and
  Freelance, then identify the lime-green Mushi Matcha project, then
  navigate to the creator's Instagram link and report the follower count."
- "First search Google Flights for NYC→LON, then verify the displayed
  fare against Skyscanner and Kayak, then return to Google Flights to
  finalize the selection."
- "Use the configurator to select Carmine Red and the third wheel option
  in the bottom row of the 20\\"/21\\" grid, then search external sources
  for dealer markup."

Other rules for `question`:
- Concrete (no "find a good X" — give specific constraints: cities, dates,
  prices, IDs, visual attributes). Pull the concrete values from observations.
- Tight and logically connected. Drop exploration moves that don't serve
  the final intent.
- If the exploration feels incomplete, do BEST EFFORT — record an intent
  the run supports enough to believe is achievable. A best-effort task is
  better than no task.

The `route_plan` field is where S/O paths and step sequencing belong. The
`question` and the `route_plan` are two DIFFERENT representations: the
question is the user's wish, the route_plan is the system's execution plan.
They should not look alike.

Visual anchors (from OPS-side `visual_note` fields in history):
- Some OPS turns recorded a `visual_note` describing a concrete visual detail
  on the page (color, layout, image content, scene composition). These notes
  are NOT exhaustive — only the steps where something visually distinctive
  was worth pinning down.
- When such notes exist, you MAY weave them into the task as visual anchors —
  constraints that the agent must SEE (not just read) to satisfy. Examples:
  "the mustard-yellow chair in the search results", "the listing whose
  exterior matches a Victorian style with bay windows".
- This makes the task genuinely image-grounded rather than pure text/ID lookup.
- Set `image_required:true` if the task's correctness depends on visual
  inspection of the page (the answer cannot be derived from URL/alt/text alone).
- Visual anchors are OPTIONAL; if the exploration produced no visual_note
  worth using, write a normal text-constrained task and set image_required:false.

No code fences. No prose around the JSON. Just one record_task with route_plan.
"""


SYS_RECORD_OR_DONE = """You are wrapping up. The run has already recorded at
least one hybrid task on this site. Decide ONE of two outcomes:

1. If the exploration uncovered a DISTINCTLY different second task category
   (not just a minor variation of what is already recorded), output another
   record_task with the SAME schema (including route_plan) as before:
     {"kind":"record_task","task":{
        "question":"...","start_mode":"S"|"O",
        "switch_kind":"S_then_O"|"O_needs_S_then_O",
        "evidence":"...","image_required":<true|false>,
        "route_plan":[{"mode":"S"|"O","subgoal":"...","done_when":"..."}, ...]}}

   route_plan rules (binding):
   - route_plan[0].mode == start_mode
   - At least 2 phases, at least 1 switch via alternating mode
   - No explicit "switch" entries

   question rules — CRITICAL:
   - The `question` is what a REAL USER would TYPE — intent + constraints,
     NOT a recipe. Do NOT prescribe steps, name S vs O paths, reference
     tools/filters/buttons/selectors, or use connectors like "then" /
     "after that" / "before X, check Y".
   - One natural sentence (first-person or imperative). Concrete (specific
     dates, prices, names, visual attributes). Stop there.
   - The route_plan is the workflow; the question is the wish. They should
     NOT look alike.

2. Otherwise, finish the run with a short summary:
     {"kind":"done","summary":"<one sentence: what was found on this site>"}

No code fences. No prose. Just one JSON object — record_task OR done.
"""


SYSTEM_PROMPT_BY_MODE = {
    "RECORD": SYS_RECORD,
    "RECORD_OR_DONE": SYS_RECORD_OR_DONE,
    # ACT_OPS / ACT_STAR are added later once their tool catalogues are baked.
}

def _side_for(mode: str) -> str:
    return "O" if mode == "ACT_OPS" else "S"


def _tool_names_for(mode: str) -> set[str]:
    """Allowed tool names for the given act mode (full hybrid_guided_routed side
    catalogue — NO filtering).  Used both to render the system prompt and to
    enforce the contract on LLM output.
    """
    side = _side_for(mode)
    if get_tools_for_mode is None:
        # Safe fallback (matches GUIDED_ROUTED_SIDE_TOOL_NAMES in hybrid_prompts.py)
        if side == "O":
            return {
                "goto", "new_tab", "click", "double_click", "hover", "fill",
                "select_option", "scroll", "drag", "wait", "go_back",
                "press_key", "switch_tab", "close_tab", "code_sandbox",
            }
        return {"search", "scholar", "visit", "code_sandbox"}
    return {t["function"]["name"] for t in
            get_tools_for_mode("hybrid_guided_routed", side, expose_switch=False)}


def _tool_schemas_for(mode: str) -> str:
    """Render the FULL hybrid_guided_routed tool catalogue for this side."""
    side = _side_for(mode)
    if get_tools_for_mode is None:
        return "(tool schemas unavailable — refer to hybrid_routed defaults)"
    tools = get_tools_for_mode("hybrid_guided_routed", side, expose_switch=False)
    return json.dumps(tools, ensure_ascii=False, indent=2)


# Build whitelists from the same source the prompts use so they cannot drift.
TOOL_WHITELIST_BY_MODE = {
    "ACT_OPS": _tool_names_for("ACT_OPS"),
    "ACT_STAR": _tool_names_for("ACT_STAR"),
}


# Bake the tool catalogue into the per-mode system prompts at import time.
SYS_EXPLORE_OPS = SYS_EXPLORE_OPS_TEMPLATE.format(tool_str=_tool_schemas_for("ACT_OPS"))
SYS_EXPLORE_STAR = SYS_EXPLORE_STAR_TEMPLATE.format(tool_str=_tool_schemas_for("ACT_STAR"))

# Now that they exist, register them under the mode map.
SYSTEM_PROMPT_BY_MODE.update({
    "ACT_OPS": SYS_EXPLORE_OPS,
    "ACT_STAR": SYS_EXPLORE_STAR,
})


JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def validate_route_plan(task: dict, kb: dict | None = None) -> tuple[bool, str]:
    """Check that route_plan + hybrid_dependency satisfy the contract.

    Structural checks (always):
      - route_plan is a list of >= 3 phases
      - each phase has mode ∈ {"S","O"} + non-empty subgoal + non-empty done_when
      - route_plan[0].mode == task.start_mode
      - phases STRICTLY alternate S↔O (no two consecutive same mode)

    KB-aware hybrid_dependency checks (when kb is provided):
      - hybrid_dependency object is present with valid `type`
      - blocking_facts / dependent_facts are non-empty lists of strings
      - every cited fact key MUST exist in kb.facts
      - the cited keys cover BOTH O and S sides (not all from one side)

    Returns (ok, reason).
    """
    plan = task.get("route_plan")
    start_mode = task.get("start_mode")
    if start_mode not in ("S", "O"):
        return False, f"start_mode must be S or O, got {start_mode!r}"
    if not isinstance(plan, list):
        return False, "route_plan must be a list"
    if len(plan) < 3:
        return False, f"route_plan must have >= 3 phases, got {len(plan)}"
    modes = []
    for i, p in enumerate(plan):
        if not isinstance(p, dict):
            return False, f"route_plan[{i}] is not an object"
        m = p.get("mode")
        if m not in ("S", "O"):
            return False, f"route_plan[{i}].mode must be S or O, got {m!r}"
        sg = (p.get("subgoal") or "").strip()
        dw = (p.get("done_when") or "").strip()
        if not sg:
            return False, f"route_plan[{i}].subgoal is empty"
        if not dw:
            return False, f"route_plan[{i}].done_when is empty"
        modes.append(m)
    if modes[0] != start_mode:
        return False, (
            f"route_plan[0].mode ({modes[0]!r}) must equal start_mode "
            f"({start_mode!r})"
        )
    # Strict alternation: no two consecutive phases may share mode.
    for i in range(1, len(modes)):
        if modes[i] == modes[i - 1]:
            return False, (
                f"route_plan phases must alternate S↔O; "
                f"phase {i-1} and {i} are both {modes[i]!r}"
            )

    # ---------- KB-aware hybrid_dependency checks ----------
    if kb is None:
        return True, ""
    facts = (kb or {}).get("facts") or {}
    dep = task.get("hybrid_dependency")
    if not isinstance(dep, dict):
        return False, "hybrid_dependency missing or not an object"
    dtype = dep.get("type")
    if dtype not in ("O_blocked_by_S", "S_validates_O", "S_disambiguates_O"):
        return False, (
            f"hybrid_dependency.type must be one of "
            "O_blocked_by_S | S_validates_O | S_disambiguates_O"
            f"; got {dtype!r}"
        )
    narrative = (dep.get("narrative") or "").strip()
    if len(narrative) < 40:
        return False, "hybrid_dependency.narrative too short (need 2-3 sentences)"
    blocking = dep.get("blocking_facts") or []
    dependent = dep.get("dependent_facts") or []
    if not isinstance(blocking, list) or not blocking:
        return False, "hybrid_dependency.blocking_facts must be a non-empty list"
    if not isinstance(dependent, list) or not dependent:
        return False, "hybrid_dependency.dependent_facts must be a non-empty list"
    # Every cited key must exist in KB.
    missing = [k for k in (blocking + dependent)
               if not isinstance(k, str) or k not in facts]
    if missing:
        return False, (
            "hybrid_dependency cites fact key(s) not in KB: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    # Coverage: blocking + dependent must span BOTH O and S sides
    # (otherwise it's a one-side chain, not hybrid).
    sides = {facts[k].get("mode") for k in (blocking + dependent)}
    if not {"O", "S"}.issubset(sides):
        return False, (
            "hybrid_dependency must cite facts from BOTH O and S sides; "
            f"sides covered: {sides}"
        )
    return True, ""


# ====================== ExplorerRouter ======================
class ExplorerRouter:
    """Deterministic state machine that decides the worker's mode each turn.

    Modes:
      ACT_OPS          — worker must emit {"kind":"act", ...} using a BROWSER tool only
      ACT_STAR         — worker must emit {"kind":"act", ...} using SEARCH/VISIT only
      RECORD           — worker MUST emit {"kind":"record_task", ...}
      RECORD_OR_DONE   — worker emits either record_task or done
      STOP             — loop terminates immediately

    The OPS/STAR split ensures both S-side and O-side history exist BEFORE the
    first RECORD turn so the LLM has concrete evidence for both paths.
    """
    ACT_OPS = "ACT_OPS"
    ACT_STAR = "ACT_STAR"
    RECORD = "RECORD"
    RECORD_OR_DONE = "RECORD_OR_DONE"
    STOP = "STOP"

    def __init__(self, *, max_steps: int = 65,
                 max_phase_steps: int = 20,
                 min_phases_before_record: int = 3,
                 start_mode: str = "O",
                 phase_budgets: dict | None = None,
                 explore_only: bool = False):
        """
        Phase-based exploration. The act budget is sliced into PHASES,
        alternating OPS ↔ STAR. After ≥ N phases the router emits ONE
        terminal RECORD; the run then ends.

        Args:
          max_steps:               total LLM turn budget (incl. the RECORD turn)
          max_phase_steps:         default turns per phase before forced flip
                                   (used when phase_budgets is None or missing entry)
          phase_budgets:           override per-mode budget, e.g. {"O": 10, "S": 5}.
                                   Falls back to max_phase_steps when key missing.
          min_phases_before_record:require at least N phases before terminal RECORD
          start_mode:              "O" → first phase is OPS; "S" → first is STAR
        """
        self.max_steps = max_steps
        self.max_phase_steps = max_phase_steps
        self.phase_budgets = dict(phase_budgets or {})
        self.min_phases_before_record = min_phases_before_record
        self.start_mode = start_mode
        # explore_only: single OPS phase, no STAR/RECORD. Run until budget or
        # saturation, then STOP. Used by the v7 tag-based pipeline (Stage A).
        self.explore_only = explore_only
        # ---- dynamic-budget knobs ----
        # Minimum steps a phase MUST serve before we consider early-flip.
        self.min_phase_steps = 3
        # If the last N successful acts produced 0 new facts, treat phase as
        # saturated and flip early (after min_phase_steps).
        self.dry_streak_threshold = 3
        # If 1 step produces ≥ extend_threshold facts, grant +extend_bonus
        # extra budget to this phase (caps at 2× base).
        self.extend_threshold = 2
        self.extend_bonus = 2
        # Track facts present at start of each step so we can detect dry runs.
        self._facts_at_step_start: int = 0
        self._dry_streak: int = 0
        # Per-phase budget extensions earned via richness signals.
        self._budget_extension: int = 0

        # Counters
        self.steps_taken = 0           # total LLM turns consumed
        self.acts_taken = 0
        self.ops_acts_taken = 0
        self.star_acts_taken = 0
        self.records_taken = 0
        # Phase tracking
        self.cur_phase: str = self.ACT_OPS if start_mode == "O" else self.ACT_STAR
        self.phase_steps_used = 0      # LLM turns served in current phase
        self.phase_index = 0           # how many phases have been STARTED (>=1 once first served)
        # Force-counter: # of consecutive RECORD attempts the LLM has flubbed
        self.consec_force_records = 0
        self.step_at_last_record: int = -1
        # ---- knowledge base (accumulates across the whole run) ----
        # facts[key] = {"value": str, "step": int, "phase_idx": int, "mode": str}
        # Later facts with the same key overwrite earlier ones (the LLM
        # may refine a value as exploration progresses).
        self.facts: dict[str, dict] = {}
        # open_questions is an ordered list of strings (dedup-keep-first).
        self.open_questions: list[str] = []
        # phase_findings[i] = brief 1-line summary written at phase boundary
        self.phase_findings: list[dict] = []
        # The next user message will inject this handoff string at the top
        # when set, then it's cleared.
        self.pending_handoff: str | None = None

    # ---------------------- KB helpers ----------------------
    @staticmethod
    def _norm_key(k: str) -> str:
        return "".join(ch.lower() if ch.isalnum() else "_" for ch in (k or "").strip())[:60]

    def absorb_facts(self, mode: str, raw_facts) -> None:
        """Merge facts_learned from the LLM into the KB.

        Key-collision policy: never silently overwrite. Identical value →
        dedupe (no-op). Different value → store under `<key>_v2`, `_v3`, ...
        so a multi-phase run keeps every distinct observation (the old
        overwrite behavior silently compressed ~2/3 of long-run facts).
        """
        if not isinstance(raw_facts, list):
            return
        side = "O" if mode == self.ACT_OPS else "S"
        for f in raw_facts:
            if not isinstance(f, dict):
                continue
            key = self._norm_key(f.get("key") or "")
            value = (f.get("value") or "").strip()
            if not key or not value:
                continue
            value = value[:300]  # keep concise
            if key in self.facts:
                if self.facts[key]["value"] == value:
                    continue  # exact duplicate — skip
                n = 2
                while f"{key}_v{n}" in self.facts:
                    if self.facts[f"{key}_v{n}"]["value"] == value:
                        break
                    n += 1
                else:
                    pass
                key = f"{key}_v{n}"
                if key in self.facts:  # matched an existing _vN duplicate
                    continue
            self.facts[key] = {
                "value": value,
                "step": self.steps_taken,
                "phase_idx": self.phase_index,
                "mode": side,
            }

    def update_facts(self, raw_updates) -> int:
        """Replace the value of EXISTING fact entries end-to-end (v8 extractor
        can decide a previously written fact has grown into a more complete
        version and emit it via `facts_updated` instead of `facts_learned`).

        Returns count of updates actually applied. Misses (unknown key, key
        norm collision with a `_vN` variant, etc.) are logged but ignored —
        they degrade gracefully to "no-op", never raise.
        """
        if not isinstance(raw_updates, list):
            return 0
        applied = 0
        for u in raw_updates:
            if not isinstance(u, dict):
                continue
            raw_key = (u.get("key") or "").strip()
            key = self._norm_key(raw_key)
            value = (u.get("value") or "").strip()
            if not key or not value:
                continue
            value = value[:300]
            if key not in self.facts:
                # Lookup tolerance: extractor may have referenced a `_vN`
                # variant or the un-normalized form. Try those too.
                cand = None
                for k in self.facts:
                    if k == raw_key or self._norm_key(k) == key:
                        cand = k
                        break
                if cand is None:
                    continue
                key = cand
            self.facts[key]["value"] = value
            # carry forward the updated step so dynamic signals see fresh activity
            self.facts[key]["step"] = self.steps_taken
            applied += 1
        return applied

    def absorb_questions(self, raw_qs) -> None:
        """Merge facts_to_pursue into open_questions, dedup-keep-first."""
        if not isinstance(raw_qs, list):
            return
        seen = {q.strip().lower() for q in self.open_questions}
        for q in raw_qs:
            if not isinstance(q, str):
                continue
            q = q.strip()[:240]
            if not q or q.lower() in seen:
                continue
            self.open_questions.append(q)
            seen.add(q.lower())
        # Cap open_questions so the KB block doesn't explode.
        if len(self.open_questions) > 12:
            self.open_questions = self.open_questions[-12:]

    def retire_answered_questions(self, raw_facts) -> None:
        """If a new fact's key obviously matches an open question, drop the
        question. Heuristic: substring match of fact key tokens in question."""
        if not isinstance(raw_facts, list):
            return
        tokens = set()
        for f in raw_facts:
            if isinstance(f, dict):
                tokens.update(self._norm_key(f.get("key") or "").split("_"))
        tokens.discard("")
        if not tokens:
            return
        kept = []
        for q in self.open_questions:
            ql = q.lower()
            if any(t in ql for t in tokens if len(t) >= 4):
                continue
            kept.append(q)
        self.open_questions = kept

    def update_dynamic_signals(self, facts_added_this_step: int) -> None:
        """Called by the main loop after a successful act, BEFORE notify_turn.
        Adjusts dry_streak (saturation) and budget_extension (richness)."""
        if facts_added_this_step <= 0:
            self._dry_streak += 1
        else:
            self._dry_streak = 0
            if facts_added_this_step >= self.extend_threshold:
                # Richness! Grant this phase a budget bonus.
                self._budget_extension += self.extend_bonus

    def kb_snapshot(self) -> dict:
        return {
            "facts": dict(self.facts),
            "open_questions": list(self.open_questions),
            "phase_findings": list(self.phase_findings),
            "phase_index": self.phase_index,
            "dynamic": {
                "dry_streak": self._dry_streak,
                "budget_extension": self._budget_extension,
                "phase_steps_used": self.phase_steps_used,
            },
        }

    def _flip_phase(self):
        prev_phase = self.cur_phase
        prev_idx = self.phase_index
        # Record a handoff hint for the next phase: pick the most recent
        # open question (LLM's freshest "what to chase").
        if self.open_questions:
            self.pending_handoff = self.open_questions[-1]
        # Record a phase finding (light snapshot — RECORD reads these too).
        # We summarize as "phase i (mode): facts added since last flip".
        added_keys = [
            k for k, v in self.facts.items()
            if v["phase_idx"] == prev_idx
        ]
        self.phase_findings.append({
            "phase_idx": prev_idx,
            "mode": "O" if prev_phase == self.ACT_OPS else "S",
            "facts_added": added_keys,
            "handoff": self.pending_handoff or "",
        })
        self.cur_phase = (
            self.ACT_STAR if self.cur_phase == self.ACT_OPS else self.ACT_OPS
        )
        # Reset dynamic-budget tracking on phase boundary
        self._dry_streak = 0
        self._budget_extension = 0
        self.phase_steps_used = 0
        self.phase_index += 1

    def decide(self) -> str:
        # Already produced the terminal record → stop.
        if self.records_taken >= 1:
            return self.STOP
        # Hard stop after budget
        if self.steps_taken >= self.max_steps:
            return self.STOP
        # Repeated record violations → STOP to avoid infinite force loop
        if self.consec_force_records >= 3:
            return self.STOP

        # ---- explore-only mode (v7 Stage A): one OPS phase, stop on budget
        # or saturation. No STAR, no RECORD — facts are tagged offline.
        # Early steps are form-setup (navigate, dismiss modals, fill
        # origin/destination) and legitimately yield no facts, so:
        #   (a) a GRACE window never triggers a stop, and
        #   (b) saturation only fires once at least one fact has landed
        #       (a site that never yields a fact runs to budget — unless it
        #        is genuinely walled, which CAPTCHA-type facts would show).
        if self.explore_only:
            import os as _os
            GRACE = int(_os.environ.get("EXPLORE_GRACE", "6"))
            # v8's decoupled extractor returns empty more often; allow EXPLORE_DRY
            # env to override the saturation streak threshold without touching v7.
            DRY = int(_os.environ.get("EXPLORE_DRY", "4"))
            BARREN = int(_os.environ.get("EXPLORE_BARREN", "10"))
            if self.phase_steps_used >= BARREN and not self.facts:
                return self.STOP
            if (self.phase_steps_used >= GRACE
                    and len(self.facts) > 0
                    and self._dry_streak >= DRY):
                return self.STOP
            if self.phase_index == 0:
                self.phase_index = 1
            return self.ACT_OPS

        # End-of-budget terminal RECORD: reserve the last LLM turn for RECORD,
        # but ONLY after we've served at least `min_phases_before_record`
        # alternating phases.
        steps_left = self.max_steps - self.steps_taken
        ready_for_record = (
            self.phase_index >= self.min_phases_before_record  # 0-indexed phases STARTED
            and self.ops_acts_taken >= 1
            and self.star_acts_taken >= 1
        )
        if steps_left <= 1 or (ready_for_record and steps_left <= 2):
            return self.RECORD

        # Budget per phase is mode-specific via phase_budgets
        # (e.g. {"O":10,"S":5}); falls back to max_phase_steps if not specified.
        # We then apply dynamic adjustments: rich-vein extension and saturation
        # early-flip.
        cur_side = "O" if self.cur_phase == self.ACT_OPS else "S"
        base_budget = int(self.phase_budgets.get(cur_side, self.max_phase_steps))
        # Effective budget = base + extensions earned this phase, capped at 2x.
        eff_budget = min(base_budget * 2, base_budget + self._budget_extension)

        # DYNAMIC SIGNAL 1 — saturation: phase served at least min steps AND
        # last N steps produced 0 new facts → flip early.
        saturated = (
            self.phase_steps_used >= self.min_phase_steps
            and self._dry_streak >= self.dry_streak_threshold
        )

        if self.phase_steps_used >= eff_budget or saturated:
            # When min-phase floor is met AND KB has cross-side coverage that
            # makes a hybrid_dependency plausible, end early with RECORD instead
            # of extending into yet another phase.
            if ready_for_record and self._kb_has_cross_side_coverage():
                return self.RECORD
            # Otherwise, if min_phases not met OR KB still one-sided, flip
            # to the other side and keep accumulating.
            if ready_for_record and not self._kb_has_cross_side_coverage():
                # KB lopsided. Force more time on the under-represented side.
                pass  # handled by flip below — fresh phase on the other side
            self._flip_phase()

        # First call: phase_index counts the active phase (initial state is 0,
        # not yet started → bump to 1 on first decide so we report correctly).
        if self.phase_index == 0:
            self.phase_index = 1
        return self.cur_phase

    def _kb_has_cross_side_coverage(self) -> bool:
        """True if facts contain at least one O-side AND one S-side fact."""
        seen_o = seen_s = False
        for v in self.facts.values():
            m = v.get("mode")
            if m == "O":
                seen_o = True
            elif m == "S":
                seen_s = True
            if seen_o and seen_s:
                return True
        return False

    def notify_turn(self, mode: str, success: bool, kind: str):
        """Called after each turn, BEFORE the next decide()."""
        self.steps_taken += 1
        if kind == "act":
            # Within a phase, every served act counts toward phase_steps_used,
            # even a failed one (gateway error / bad tool / etc.) — otherwise
            # a stuck LLM could burn the whole budget on one phase.
            if mode in (self.ACT_OPS, self.ACT_STAR):
                self.phase_steps_used += 1
            if success:
                self.acts_taken += 1
                if mode == self.ACT_OPS:
                    self.ops_acts_taken += 1
                elif mode == self.ACT_STAR:
                    self.star_acts_taken += 1
            self.consec_force_records = 0
        elif kind == "record_task":
            if success:
                self.records_taken += 1
                self.step_at_last_record = self.steps_taken
            self.consec_force_records = 0
        elif kind == "done":
            self.consec_force_records = 0
        else:
            # worker disobeyed
            if mode == self.RECORD:
                self.consec_force_records += 1
            else:
                self.consec_force_records = 0
                # Treat as a consumed phase step so we don't loop forever
                if mode in (self.ACT_OPS, self.ACT_STAR):
                    self.phase_steps_used += 1


# ====================== prompt fragments ======================
# (Per-mode INSTRUCTIONS blocks are no longer used; SYSTEM_PROMPT_BY_MODE above
# carries the full system prompt for each mode independently.)


def _data_url_from_png(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _truncate(s, n=2000):
    s = s or ""
    return s if len(s) <= n else s[: n - 1] + "…"


class ExplorerAgent:
    def __init__(self, *, site: dict, runtime, max_steps: int = 65,
                 use_vision: bool = True, max_history: int = 8,
                 max_phase_steps: int = 20,
                 min_phases_before_record: int = 3,
                 phase_budgets: dict | None = None,
                 explore_only: bool = False):
        self.site = site
        self.runtime = runtime
        self.max_steps = max_steps
        self.use_vision = use_vision
        self.max_history = max_history
        self.history: list[dict] = []  # rolling log of turns
        self.candidate_tasks: list[dict] = []
        self.notes: list[dict] = []     # raw model responses for debug
        # start_mode is fixed by the site row: has entry_url → O-start, else S
        site_start_mode = "O" if (site.get("entry_url") or "").strip() else "S"
        self.router = ExplorerRouter(
            max_steps=max_steps,
            max_phase_steps=max_phase_steps,
            min_phases_before_record=min_phases_before_record,
            start_mode=site_start_mode,
            phase_budgets=phase_budgets,
            explore_only=explore_only,
        )

    # ---------------- prompt building -----------------
    def _build_messages(self, state: dict, mode: str) -> list[dict]:
        site = self.site
        # System message: per-mode independent prompt (no cross-mode talk)
        system_text = SYSTEM_PROMPT_BY_MODE.get(mode, SYS_EXPLORE_OPS)
        # MODEL-AWARE coordinate hint — kimi-k2.x only. Its training mixes
        # 0-1 floats and 0-1000 ints; without explicit anchoring it flips
        # between them mid-run. Other VLMs (Claude / Qwen-VL) use their
        # native convention via coord_scale.
        _model_lower = MODEL.lower()
        if "kimi" in _model_lower:
            system_text = system_text + (
                "\n\nCOORDINATE CONVENTION — kimi-k2 family only:\n"
                "Output (x, y) as relative floats in [0.0, 1.0]: (0,0) is the "
                "top-left corner of the viewport, (1,1) is the bottom-right. "
                "Example: x=0.336, y=0.601 means 33.6% across, 60.1% down. "
                "Use 2-3 decimal places. Do NOT output integer pixel values."
            )
        msgs = [{"role": "system", "content": system_text}]

        site_frame = (
            f"Target site: {site.get('domain','?')}  ({site.get('category','?')})\n"
            f"Entry URL: {site.get('entry_url','')}\n"
            f"Switch kind hint from offline probe: {site.get('switch_kind','?')}\n"
            f"Probe note: { _truncate(site.get('switch_trigger',''), 400) }\n"
        )

        # ACT_OPS / ACT_STAR: truncated recent history (worker needs context,
        #                     not full log)
        # RECORD / RECORD_OR_DONE: FULL history (worker must summarize the
        #                          whole exploration end-to-end)
        if mode in (ExplorerRouter.ACT_OPS, ExplorerRouter.ACT_STAR):
            recent = self.history[-self.max_history :]
        else:
            recent = self.history
        recent_lines = []
        visual_index: list[tuple[int, str]] = []  # (step, visual_note)
        for h in recent:
            resp = h.get("response_json", {}) or {}
            cs = json.dumps(resp, ensure_ascii=False)
            obs_d = h.get("observation", {})
            obs = json.dumps(obs_d, ensure_ascii=False)
            vnote = (resp.get("visual_note") or "").strip()
            extra = f"\nVISUAL: {_truncate(vnote, 300)}" if vnote else ""
            if vnote:
                visual_index.append((h["step"], vnote))
            # Surface STICKY OVERLAY warning: when a click hits a fixed/sticky
            # ancestor (cookie banner, nav header, modal), the click did NOT
            # reach the underlying target. The model otherwise sees only
            # `tag: 'nav'` in the diagnostics and may keep retrying same coord.
            overlay_warn = ""
            if isinstance(obs_d, dict):
                ts = ((obs_d.get("diagnostics") or {}).get("target_state") or {})
                so = ts.get("sticky_overlay") if isinstance(ts, dict) else None
                if so and isinstance(so, dict):
                    overlay_warn = (
                        f"\n⚠ CLICK HIT A {so.get('position','sticky')} OVERLAY "
                        f"<{so.get('tag','?')}> "
                        f"({so.get('width','?')}×{so.get('height','?')}px). "
                        f"The actual target underneath was NOT clicked. "
                        f"Scroll the overlay out of the way, or `press_key` "
                        f"PageDown / End to push the target below the overlay, "
                        f"or click somewhere away from this overlay first."
                    )
                # Surface the actual text/aria_label of what was clicked, so
                # the model can recognise wrong targets.
                hit_text = ts.get("text") if isinstance(ts, dict) else None
                hit_aria = ts.get("aria_label") if isinstance(ts, dict) else None
                if hit_text or hit_aria:
                    label = hit_text or hit_aria
                    overlay_warn += f"\nYOU ACTUALLY HIT: '{label[:60]}'"
            recent_lines.append(
                f"--- step {h['step']} ---\n"
                f"YOU OUTPUT: {_truncate(cs, 600)}\n"
                f"RESULT: {_truncate(obs, 500)}{extra}{overlay_warn}"
            )
        recent_blob = "\n\n".join(recent_lines) if recent_lines else "(no prior turns)"

        # --- REPEAT DETECTOR ---
        # Scan FULL history (not just the visible window) for identical
        # action_descriptions that the operator keeps emitting. Surface them
        # so the model can't ignore the loop just because the window cut off
        # the earlier attempts.
        repeat_lines = []
        if mode in (ExplorerRouter.ACT_OPS, ExplorerRouter.ACT_STAR):
            from collections import Counter as _Ctr
            full_acts = []
            for h in self.history:
                rj = h.get("response_json") or {}
                tc = rj.get("tool_call") or {}
                ad = (tc.get("action_description") or "").strip()
                nm = tc.get("name", "") or ""
                if ad:
                    full_acts.append((nm, ad))
            counts = _Ctr(full_acts)
            repeats = [(nm, ad, c) for (nm, ad), c in counts.items() if c >= 3]
            if repeats:
                repeats.sort(key=lambda x: -x[2])
                for nm, ad, c in repeats[:5]:
                    repeat_lines.append(f"  - {c}×  {nm}: {_truncate(ad, 100)}")
            # Also check: same URL stuck for last N >= 6 turns?
            recent_urls = []
            for h in self.history[-10:]:
                obs = h.get("observation") or {}
                u = obs.get("ended_url") if isinstance(obs, dict) else ""
                if u:
                    recent_urls.append(u)
            run = 1
            longest_url = recent_urls[-1] if recent_urls else ""
            for i in range(len(recent_urls) - 2, -1, -1):
                if recent_urls[i] == recent_urls[-1]:
                    run += 1
                else:
                    break
            if run >= 6 and longest_url:
                repeat_lines.append(
                    f"  - LAST {run} TURNS all on same URL: {_truncate(longest_url, 80)} "
                    f"— this page does not progress your goal; navigate AWAY."
                )
        repeats_block = ("\n".join(repeat_lines) if repeat_lines
                         else "  (none — you are progressing)")

        if visual_index:
            visual_blob = "\n".join(
                f"  step {s}: {_truncate(n, 240)}" for s, n in visual_index
            )
        else:
            visual_blob = "  (none recorded — task should rely on text/structured signals)"

        cur_url = state.get("current_url", "")
        cur_title = state.get("current_title", "")
        state_text = f"Current URL: {cur_url}\nCurrent Title: {cur_title}"

        rec_block = "\n".join(
            f"- recorded {i + 1}: {t.get('question','?')[:240]}"
            for i, t in enumerate(self.candidate_tasks)
        ) or "(none yet)"

        # -------- KB / handoff blocks (shared across modes) --------
        kb = self.router.kb_snapshot() if hasattr(self, "router") else {"facts":{},"open_questions":[],"phase_findings":[]}
        # Split facts by mode so the model can see balance — an empty S-side
        # section is a strong visual cue that the next STAR phase should add
        # external facts.
        o_facts = [(k, v) for k, v in kb["facts"].items() if v.get("mode") == "O"]
        s_facts = [(k, v) for k, v in kb["facts"].items() if v.get("mode") == "S"]
        def _fmt_facts(rows):
            return "\n".join(
                f"  - {k} [step {v['step']}]: {v['value']}" for k, v in rows
            )
        o_block = _fmt_facts(o_facts) if o_facts else "  (none yet)"
        s_block = _fmt_facts(s_facts) if s_facts else "  (none yet — S phases must add external facts here)"
        kb_facts_block = (
            f"### From the target site (O-side, {len(o_facts)} facts):\n{o_block}\n"
            f"### From external sources (S-side, {len(s_facts)} facts):\n{s_block}"
        )
        if kb["open_questions"]:
            open_q_block = "\n".join(f"  - {q}" for q in kb["open_questions"])
        else:
            open_q_block = "  (none — propose new ones in facts_to_pursue)"
        if kb["phase_findings"]:
            pf_block = "\n".join(
                f"  - phase {pf['phase_idx']} ({pf['mode']}-side): added "
                f"{len(pf['facts_added'])} facts; handoff to next phase: "
                f"\"{(pf['handoff'] or '(none)')[:120]}\""
                for pf in kb["phase_findings"]
            )
        else:
            pf_block = "  (no completed phases yet — this is the first phase)"

        # Pull and clear pending phase handoff so it's shown exactly once.
        handoff_line = ""
        if hasattr(self, "router") and self.router.pending_handoff:
            handoff_line = (
                f"## PHASE HANDOFF NOTE — top priority this phase\n"
                f"  -> {self.router.pending_handoff}\n\n"
            )
            self.router.pending_handoff = None

        # Build user content differently per mode:
        if mode in (ExplorerRouter.ACT_OPS, ExplorerRouter.ACT_STAR):
            # Worker is acting — feed it state + history. Vision (screenshot)
            # only matters for OPS turns where x/y coordinates depend on layout.
            text_content = (
                f"{site_frame}\n"
                f"{handoff_line}"
                f"## Known facts (KB built up across previous steps)\n{kb_facts_block}\n\n"
                f"## Open questions (chase one of these or propose new)\n{open_q_block}\n\n"
                f"## Phase log\n{pf_block}\n\n"
                f"## ⚠ REPEATED ACTIONS — IF YOU SEE YOUR ACTION HERE, STOP DOING IT\n"
                f"{repeats_block}\n"
                f"Anything appearing 3+ times has NOT worked. The same click/scroll "
                f"WILL keep failing. Switch strategies: try a different element, "
                f"`goto` a deep URL, `press_key` Escape, navigate to a new section.\n\n"
                f"## Recent turns\n{recent_blob}\n\n"
                f"## Current state\n{state_text}\n\n"
                f"Take the next action. Record any concrete new facts in `facts_learned`."
            )
            content_parts: list[dict] = [{"type": "text", "text": text_content}]
            if mode == ExplorerRouter.ACT_OPS:
                img_path = state.get("img_path") or ""
                if self.use_vision and img_path and os.path.exists(img_path):
                    try:
                        data_url = _data_url_from_png(img_path)
                        content_parts.append(
                            {"type": "image_url", "image_url": {"url": data_url}}
                        )
                    except Exception:
                        pass
            msgs.append({"role": "user", "content": content_parts})
            return msgs

        # RECORD / RECORD_OR_DONE — summary mode: NO screenshot.
        # PRIMARY input is the KB block (facts collected). History is kept
        # for context but the LLM should anchor the task on the KB.
        text_content = (
            f"{site_frame}\n"
            f"## Known facts (PRIMARY input — anchor the task on these)\n{kb_facts_block}\n\n"
            f"## Open questions (still unanswered at end of run)\n{open_q_block}\n\n"
            f"## Phase log (alternation pattern + per-phase contribution)\n{pf_block}\n\n"
            f"## Visual notes index (OPS-side observations worth pinning)\n{visual_blob}\n\n"
            f"## Full run history (context only — KB above is the source of truth)\n{recent_blob}\n\n"
            f"## Last seen page\n{state_text}\n\n"
            f"## Tasks already recorded this run\n{rec_block}\n\n"
            f"Produce the JSON now. Match task complexity to KB size."
        )
        msgs.append({"role": "user", "content": text_content})
        return msgs

    # ---------------- model call ----------------------
    async def _call_llm(self, messages: list[dict], retries: int = 4,
                        mode: str | None = None) -> str:
        # Route RECORD / RECORD_OR_DONE turns through the stronger model on
        # whatever endpoint RECORD_API_BASE points at; everything else uses
        # the explore endpoint (kimi-k2.6 / qwen3.7-plus / etc.).
        is_record = mode in ("RECORD", "RECORD_OR_DONE")
        base = RECORD_API_BASE if is_record else API_BASE
        key  = RECORD_API_KEY  if is_record else API_KEY
        model = RECORD_MODEL    if is_record else MODEL
        body = {
            "model": model,
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 16000,
        }
        last_err = None
        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(timeout=LLM_TIMEOUT_S) as client:
                    r = await client.post(
                        f"{base}/chat/completions",
                        json=body,
                        headers={"Authorization": f"Bearer {key}",
                                 "Content-Type": "application/json"},
                    )
                    # 418/4xx (except 429) are fatal — retrying makes things
                    # worse (gateway rate-limit / anti-abuse). Only retry on
                    # 5xx and 429.
                    if r.status_code == 200:
                        data = r.json()
                        return data["choices"][0]["message"]["content"]
                    if r.status_code in (429,) or 500 <= r.status_code < 600:
                        last_err = httpx.HTTPStatusError(
                            f"{r.status_code}", request=r.request, response=r)
                    else:
                        # 418, 401, 403, etc. — fatal; no retry.
                        raise httpx.HTTPStatusError(
                            f"fatal {r.status_code}: {r.text[:200]}",
                            request=r.request, response=r)
            except httpx.HTTPStatusError as exc:
                if 500 <= getattr(exc.response, "status_code", 0) < 600 or exc.response.status_code == 429:
                    last_err = exc
                else:
                    raise
            except Exception as exc:
                last_err = exc
            # exponential backoff with floor — give the gateway time to recover
            await asyncio.sleep(min(30.0, 5.0 * (2 ** attempt)))
        raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")

    @staticmethod
    def _parse_response(raw: str) -> dict | None:
        if not raw:
            return None
        s = raw.strip()
        # strip code fences
        if s.startswith("```"):
            s = s.strip("`")
            if s.lower().startswith("json"):
                s = s[4:]
            s = s.strip()
        # try direct parse, then regex
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

    def _checkpoint(self):
        """Write a partial result.json every step so partial progress survives kills."""
        try:
            out_path = os.path.join(self.runtime.task_dir, "result.json")
            with open(out_path, "w") as f:
                json.dump(self._snapshot(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _snapshot(self) -> dict:
        return {
            "site": self.site,
            "candidates": self.candidate_tasks,
            "history": self.history,
            "notes": self.notes,
            "steps_used": len(self.history),
            "kb": self.router.kb_snapshot() if hasattr(self, "router") and self.router else {},
        }

    # ---------------- main loop -----------------------
    async def run(self) -> dict:
        site = self.site
        try:
            await self.runtime.start()
        except Exception as exc:
            res = {
                "site": site,
                "fatal": True,
                "error": f"runtime.start: {type(exc).__name__}: {exc}",
                "candidates": [],
                "history": [],
            }
            try:
                with open(os.path.join(self.runtime.task_dir, "result.json"), "w") as f:
                    json.dump(res, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return res

        step_idx = 0
        while True:
            step_idx += 1
            # Router decides BEFORE LLM call
            mode = self.router.decide()
            if mode == ExplorerRouter.STOP:
                self.notes.append({"step": step_idx, "router_stop": True})
                break
            if step_idx > self.max_steps + 4:
                # absolute safety cap (max_steps + some retries on misbehaviour)
                self.notes.append({"step": step_idx, "router_stop": "safety_cap"})
                break

            try:
                state = await self.runtime.state()
            except Exception as exc:
                self.notes.append({"step": step_idx, "error": f"state: {exc}"})
                break

            messages = self._build_messages(state, mode)
            # gentle inter-call pacing to avoid gateway anti-abuse (418)
            await asyncio.sleep(1.5)
            try:
                raw = await self._call_llm(messages, mode=mode)
            except Exception as exc:
                err_str = str(exc)
                self.notes.append({"step": step_idx, "mode": mode, "error": f"llm: {err_str}"})
                if any(t in err_str for t in ("fatal 4", "fatal 401", "fatal 403")):
                    break
                await asyncio.sleep(15)
                # treat as failed turn so router can react
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

            # ---- enforce router contract ----
            allowed_kinds = {
                ExplorerRouter.ACT_OPS: {"act"},
                ExplorerRouter.ACT_STAR: {"act"},
                ExplorerRouter.RECORD: {"record_task"},
            }.get(mode, set())
            if kind not in allowed_kinds:
                obs = {"status": "router_violation",
                       "mode": mode, "got_kind": kind,
                       "msg": f"router required {allowed_kinds}, you gave {kind}"}
                self.history.append({"step": step_idx, "mode": mode,
                                     "response_json": parsed, "observation": obs})
                self.router.notify_turn(mode, success=False, kind=kind or "unknown")
                continue

            # ---- enforce TOOL whitelist on ACT_OPS / ACT_STAR ----
            if mode in (ExplorerRouter.ACT_OPS, ExplorerRouter.ACT_STAR):
                tool_whitelist = TOOL_WHITELIST_BY_MODE.get(mode, set())
                tc_name = (parsed.get("tool_call") or {}).get("name", "")
                if tc_name not in tool_whitelist:
                    obs = {"status": "tool_violation",
                           "mode": mode, "got_tool": tc_name,
                           "msg": f"router required tool in {sorted(tool_whitelist)}, "
                                  f"you used {tc_name!r}"}
                    self.history.append({"step": step_idx, "mode": mode,
                                         "response_json": parsed, "observation": obs})
                    self.router.notify_turn(mode, success=False, kind="act")
                    continue

            # ---- execute the allowed kind ----
            if kind == "act":
                tc = parsed.get("tool_call") or {}
                rec = await self.runtime.act(tc)
                obs = {"status": rec.get("status"),
                       "ended_url": rec.get("ended_url"),
                       "error": rec.get("error"),
                       "diagnostics": rec.get("diagnostics")}
                self.history.append({"step": step_idx, "mode": mode,
                                     "response_json": parsed, "observation": obs})
                ok = rec.get("status") == "success"
                if ok:
                    # Accumulate LLM-declared facts/questions into the KB BEFORE
                    # notify_turn (which may flip phase and snapshot phase_findings).
                    raw_facts = parsed.get("facts_learned")
                    raw_qs = parsed.get("facts_to_pursue")
                    facts_before = len(self.router.facts)
                    self.router.absorb_facts(mode, raw_facts)
                    self.router.retire_answered_questions(raw_facts)
                    self.router.absorb_questions(raw_qs)
                    # Dynamic-budget signal: did this step actually grow KB?
                    added = len(self.router.facts) - facts_before
                    self.router.update_dynamic_signals(added)
                self.router.notify_turn(mode, success=ok, kind="act")
            elif kind == "record_task":
                task = parsed.get("task") or {}
                question_ok = bool((task.get("question") or "").strip())
                kb_snap = self.router.kb_snapshot()
                plan_ok, plan_reason = validate_route_plan(task, kb=kb_snap)
                if question_ok and plan_ok:
                    self.candidate_tasks.append(task)
                    success = True
                    obs = {"status": "recorded",
                           "tasks_so_far": len(self.candidate_tasks)}
                else:
                    success = False
                    obs = {"status": "plan_violation" if question_ok else "empty_record",
                           "reason": plan_reason if question_ok else "missing question",
                           "tasks_so_far": len(self.candidate_tasks)}
                self.history.append({"step": step_idx, "mode": mode,
                                     "response_json": parsed, "observation": obs})
                self.router.notify_turn(mode, success=success, kind="record_task")
            elif kind == "done":
                self.history.append({"step": step_idx, "mode": mode,
                                     "response_json": parsed,
                                     "observation": {"status": "done"}})
                self.router.notify_turn(mode, success=True, kind="done")
                break

        await self.runtime.close()
        self._checkpoint()

        return self._snapshot()

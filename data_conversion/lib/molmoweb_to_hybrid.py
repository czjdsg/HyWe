"""Convert a MolmoWeb-HumanTrajs trajectory dict into a sequence of hybrid steps.

Each hybrid step is a dict:
  {
    "src_indices": [int, ...],   # source step indices (1-based) that were merged into this hybrid step
    "tool_call": {"name": str, "arguments": dict} | None,
    "answer": str | None,        # only set on the final hybrid step
    "kind": "tool" | "answer",
    "merge_note": str,           # human-readable summary of why these were merged
  }

The conversion implements:
  - Drop step 1 if `goto` (URL goes into initial_url).
  - Drop all send_msg_to_user steps; instead pick the LAST distinct [ANSWER] text
    and emit a single final hybrid step `kind="answer"` with that text.
  - Collapse `click → keyboard_type+ → keyboard_press(Enter)?` into one `fill`.
  - Map `mouse_drag_and_drop` → hybrid `drag` (now supported).
  - Map `scroll(delta_x, delta_y)` → `scroll({direction:"up"|"down"})`, merging
    consecutive same-direction scrolls at the same region and emitting
    ceil(total_delta / viewport_h) hybrid scrolls (capped at 3).
  - `scroll_at(x, y, ...)` → `scroll({direction, x, y})`, same merging logic.
  - Horizontal-only scroll (`delta_y == 0`) → dropped.
  - `keyboard_press(key)` standalone → `press_key({keys: [key]})`.
  - Lone `keyboard_type` (no preceding click) → kept as `fill` with x,y from the
    immediately preceding action's coordinates if available, else dropped.
  - `goto` at step ≥ 2 → `goto({url})` (mid-trajectory navigation).
  - `browser_nav("back"/"forward"/"refresh")` → `go_back` (back) or kept as
    informational note; forward/refresh have no hybrid equivalent and are dropped.

Output also includes `initial_url` (str), `n_source_steps` (int), `n_hybrid_steps`
(int) and lists of dropped/notes for debugging.
"""

from __future__ import annotations

from typing import Any


TARGET_W_DEFAULT = 1280
TARGET_H_DEFAULT = 720
MAX_SCROLL_REPLICAS = 3


def _ao(step: dict) -> dict:
    return (step.get("action") or {}).get("action_output") or {}


def _act(step: dict) -> dict:
    return (_ao(step).get("action") or {})


def _name(step: dict) -> str:
    return _ao(step).get("action_name", "?")


def _step_dims(step: dict, default_w: int, default_h: int) -> tuple[int, int]:
    w = step.get("image_w") or default_w
    h = step.get("image_h") or default_h
    try:
        return max(1, int(w)), max(1, int(h))
    except (TypeError, ValueError):
        return default_w, default_h


def _scale_coords(act: dict, src_w: int, src_h: int, target_w: int, target_h: int) -> dict:
    """Return a copy of action params with all coord fields rescaled to target
    image space and rounded to integer pixel positions (hybrid uses int coords)."""
    if not act:
        return {}
    sx = target_w / max(1, src_w)
    sy = target_h / max(1, src_h)
    out = dict(act)
    for k in ("x", "from_x", "to_x"):
        if k in out:
            try:
                out[k] = int(round(float(out[k]) * sx))
            except (TypeError, ValueError):
                pass
    for k in ("y", "from_y", "to_y"):
        if k in out:
            try:
                out[k] = int(round(float(out[k]) * sy))
            except (TypeError, ValueError):
                pass
    return out


def convert_trajectory(traj: dict,
                       target_w: int = TARGET_W_DEFAULT,
                       target_h: int = TARGET_H_DEFAULT) -> dict:
    """Convert a MolmoWeb-HumanTrajs trajectory (dict keyed by stringified step id)
    into hybrid step sequence.

    All coordinate-bearing actions are rescaled from the source step's
    (image_w, image_h) into (target_w, target_h) and rounded to integers,
    matching the hybrid runtime's expected integer pixel space."""
    keys = sorted(traj.keys(), key=int)
    src = [traj[k] for k in keys]
    n = len(src)
    initial_url = ""
    hybrid_steps: list[dict] = []
    answer_texts_ordered: list[str] = []
    notes: list[str] = []
    viewport_h = target_h  # scroll page-merge heuristic uses output viewport height

    # Step 0: strip ALL leading consecutive goto steps; use the LAST one's URL
    # as initial_url so the agent starts from the resulting target page.
    start_i = 0
    stripped_urls: list[str] = []
    while start_i < n and _name(src[start_i]) == "goto":
        u = _act(src[start_i]).get("url", "") or ""
        if u:
            stripped_urls.append(u)
        start_i += 1
    if stripped_urls:
        initial_url = stripped_urls[-1]  # the final landing URL after the chained gotos
        notes.append(
            f"stripped {start_i} leading goto step(s); initial_url = {initial_url}"
            + (f" (also passed through: {stripped_urls[:-1]})" if len(stripped_urls) > 1 else "")
        )
    # Fallback: take URL from the first non-goto step's other_obs
    if not initial_url and start_i < n:
        initial_url = (src[start_i].get("other_obs") or {}).get("url", "") or ""
    elif not initial_url and src:
        initial_url = (src[0].get("other_obs") or {}).get("url", "") or ""

    # Collect answer text (last distinct [ANSWER])
    for s in src:
        if _name(s) == "send_msg_to_user":
            msg = (_act(s).get("msg") or "").strip()
            up = msg.upper()
            if up.startswith("[ANSWER]"):
                body = msg[len("[ANSWER]"):].strip()
                if body and body not in answer_texts_ordered:
                    answer_texts_ordered.append(body)

    chosen_answer = answer_texts_ordered[-1] if answer_texts_ordered else ""

    # Coord-cache for orphan keyboard_type (uses last click position)
    last_xy: tuple[float, float] | None = None

    # Coord-cache for orphan keyboard_type stores TARGET-space coords (already scaled).
    # `last_xy` is None or (int, int).
    i = start_i
    while i < n:
        s = src[i]
        name = _name(s)
        raw_act = _act(s)
        s_w, s_h = _step_dims(s, target_w, target_h)
        act = _scale_coords(raw_act, s_w, s_h, target_w, target_h)

        # Drop send_msg_to_user; answer is emitted at the very end.
        if name == "send_msg_to_user":
            i += 1
            continue

        # Mid-trajectory goto
        if name == "goto":
            url = raw_act.get("url", "")
            if url:
                hybrid_steps.append({
                    "src_indices": [i + 1],   # 1-based
                    "tool_call": {"name": "goto", "arguments": {"url": url}},
                    "kind": "tool",
                    "merge_note": "mid-trajectory navigation",
                })
            i += 1
            continue

        # click + keyboard_type [+ keyboard_press Enter] → fill
        if name == "click" and i + 1 < n and _name(src[i + 1]) == "keyboard_type":
            x = act.get("x"); y = act.get("y")
            texts: list[str] = []
            j = i + 1
            while j < n and _name(src[j]) == "keyboard_type":
                texts.append(_act(src[j]).get("text", "") or "")
                j += 1
            submit = False
            if j < n and _name(src[j]) == "keyboard_press":
                key = (_act(src[j]).get("key") or "").lower()
                if key == "enter":
                    submit = True
                    j += 1
            if not isinstance(x, int) or not isinstance(y, int) \
                    or isinstance(x, bool) or isinstance(y, bool):
                notes.append(f"merge_failed: click@step{i+1} missing x/y, skipped block")
                i = j
                continue
            hybrid_steps.append({
                "src_indices": list(range(i + 1, j + 1)),
                "tool_call": {"name": "fill", "arguments": {
                    "x": x, "y": y,
                    "text": "".join(texts),
                    **({"submit": True} if submit else {})
                }},
                "kind": "tool",
                "merge_note": f"merged click + {len(texts)}× keyboard_type" + (" + Enter" if submit else ""),
            })
            last_xy = (x, y)
            i = j
            continue

        # plain click
        if name == "click":
            x = act.get("x"); y = act.get("y")
            if not isinstance(x, int) or not isinstance(y, int) \
                    or isinstance(x, bool) or isinstance(y, bool):
                notes.append(f"drop: click@step{i+1} bad coords")
                i += 1; continue
            hybrid_steps.append({
                "src_indices": [i + 1],
                "tool_call": {"name": "click", "arguments": {"x": x, "y": y}},
                "kind": "tool",
                "merge_note": "",
            })
            last_xy = (x, y)
            i += 1
            continue

        # scroll / scroll_at (with merging).
        # Note: delta_y is read from raw_act (source-pixel units) and compared against
        # the source step's image_h, so the page-count heuristic stays meaningful even
        # when target_h differs. The region (x, y) is taken from act (already scaled).
        if name in ("scroll", "scroll_at"):
            dy = raw_act.get("delta_y") or 0
            if dy == 0:
                notes.append(f"drop: {name}@step{i+1} delta_y=0 (horizontal/no-op)")
                i += 1
                continue
            direction = "down" if dy > 0 else "up"
            region_src: tuple[float, float] | None = None
            region_scaled: tuple[int, int] | None = None
            if name == "scroll_at":
                try:
                    region_src = (float(raw_act["x"]), float(raw_act["y"]))
                    region_scaled = (act["x"], act["y"])
                except (KeyError, TypeError, ValueError):
                    region_src = None
                    region_scaled = None
            total = abs(float(dy))
            ref_image_h = s_h   # use the first step's source viewport for page count
            consumed = [i + 1]
            j = i + 1
            while j < n:
                t = src[j]
                tn = _name(t)
                if tn not in ("scroll", "scroll_at"):
                    break
                t_raw = _act(t)
                tdy = t_raw.get("delta_y") or 0
                if tdy == 0:
                    break
                tdir = "down" if tdy > 0 else "up"
                t_region_src: tuple[float, float] | None = None
                if tn == "scroll_at":
                    try:
                        t_region_src = (float(t_raw["x"]), float(t_raw["y"]))
                    except (KeyError, TypeError, ValueError):
                        t_region_src = None
                if tdir != direction or t_region_src != region_src:
                    break
                total += abs(float(tdy))
                consumed.append(j + 1)
                j += 1
            n_pages = max(1, min(MAX_SCROLL_REPLICAS, round(total / max(1, ref_image_h))))
            for r in range(n_pages):
                args: dict = {"direction": direction}
                if region_scaled is not None:
                    args["x"] = region_scaled[0]; args["y"] = region_scaled[1]
                hybrid_steps.append({
                    "src_indices": consumed if r == 0 else [],
                    "tool_call": {"name": "scroll", "arguments": args},
                    "kind": "tool",
                    "merge_note": (
                        f"merged {len(consumed)} {name} (src |delta_y|={total:.0f}, "
                        f"emit {n_pages}× scroll)" if r == 0 else "replica scroll"
                    ),
                })
            i = j
            continue

        # mouse_drag_and_drop → drag
        if name == "mouse_drag_and_drop":
            fx = act.get("from_x"); fy = act.get("from_y")
            tx = act.get("to_x");   ty = act.get("to_y")
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in (fx, fy, tx, ty)):
                notes.append(f"drop: drag@step{i+1} bad coords")
                i += 1; continue
            hybrid_steps.append({
                "src_indices": [i + 1],
                "tool_call": {"name": "drag", "arguments": {
                    "from_x": fx, "from_y": fy, "to_x": tx, "to_y": ty}},
                "kind": "tool",
                "merge_note": "",
            })
            i += 1
            continue

        # standalone keyboard_press (not consumed by fill) → press_key.
        # MolmoWeb stores combos as a single "+"-joined string (e.g. "ControlOrMeta+a");
        # split into [modifier..., trigger] so hybrid's key_combination can issue
        # explicit keyboard.down for modifiers and keyboard.press for the trigger.
        if name == "keyboard_press":
            key = raw_act.get("key", "") or ""
            if key:
                keys = [k for k in key.split("+") if k]
                hybrid_steps.append({
                    "src_indices": [i + 1],
                    "tool_call": {"name": "press_key", "arguments": {"keys": keys}},
                    "kind": "tool",
                    "merge_note": "standalone keyboard_press"
                                  + (f" (split combo: {key})" if len(keys) > 1 else ""),
                })
            i += 1
            continue

        # orphan keyboard_type (no preceding click in this iteration) → fill at last_xy
        if name == "keyboard_type":
            text = raw_act.get("text", "") or ""
            if last_xy and text:
                submit = False
                j = i + 1
                if j < n and _name(src[j]) == "keyboard_press" and (_act(src[j]).get("key") or "").lower() == "enter":
                    submit = True
                    j += 1
                hybrid_steps.append({
                    "src_indices": list(range(i + 1, j + 1)),
                    "tool_call": {"name": "fill", "arguments": {
                        "x": last_xy[0], "y": last_xy[1], "text": text,
                        **({"submit": True} if submit else {})
                    }},
                    "kind": "tool",
                    "merge_note": "orphan keyboard_type → fill at last click pos",
                })
                i = j
                continue
            notes.append(f"drop: orphan keyboard_type@step{i+1} no anchor coord")
            i += 1
            continue

        # hover → hover (rare in MolmoWeb but possible)
        if name == "hover":
            x = act.get("x"); y = act.get("y")
            if not isinstance(x, int) or not isinstance(y, int) \
                    or isinstance(x, bool) or isinstance(y, bool):
                i += 1; continue
            hybrid_steps.append({
                "src_indices": [i + 1],
                "tool_call": {"name": "hover", "arguments": {"x": x, "y": y}},
                "kind": "tool",
                "merge_note": "",
            })
            i += 1
            continue

        # browser_nav: back → go_back; forward/refresh dropped (no hybrid equiv)
        if name == "browser_nav":
            sub = (raw_act.get("direction") or raw_act.get("nav") or "").lower()
            if sub == "back":
                hybrid_steps.append({
                    "src_indices": [i + 1],
                    "tool_call": {"name": "go_back", "arguments": {}},
                    "kind": "tool",
                    "merge_note": "browser_nav back",
                })
            else:
                notes.append(f"drop: browser_nav@step{i+1} subtype={sub!r} (no hybrid equiv)")
            i += 1
            continue

        # Unknown — drop with note
        notes.append(f"drop: unknown action @step{i+1} name={name!r}")
        i += 1

    # Finally emit the answer step
    if chosen_answer:
        hybrid_steps.append({
            "src_indices": [],   # synthetic; no specific source step
            "tool_call": None,
            "answer": chosen_answer,
            "kind": "answer",
            "merge_note": "final answer extracted from [ANSWER] tag(s)",
        })

    return {
        "initial_url": initial_url,
        "n_source_steps": n,
        "n_hybrid_steps": len(hybrid_steps),
        "hybrid_steps": hybrid_steps,
        "n_answer_distinct": len(answer_texts_ordered),
        "chosen_answer": chosen_answer,
        "notes": notes,
    }

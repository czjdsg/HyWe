"""Convert a WebChain trajectory JSON into hybrid step sequence.

WebChain step types we keep + how they map:
  launchApp  → drop step (its href becomes initial_url)
  click      → click({x, y})           [bbox center]
  type       → fill({x, y, text, submit?})  [submit if next step is press_enter]
  press_enter→ consumed by preceding type (or standalone press_key([Enter]))
  hover      → hover({x, y})
  drag       → drag({from_x, from_y, to_x, to_y})  [bbox center → pageX/pageY]
  select     → select_option({x, y, value})
  back       → go_back
  double_click → double_click({x, y})
We DROP entire trajs that contain: copy, paste, right_click.
"""

from __future__ import annotations
from typing import Any


EXCLUDE_TYPES = {"copy", "paste", "right_click"}
TARGET_ASPECT = 1280 / 720  # 16:9 = 1.7778
MAX_ASPECT_DEV = 0.20        # allow aspect ratio in [1.42, 2.13]


def _rect_center(step: dict) -> tuple[float, float] | None:
    rect = step.get("rect") or {}
    try:
        x = float(rect["x"]); y = float(rect["y"])
        w = float(rect.get("width", 0)); h = float(rect.get("height", 0))
    except (KeyError, TypeError, ValueError):
        return None
    return (x + w / 2, y + h / 2)


def _scale_xy(x: float, y: float, src_w: int, src_h: int,
              target_w: int, target_h: int) -> tuple[int, int]:
    sx = target_w / max(1, src_w)
    sy = target_h / max(1, src_h)
    return int(round(x * sx)), int(round(y * sy))


def _step_dims(step: dict, default_w: int, default_h: int) -> tuple[int, int]:
    vp = step.get("viewport") or {}
    try:
        w = int(vp.get("width") or default_w)
        h = int(vp.get("height") or default_h)
        return max(1, w), max(1, h)
    except (TypeError, ValueError):
        return default_w, default_h


def convert_trajectory(traj: dict, target_w: int = 1280, target_h: int = 720) -> dict:
    """Convert one WebChain trajectory into hybrid step sequence.

    Returns dict with:
      - skip_reason: str | None (if trajectory should be dropped)
      - initial_url: str
      - n_source_steps: int
      - n_hybrid_steps: int
      - hybrid_steps: list[dict]   (each has src_indices, tool_call, kind, merge_note)
      - notes: list[str]
    """
    steps = traj.get("steps") or []
    n_src = len(steps)
    notes: list[str] = []

    # 1) Skip trajs with excluded action types
    types_in = {s.get("type") for s in steps}
    bad = types_in & EXCLUDE_TYPES
    if bad:
        return {
            "skip_reason": f"contains excluded action(s): {sorted(bad)}",
            "initial_url": "",
            "n_source_steps": n_src, "n_hybrid_steps": 0,
            "hybrid_steps": [], "notes": notes,
        }

    if n_src == 0:
        return {
            "skip_reason": "empty_trajectory",
            "initial_url": "", "n_source_steps": 0, "n_hybrid_steps": 0,
            "hybrid_steps": [], "notes": notes,
        }

    # 1.5) Aspect ratio filter — use the FIRST step's viewport that has a sensible size.
    # Drop trajs whose recorded viewport deviates > MAX_ASPECT_DEV from 16:9 target,
    # since resizing those screenshots to 1280×720 would distort too much.
    src_aspect = None
    for s in steps:
        vp = s.get("viewport") or {}
        try:
            w = float(vp.get("width") or 0); h = float(vp.get("height") or 0)
            if w > 0 and h > 0:
                src_aspect = w / h
                break
        except (TypeError, ValueError):
            continue
    if src_aspect is not None:
        dev = abs(src_aspect / TARGET_ASPECT - 1.0)
        if dev > MAX_ASPECT_DEV:
            return {
                "skip_reason": f"aspect_too_different (ratio={src_aspect:.3f}, dev={dev:.2f})",
                "initial_url": "", "n_source_steps": n_src, "n_hybrid_steps": 0,
                "hybrid_steps": [], "notes": notes,
            }

    # 2) Pull initial_url from earliest available href (first launchApp or first non-empty href)
    initial_url = ""
    for s in steps:
        h = (s.get("href") or "").strip()
        if h:
            initial_url = h
            break

    hybrid_steps: list[dict] = []
    i = 0

    # 3) Skip a leading launchApp step (its href is the initial URL)
    while i < n_src and steps[i].get("type") == "launchApp":
        notes.append(f"stripped leading launchApp@step{i+1}")
        i += 1

    while i < n_src:
        s = steps[i]
        t = s.get("type", "?")
        src_w, src_h = _step_dims(s, target_w, target_h)

        if t == "launchApp":
            # Mid-trajectory launchApp = navigation event; drop (no clean hybrid mapping)
            notes.append(f"drop: launchApp@step{i+1}")
            i += 1
            continue

        if t == "click":
            ctr = _rect_center(s)
            if ctr is None:
                notes.append(f"drop: click@step{i+1} missing rect")
                i += 1; continue
            x, y = _scale_xy(ctr[0], ctr[1], src_w, src_h, target_w, target_h)
            hybrid_steps.append({
                "src_indices": [i + 1],
                "tool_call": {"name": "click", "arguments": {"x": x, "y": y}},
                "kind": "tool", "merge_note": "",
            })
            i += 1; continue

        if t == "type":
            ctr = _rect_center(s)
            if ctr is None:
                notes.append(f"drop: type@step{i+1} missing rect")
                i += 1; continue
            x, y = _scale_xy(ctr[0], ctr[1], src_w, src_h, target_w, target_h)
            text = s.get("value", "") or ""
            consumed = [i + 1]
            # Look ahead: if next is press_enter, merge as submit
            submit = False
            j = i + 1
            if j < n_src and steps[j].get("type") == "press_enter":
                submit = True
                consumed.append(j + 1)
                j += 1
            args = {"x": x, "y": y, "text": text}
            if submit: args["submit"] = True
            hybrid_steps.append({
                "src_indices": consumed,
                "tool_call": {"name": "fill", "arguments": args},
                "kind": "tool",
                "merge_note": "type + press_enter → fill+submit" if submit else "",
            })
            i = j; continue

        if t == "press_enter":
            # Standalone press_enter → press_key([Enter])
            hybrid_steps.append({
                "src_indices": [i + 1],
                "tool_call": {"name": "press_key", "arguments": {"keys": ["Enter"]}},
                "kind": "tool",
                "merge_note": "standalone press_enter",
            })
            i += 1; continue

        if t == "hover":
            ctr = _rect_center(s)
            if ctr is None:
                notes.append(f"drop: hover@step{i+1} missing rect")
                i += 1; continue
            x, y = _scale_xy(ctr[0], ctr[1], src_w, src_h, target_w, target_h)
            hybrid_steps.append({
                "src_indices": [i + 1],
                "tool_call": {"name": "hover", "arguments": {"x": x, "y": y}},
                "kind": "tool", "merge_note": "",
            })
            i += 1; continue

        if t == "drag":
            ctr = _rect_center(s)
            if ctr is None:
                notes.append(f"drop: drag@step{i+1} missing rect")
                i += 1; continue
            try:
                to_x_src = float(s.get("pageX") or s.get("clientX") or 0)
                to_y_src = float(s.get("pageY") or s.get("clientY") or 0)
            except (TypeError, ValueError):
                notes.append(f"drop: drag@step{i+1} missing target coords")
                i += 1; continue
            if to_x_src == 0 and to_y_src == 0:
                notes.append(f"drop: drag@step{i+1} target coords (0, 0)")
                i += 1; continue
            from_x, from_y = _scale_xy(ctr[0], ctr[1], src_w, src_h, target_w, target_h)
            to_x, to_y = _scale_xy(to_x_src, to_y_src, src_w, src_h, target_w, target_h)
            hybrid_steps.append({
                "src_indices": [i + 1],
                "tool_call": {"name": "drag", "arguments": {
                    "from_x": from_x, "from_y": from_y, "to_x": to_x, "to_y": to_y,
                }},
                "kind": "tool", "merge_note": "",
            })
            i += 1; continue

        if t == "select":
            # WebChain records select on the <option> (no rect), but the IMMEDIATELY
            # PRECEDING step is typically a click on the parent <select> with the
            # right rect. Reuse it.
            value = (s.get("value") or "").strip()
            if not value:
                notes.append(f"drop: select@step{i+1} missing value")
                i += 1; continue
            # Look back up to 3 steps for a click whose selector is a prefix of this select's selector
            ctr = _rect_center(s)
            reused_from = None
            if ctr is None:
                sel = (s.get("selector") or "")
                # Strip the option-suffix to get the parent <select> selector
                parent_sel = sel.split(">option", 1)[0] if ">option" in sel else sel
                for back in range(1, 4):
                    j = i - back
                    if j < 0: break
                    prev = steps[j]
                    if prev.get("type") not in ("click", "hover"):
                        continue
                    prev_sel = (prev.get("selector") or "")
                    # Match by selector prefix OR just use the most recent click rect
                    if prev_sel and parent_sel and (parent_sel.startswith(prev_sel) or prev_sel.startswith(parent_sel)):
                        ctr = _rect_center(prev)
                        if ctr:
                            reused_from = j + 1
                            break
                    # Fallback: just take the most recent click that has a rect
                    if ctr is None and prev.get("type") == "click":
                        ctr = _rect_center(prev)
                        if ctr:
                            reused_from = j + 1
                            break
            if ctr is None:
                notes.append(f"drop: select@step{i+1} missing rect (no usable prior click)")
                i += 1; continue
            x, y = _scale_xy(ctr[0], ctr[1], src_w, src_h, target_w, target_h)
            note = f"reused rect from step {reused_from}" if reused_from else ""
            hybrid_steps.append({
                "src_indices": [i + 1],
                "tool_call": {"name": "select_option", "arguments": {
                    "x": x, "y": y, "value": value,
                }},
                "kind": "tool", "merge_note": note,
            })
            i += 1; continue

        if t == "double_click":
            ctr = _rect_center(s)
            if ctr is None:
                notes.append(f"drop: double_click@step{i+1} missing rect")
                i += 1; continue
            x, y = _scale_xy(ctr[0], ctr[1], src_w, src_h, target_w, target_h)
            hybrid_steps.append({
                "src_indices": [i + 1],
                "tool_call": {"name": "double_click", "arguments": {"x": x, "y": y}},
                "kind": "tool", "merge_note": "",
            })
            i += 1; continue

        if t == "back":
            hybrid_steps.append({
                "src_indices": [i + 1],
                "tool_call": {"name": "go_back", "arguments": {}},
                "kind": "tool", "merge_note": "",
            })
            i += 1; continue

        # Unknown — drop with note (shouldn't happen after EXCLUDE filter)
        notes.append(f"drop: unknown type@step{i+1} type={t!r}")
        i += 1

    return {
        "skip_reason": None,
        "initial_url": initial_url,
        "n_source_steps": n_src,
        "n_hybrid_steps": len(hybrid_steps),
        "hybrid_steps": hybrid_steps,
        "notes": notes,
    }

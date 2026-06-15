#!/usr/bin/env python3
"""Side-by-side viewer: SOURCE trajectory step  vs  STAGE-2 REWRITE.

Joins the skeleton file (has source_think / source_observation_after) with the
rewrite file (has the DeepSeek rewrite dict) on (source_traj_id, step_idx), and
shows them column-by-column per step so you can eyeball whether the rewrite
preserves the original reasoning — especially the "this query failed, so I
switch" negative-feedback signal.

Memory-safe: builds a byte-offset index per file at startup (no full load);
reads a trajectory's lines on demand.

Left column shows BOTH the raw original assistant text (source_raw_assistant,
the native <think>/<tool_call>/<answer> format) AND the parsed source_think /
observation. Right column shows the structured rewrite.

--skeleton accepts a comma-separated list so RedSearcher AND OpenSeeker sources
can be browsed together; use the source filter in the top bar to narrow down.

Usage:
    python viewer/src_vs_rewrite_viewer.py [PORT] \
        --skeleton data/redsearcher_skeleton_correct.jsonl,data/openseeker_skeleton_correct.jsonl \
        --rewrite  data/hybrid_search_deepseek_v4_flash_full.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

from flask import Flask, render_template_string, request

app = Flask(__name__)

# --- Search-mode (join skeleton + rewrite) ---
SK_PATHS: list[str] = []
RW_PATH = None
SK_INDEX: dict[str, tuple[str, list[int]]] = {}
RW_INDEX: dict[str, list[int]] = defaultdict(list)

# --- Operation-mode (single self-contained file per source) ---
# tid -> (file_path, [offsets]); op tasks carry rewrite + screenshot_ref inline.
OP_INDEX: dict[str, tuple[str, list[int]]] = {}
OP_FILES: list[str] = []  # list of op rewrite jsonl paths

TRAJ_IDS: list[str] = []

# Words that signal the agent hit a dead-end / negative feedback in the SOURCE.
NEG_PAT = re.compile(
    r"(不理想|没帮助|没什么帮助|没有结果|没有直接|换个思路|换一个|重新|再试|无效|不对|"
    r"没有找到|未找到|no results?|not helpful|didn't help|did not help|no direct|"
    r"unproductive|irrelevant|let me try|instead|however)",
    re.IGNORECASE,
)


_TID_RE = re.compile(rb'"source_traj_id"\s*:\s*"([^"]+)"')


def build_index(path: str) -> dict[str, list[int]]:
    """Fast index: regex-extract source_traj_id per line; no full json parse."""
    idx: dict[str, list[int]] = defaultdict(list)
    n = 0
    with open(path, "rb") as f:
        offset = f.tell()
        line = f.readline()
        while line:
            m = _TID_RE.search(line)
            if m:
                idx[m.group(1).decode("utf-8", "replace")].append(offset)
            n += 1
            offset = f.tell()
            line = f.readline()
    print(f"[viewer] indexed {len(idx)} trajs / {n} lines from {path}")
    return idx


def read_lines(path: str, offsets: list[int]) -> list[dict]:
    out = []
    with open(path, "rb") as f:
        for off in offsets:
            f.seek(off)
            try:
                out.append(json.loads(f.readline()))
            except Exception:
                pass
    return out


import base64

# Image dir per operation source (screenshots are the "observation" for op tasks).
OP_IMAGE_DIRS = {
    "molmoweb": "data/molmoweb_images",
    "webchain": "data/webchain_images",
}


def op_image_b64(record: dict, image_dir: str) -> str:
    """Load the BEFORE-action screenshot for an operation step, draw the
    click/drag marker, return base64 jpeg. Tries {step}.jpg then {step}.png."""
    tid = record.get("source_traj_id", "")
    sid = tid.split(":", 1)[-1]
    ref = record.get("screenshot_ref") or {}
    src_step = ref.get("source_step_idx", record.get("step_idx", 0))
    base = Path(image_dir) / sid
    path = None
    for ext in (".jpg", ".png", ".jpeg"):
        p = base / f"{src_step}{ext}"
        if p.exists():
            path = p
            break
    if path is None:
        return ""
    try:
        from PIL import Image, ImageDraw
        import io

        img = Image.open(path).convert("RGB")
        tc = record.get("tool_call") or record.get("mapped_tool_call") or {}
        args = tc.get("arguments", {}) if isinstance(tc, dict) else {}

        def _num(v):
            try:
                return int(float(v))
            except (TypeError, ValueError):
                return None

        draw = ImageDraw.Draw(img)
        x, y = _num(args.get("x")), _num(args.get("y"))
        if x is not None and y is not None:
            r = 12
            draw.ellipse([x - r, y - r, x + r, y + r], outline="red", width=3)
            draw.line([x - r, y, x + r, y], fill="red", width=2)
            draw.line([x, y - r, x, y + r], fill="red", width=2)
            draw.text((x + r + 4, y - 8), f"({x},{y})", fill="red")
        fx, fy, tx, ty = (_num(args.get(k)) for k in ("from_x", "from_y", "to_x", "to_y"))
        if None not in (fx, fy, tx, ty):
            draw.ellipse([fx - 8, fy - 8, fx + 8, fy + 8], outline="blue", width=2)
            draw.ellipse([tx - 8, ty - 8, tx + 8, ty + 8], outline="green", width=2)
            draw.line([fx, fy, tx, ty], fill="yellow", width=2)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        try:
            return base64.b64encode(path.read_bytes()).decode("ascii")
        except Exception:
            return ""


# Words that suggest the quality reason is describing the PREVIOUS step's
# action/result (a "backward-looking" reason on a forward-looking label —
# direct visual evidence that the label is shifted by one step).
BACKWARD_PAT = re.compile(
    r"\b(previous|prior|last|initial|the result|the search result|returned|"
    r"was too|too broad|too narrow|too specific|too vague|too generic|"
    r"confirmed|matched|directly matches|first attempt|first search|"
    r"yielded|gave|found|surfaced|revealed|no progress|backtrack|"
    r"上一(次|步|轮)|前一(步|次)|之前|此前|刚才|之前的|前面的|"
    r"结果|搜索结果|返回|过于宽泛|过于狭窄|太宽|太窄|too narrow|earlier)\b",
    re.IGNORECASE,
)


def highlight_backward(text: str) -> str:
    """Mark backward-looking words in the quality reason."""
    if not text:
        return ""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return BACKWARD_PAT.sub(lambda m: f"<mark class='back'>{m.group(0)}</mark>", text)


_RAW_TOOLCALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_RAW_CODE_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)


def extract_raw_tool_call(raw_assistant: str) -> str:
    """Pull the PRE-MAPPING original tool_call out of source_raw_assistant.

    Returns a readable string: the raw <tool_call> JSON, plus any <code> block
    (PythonInterpreter puts code outside the JSON). Empty string if none.
    """
    if not raw_assistant:
        return ""
    m = _RAW_TOOLCALL_RE.search(raw_assistant)
    if not m:
        return ""
    out = m.group(1).strip()
    c = _RAW_CODE_RE.search(raw_assistant)
    if c:
        out += "\n<code>\n" + c.group(1).strip() + "\n</code>"
    return out


def highlight_neg(text: str) -> str:
    """Wrap negative-feedback phrases in a <mark> so they pop visually."""
    if not text:
        return ""
    # escape minimal html
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return NEG_PAT.sub(lambda m: f"<mark class='neg'>{m.group(0)}</mark>", text)


HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Source vs Rewrite</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background:#f0f2f5; }
  .top { position: sticky; top:0; background:#111827; color:#fff; padding:10px 18px; z-index:10;
         display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
  .top a { color:#fff; background:#2563eb; padding:5px 12px; border-radius:6px; text-decoration:none; font-size:13px; }
  .top a:hover { background:#1d4ed8; }
  .top form { display:inline; }
  .top input { padding:4px 8px; border-radius:5px; border:none; font-size:13px; width:170px; }
  .meta { background:#fff; margin:14px 18px; padding:10px 16px; border-radius:8px; font-size:13px; border:1px solid #e5e7eb;}
  .meta b{color:#374151;}
  .legend { font-size:12px; color:#6b7280; margin-left:auto; }
  mark.neg { background:#fde68a; color:#7c2d12; padding:0 2px; border-radius:3px; }
  mark.back{ background:#fecaca; color:#7f1d1d; padding:0 2px; border-radius:3px; }
  .qreason { display:inline-block; max-width:560px; margin-left:8px; padding:3px 8px;
             background:#fff7ed; border:1px solid #fed7aa; border-radius:5px;
             font-size:11.5px; color:#7c2d12; vertical-align:middle; line-height:1.3; }
  .step { margin:14px 18px; background:#fff; border:1px solid #e5e7eb; border-radius:10px; overflow:hidden; }
  .step-hd { background:#f9fafb; padding:8px 14px; font-weight:600; font-size:13px; border-bottom:1px solid #eee;
             display:flex; gap:10px; align-items:center;}
  .tag{ padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600;}
  .t-step{background:#ede9fe;color:#5b21b6;} .t-final{background:#dcfce7;color:#166534;}
  .t-visit{background:#fef9c3;color:#854d0e;} .t-search{background:#dbeafe;color:#1e40af;}
  .q-good{background:#dcfce7;color:#166534;} .q-mediocre{background:#fef3c7;color:#92400e;}
  .q-poor{background:#fee2e2;color:#991b1b;} .q-fatal{background:#7f1d1d;color:#fff;}
  .neg-flag{background:#fef08a;color:#854d0e;}
  .cols{ display:grid; grid-template-columns:1fr 1fr; gap:0;}
  .col{ padding:12px 14px; font-size:12.5px; line-height:1.55;}
  .col.src{ border-right:2px solid #e5e7eb; background:#fffdf7;}
  .col.rw{ background:#f7fbff;}
  .col h5{ margin:0 0 8px; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
  .src h5{color:#92400e;} .rw h5{color:#1e40af;}
  .fl{ color:#6b7280; font-weight:700; font-size:11px; text-transform:uppercase; margin-top:8px;}
  .fv{ margin:2px 0 6px; white-space:pre-wrap; }
  pre{ background:#f3f4f6; padding:6px 8px; border-radius:4px; font-size:11.5px; white-space:pre-wrap;
       max-height:260px; overflow:auto; margin:3px 0;}
  pre.raw{ background:#fef6e7; border:1px solid #f0d9a8; max-height:340px; color:#3f3f46;}
  pre.mapped{ background:#eff6ff; border:1px solid #bfdbfe; color:#1e3a5f;}
  details summary{ list-style:none; }
  details summary::-webkit-details-marker{ display:none; }
  .dim{color:#9ca3af;}
  .shot{ max-width:100%; border:1px solid #d1d5db; border-radius:6px; cursor:zoom-in; margin:4px 0; }
  .shot.zoom{ max-width:none; width:auto; cursor:zoom-out; position:relative; z-index:5; }
  .no-img{ color:#9ca3af; font-style:italic; padding:8px; }
  .merge{ background:#ede9fe; color:#5b21b6; padding:5px 9px; border-radius:5px; font-size:12px;
          font-weight:600; margin:4px 0; border:1px solid #ddd6fe;}
</style></head><body>
<div class="top">
  <a href="/?random=1&src={{ src_filter }}">🎲 Random</a>
  <a href="/?idx={{ prev_idx }}&src={{ src_filter }}">◀ Prev</a>
  <a href="/?idx={{ next_idx }}&src={{ src_filter }}">Next ▶</a>
  <span style="font-size:13px;">{{ idx+1 }} / {{ total }}</span>
  <span style="font-size:12px;">source:</span>
  <a href="/?src=all" style="background:{{ '#16a34a' if src_filter=='all' else '#374151' }}">all ({{ n_red+n_os+n_mw+n_wc }})</a>
  <a href="/?src=redsearcher" style="background:{{ '#16a34a' if src_filter=='redsearcher' else '#374151' }}">RedSearcher ({{ n_red }})</a>
  <a href="/?src=openseeker" style="background:{{ '#16a34a' if src_filter=='openseeker' else '#374151' }}">OpenSeeker ({{ n_os }})</a>
  <a href="/?src=molmoweb" style="background:{{ '#16a34a' if src_filter=='molmoweb' else '#374151' }}">MolmoWeb ({{ n_mw }})</a>
  <a href="/?src=webchain" style="background:{{ '#16a34a' if src_filter=='webchain' else '#374151' }}">WebChain ({{ n_wc }})</a>
  <form action="/" method="get"><input name="tid" placeholder="jump to traj_id" value="{{ traj_id }}"></form>
  <span class="legend">left = <b>tool_call BEFORE mapping</b> + source · right = <b>tool_call AFTER mapping</b> + rewrite · <mark class="neg">yellow</mark> = neg-feedback · ⚑ src neg-feedback · 🟧 step header shows original <b>step_quality_reason</b>; <mark class="back">red words</mark> = backward-looking phrases (reason describing the PREVIOUS step's action/result)</span>
</div>
<div class="meta">
  <b>Traj:</b> {{ traj_id }} &nbsp;|&nbsp; <b>Source:</b> {{ data_source }} &nbsp;|&nbsp;
  <b>Steps:</b> {{ n_steps }} &nbsp;|&nbsp; <b>Steps w/ source neg-feedback:</b> {{ neg_count }}
  <div style="margin-top:6px;"><b>Task:</b> {{ task }}</div>
  {% if gt %}<div style="margin-top:4px;"><b>GT answer:</b> <span style="color:#059669;font-weight:600;">{{ gt }}</span></div>{% endif %}
</div>
{% for s in steps %}
<div class="step">
  <div class="step-hd">
    <span class="tag t-step">Step {{ s.step_idx }}/{{ s.total_steps }}</span>
    {% if s.is_final %}<span class="tag t-final">FINAL</span>{% endif %}
    {% if s.tool %}<span class="tag t-{{ s.tool }}">{{ s.tool }}</span>{% endif %}
    {% if s.quality %}<span class="tag q-{{ s.quality }}" title="{{ s.quality }}">{{ s.quality }}</span>{% endif %}
    {% if s.quality_reason %}<span class="qreason" title="original step_quality_reason from DeepSeek rewrite">{{ s.quality_reason|safe }}</span>{% endif %}
    {% if s.src_has_neg %}<span class="tag neg-flag">⚑ src neg-feedback</span>{% endif %}
  </div>
  <div class="cols">
    <div class="col src">
    {% if is_op %}
      <h5>Source (merged step{% if s.merge_idx %} — src steps {{ s.merge_idx }}{% endif %})</h5>
      {% if s.merge_note %}<div class="merge">⛓ {{ s.merge_note }}{% if s.merge_idx %} &nbsp;·&nbsp; merged source steps: [{{ s.merge_idx }}]{% endif %}</div>{% endif %}
      {% if s.img_b64 %}<img class="shot" src="data:image/jpeg;base64,{{ s.img_b64 }}" onclick="this.classList.toggle('zoom')">{% else %}<div class="no-img">no screenshot</div>{% endif %}
      <div class="fl">source_thought (human annotator)</div><div class="fv">{{ s.src_thought|safe }}</div>
    {% else %}
      <h5>Source (original format)</h5>
      {% if s.src_raw %}
      <details open><summary class="fl" style="cursor:pointer;">source_raw_assistant (raw &lt;think&gt;/&lt;tool_call&gt; format)</summary>
      <pre class="raw">{{ s.src_raw }}</pre></details>
      {% endif %}
      <div class="fl">source_think (parsed)</div><div class="fv">{{ s.src_think|safe }}</div>
      {% if s.tool_raw %}<div class="fl" style="color:#b45309;">tool_call — BEFORE mapping (raw)</div><pre class="raw">{{ s.tool_raw }}</pre>{% endif %}
      <div class="fl">observation_after</div><pre>{{ s.src_obs }}</pre>
    {% endif %}
    </div>
    <div class="col rw">
      <h5>Rewrite (mapped + structured)</h5>
      {% if s.tool_mapped %}<div class="fl" style="color:#1d4ed8;">tool_call — AFTER mapping (hybrid)</div><pre class="mapped">{{ s.tool_mapped }}</pre>{% endif %}
      {% if s.this_turn %}<div class="fl">this_turn</div><div class="fv">{{ s.this_turn|safe }}</div>{% endif %}
      {% if s.rationale %}<div class="fl">next_action_rationale</div><div class="fv">{{ s.rationale|safe }}</div>{% endif %}
      {% if s.action %}<div class="fl">action_description</div><div class="fv">{{ s.action }}</div>{% endif %}
      {% if s.cumulative %}<div class="fl">cumulative_progress</div><pre>{{ s.cumulative }}</pre>{% endif %}
      {% if s.remaining %}<div class="fl">remaining_subgoals</div><pre>{{ s.remaining }}</pre>{% endif %}
      {% if s.task_analysis %}<div class="fl">task_analysis</div><pre>{{ s.task_analysis }}</pre>{% endif %}
      {% if s.answer %}<div class="fl">answer</div><div class="fv" style="color:#059669;font-weight:600;">{{ s.answer }}</div>{% endif %}
      {% if s.rw_err %}<div class="fl" style="color:#b91c1c;">rewrite_error</div><pre>{{ s.rw_err }}</pre>{% endif %}
    </div>
  </div>
</div>
{% endfor %}
</body></html>"""


def _build_search_steps(tid):
    """Search-mode: join skeleton (source) + rewrite. Returns
    (steps, task, data_source, gt, neg_count)."""
    sk_path, sk_offs = SK_INDEX.get(tid, (None, []))
    sk_rows = read_lines(sk_path, sk_offs) if sk_path else []
    rw_rows = read_lines(RW_PATH, RW_INDEX.get(tid, []))
    sk_by = {r.get("step_idx"): r for r in sk_rows}
    rw_by = {r.get("step_idx"): r for r in rw_rows}
    all_steps = sorted(set(sk_by) | set(rw_by), key=lambda x: (x is None, x))

    task = (sk_rows or rw_rows or [{}])[0].get("task", "")
    data_source = (sk_rows or rw_rows or [{}])[0].get("source", "")
    gt = ""
    for r in sk_rows:
        gt = (r.get("meta") or {}).get("ground_truth_answer") or gt

    steps = []
    neg_count = 0
    for si in all_steps:
        sk = sk_by.get(si, {})
        rw_row = rw_by.get(si, {})
        rw = rw_row.get("rewrite") or {}
        think = sk.get("source_think") or ""
        src_has_neg = bool(NEG_PAT.search(think))
        if src_has_neg:
            neg_count += 1
        tc = sk.get("mapped_tool_call") or rw_row.get("tool_call") or {}
        tool = (tc.get("name") if isinstance(tc, dict) else "") or ""
        steps.append({
            "step_idx": si, "total_steps": sk.get("total_steps") or rw_row.get("total_steps") or "?",
            "is_final": sk.get("is_final") or rw_row.get("is_final"),
            "tool": tool, "quality": rw.get("step_quality", ""),
            "quality_reason": highlight_backward(rw.get("step_quality_reason", "") or ""),
            "src_has_neg": src_has_neg,
            "src_raw": (sk.get("source_raw_assistant") or "").strip(),
            "src_think": highlight_neg(think),
            "tool_raw": extract_raw_tool_call(sk.get("source_raw_assistant") or ""),
            "tool_mapped": json.dumps(tc, ensure_ascii=False, indent=2) if tc else "",
            "src_obs": (sk.get("source_observation_after") or "").strip() or "(none)",
            "img_b64": "", "merge_note": "", "merge_idx": "", "src_thought": "",
            "this_turn": highlight_neg(rw.get("this_turn", "")),
            "rationale": highlight_neg(rw.get("next_action_rationale", "")),
            "action": rw.get("action_description", ""),
            "cumulative": rw.get("cumulative_progress", ""),
            "remaining": json.dumps(rw.get("remaining_subgoals", []), ensure_ascii=False, indent=2) if rw.get("remaining_subgoals") else "",
            "task_analysis": rw.get("task_analysis", ""),
            "answer": rw.get("answer", ""), "rw_err": rw_row.get("rewrite_error", ""),
        })
    return steps, task, data_source, gt, neg_count


def _build_op_steps(tid):
    """Operation-mode (molmoweb/webchain): single self-contained file. Left =
    screenshot (the observation) + source_thought + merge info for the merged
    source steps; right = rewrite. Returns (steps, task, data_source, gt, neg)."""
    op_path, offs = OP_INDEX.get(tid, (None, []))
    rows = read_lines(op_path, offs) if op_path else []
    rows.sort(key=lambda r: (r.get("step_idx") is None, r.get("step_idx", 0)))
    src_name = tid.split(":", 1)[0]
    image_dir = OP_IMAGE_DIRS.get(src_name, "")
    task = (rows or [{}])[0].get("task", "")
    data_source = (rows or [{}])[0].get("source", src_name)

    steps = []
    neg_count = 0
    for r in rows:
        rw = r.get("rewrite") or {}
        thought = r.get("source_thought") or ""
        src_has_neg = bool(NEG_PAT.search(thought))
        if src_has_neg:
            neg_count += 1
        tc = r.get("tool_call") or {}
        tool = (tc.get("name") if isinstance(tc, dict) else "") or ""
        merge_idx = r.get("src_merge_indices") or []
        steps.append({
            "step_idx": r.get("step_idx"), "total_steps": r.get("total_steps", "?"),
            "is_final": r.get("is_final"), "tool": tool, "quality": rw.get("step_quality", ""),
            "quality_reason": highlight_backward(rw.get("step_quality_reason", "") or ""),
            "src_has_neg": src_has_neg,
            # op-specific left column
            "img_b64": op_image_b64(r, image_dir) if image_dir else "",
            "src_thought": highlight_neg(thought),
            "merge_note": r.get("merge_note") or "",
            "merge_idx": (", ".join(map(str, merge_idx)) if isinstance(merge_idx, list) and merge_idx else ""),
            "tool_mapped": json.dumps(tc, ensure_ascii=False, indent=2) if tc else "",
            # unused-in-op search fields
            "src_raw": "", "src_think": "", "tool_raw": "", "src_obs": "",
            "this_turn": highlight_neg(rw.get("this_turn", "")),
            "rationale": highlight_neg(rw.get("next_action_rationale", "")),
            "action": rw.get("action_description", ""),
            "cumulative": rw.get("cumulative_progress", ""),
            "remaining": json.dumps(rw.get("remaining_subgoals", []), ensure_ascii=False, indent=2) if rw.get("remaining_subgoals") else "",
            "task_analysis": rw.get("task_analysis", ""),
            "answer": rw.get("answer", "") or (r.get("answer") if r.get("is_final") else ""),
            "rw_err": r.get("rewrite_error", ""),
        })
    return steps, task, data_source, "", neg_count


@app.route("/")
def index():
    # Source filter: all | redsearcher | openseeker | molmoweb | webchain
    src_filter = request.args.get("src", "all").strip().lower()
    if src_filter in ("redsearcher", "openseeker", "molmoweb", "webchain"):
        ids = [t for t in TRAJ_IDS if t.startswith(src_filter)]
    else:
        ids = TRAJ_IDS
    if not ids:
        ids = TRAJ_IDS

    tid_arg = request.args.get("tid", "").strip()
    if tid_arg and (tid_arg in SK_INDEX or tid_arg in OP_INDEX):
        tid = tid_arg
        if tid not in ids:
            ids = TRAJ_IDS
        idx = ids.index(tid)
    else:
        if request.args.get("random"):
            idx = random.randint(0, len(ids) - 1)
        else:
            idx = int(request.args.get("idx", 0))
        idx = max(0, min(idx, len(ids) - 1))
        tid = ids[idx]

    is_op = tid in OP_INDEX
    if is_op:
        steps, task, data_source, gt, neg_count = _build_op_steps(tid)
    else:
        steps, task, data_source, gt, neg_count = _build_search_steps(tid)

    return render_template_string(
        HTML, traj_id=tid, data_source=data_source, task=task, gt=gt, n_steps=len(steps),
        neg_count=neg_count, steps=steps, idx=idx, total=len(ids), src_filter=src_filter,
        is_op=is_op,
        prev_idx=max(0, idx - 1), next_idx=min(len(ids) - 1, idx + 1),
        n_red=N_RED, n_os=N_OS, n_mw=N_MW, n_wc=N_WC,
    )


N_RED = 0
N_OS = 0
N_MW = 0
N_WC = 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", type=int, default=8088)
    ap.add_argument(
        "--skeleton",
        default="data/redsearcher_skeleton_correct.jsonl,data/openseeker_skeleton_correct.jsonl",
        help="comma-separated SEARCH skeleton files (joined with --rewrite)",
    )
    ap.add_argument("--rewrite", default="data/hybrid_search_deepseek_v4_flash_full.jsonl",
                    help="SEARCH rewrite file (full, with source_traj_id)")
    ap.add_argument(
        "--op",
        default="data/hybrid_op_molmoweb_kimi26_8000_full_20260529.jsonl,data/hybrid_op_webchain_v2.jsonl",
        help="comma-separated OPERATION rewrite files (molmoweb/webchain, self-contained)",
    )
    args = ap.parse_args()

    os.chdir(Path(__file__).resolve().parent.parent)
    SK_PATHS = [p.strip() for p in args.skeleton.split(",") if p.strip() and Path(p.strip()).exists()]
    RW_PATH = args.rewrite
    OP_FILES = [p.strip() for p in args.op.split(",") if p.strip() and Path(p.strip()).exists()]

    # search-mode index (join)
    for sp in SK_PATHS:
        print(f"[viewer] indexing search skeleton={sp} ...")
        for tid, offs in build_index(sp).items():
            SK_INDEX.setdefault(tid, (sp, offs))
    if RW_PATH and Path(RW_PATH).exists():
        print(f"[viewer] indexing search rewrite={RW_PATH} ...")
        RW_INDEX = build_index(RW_PATH)
    search_ids = set(SK_INDEX) & set(RW_INDEX)

    # operation-mode index (single file, only keep rows that actually have a rewrite)
    for opf in OP_FILES:
        print(f"[viewer] indexing op rewrite={opf} ...")
        for tid, offs in build_index(opf).items():
            OP_INDEX.setdefault(tid, (opf, offs))
    op_ids = set(OP_INDEX)

    TRAJ_IDS = sorted(search_ids | op_ids)
    N_RED = sum(1 for t in TRAJ_IDS if t.startswith("redsearcher"))
    N_OS = sum(1 for t in TRAJ_IDS if t.startswith("openseeker"))
    N_MW = sum(1 for t in TRAJ_IDS if t.startswith("molmoweb"))
    N_WC = sum(1 for t in TRAJ_IDS if t.startswith("webchain"))
    random.seed(0)
    random.shuffle(TRAJ_IDS)
    print(f"[viewer] {len(TRAJ_IDS)} trajs "
          f"(redsearcher={N_RED}, openseeker={N_OS}, molmoweb={N_MW}, webchain={N_WC})")
    print(f"[viewer] http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)

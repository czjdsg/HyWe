import argparse
import html
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render HybridWebAgent trajectories into a standalone HTML page."
    )
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", default="")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def resolve_output_path(input_path: Path, output_path: str) -> Path:
    if output_path:
        return Path(output_path).resolve()
    return input_path.with_suffix(".html")


def code_block(text: str, extra_class: str = "") -> str:
    safe_text = html.escape("" if text is None else str(text))
    return f'<pre class="code {extra_class}">{safe_text}</pre>'


def badge(text: str, kind: str = "default") -> str:
    return f'<span class="badge badge-{kind}">{html.escape(str(text))}</span>'


def image_card(title: str, img_src: str) -> str:
    if not img_src:
        return (
            '<div class="image-card empty-card">'
            f"<h4>{html.escape(title)}</h4>"
            '<div class="empty">No image for this step.</div>'
            "</div>"
        )
    return (
        '<div class="image-card">'
        f"<h4>{html.escape(title)}</h4>"
        f'<img src="{html.escape(img_src)}" alt="{html.escape(title)}" loading="lazy" />'
        "</div>"
    )


def render_summary_trace(summary_trace: list[dict] | None, tool_call: dict | None) -> str:
    if not tool_call or tool_call.get("name") != "visit":
        return ""

    if summary_trace is None:
        return (
            '<section class="panel">'
            "<h4>Summary Model Trace</h4>"
            '<div class="empty">This run did not record summary-model trajectory data.</div>'
            "</section>"
        )

    if not summary_trace:
        return (
            '<section class="panel">'
            "<h4>Summary Model Trace</h4>"
            '<div class="empty">No summary-model calls were recorded for this visit step.</div>'
            "</section>"
        )

    cards = []
    for index, item in enumerate(summary_trace, 1):
        meta = {
            "url": item.get("url", ""),
            "goal": item.get("goal", ""),
            "status": item.get("status", ""),
            "chunk_index": item.get("chunk_index", None),
            "chunk_count": item.get("chunk_count", None),
            "image_count": item.get("image_count", 0),
            "image_meta": item.get("image_meta", []),
            "truncation_note": item.get("truncation_note", ""),
            "raw_content_chars": item.get("raw_content_chars", None),
            "raw_content_tokens": item.get("raw_content_tokens", None),
            "summary_parsed_json": item.get("summary_parsed_json", None),
        }
        cards.append(
            '<details class="nested-card">'
            f'<summary>Summary Call {index} | {html.escape(str(item.get("status", "unknown")))}</summary>'
            f'<section class="meta-grid">'
            f'<div class="panel"><h4>Summary Meta</h4>{code_block(json.dumps(meta, ensure_ascii=False, indent=2))}</div>'
            f'<div class="panel"><h4>Fetch Result / Notes</h4>{code_block(item.get("fetch_result", "") or item.get("truncation_note", ""), "compact")}</div>'
            "</section>"
            f'<section class="meta-grid">'
            f'<div class="panel"><h4>Summary Prompt</h4>{code_block(item.get("summary_prompt_text", ""))}</div>'
            f'<div class="panel"><h4>Summary Output</h4>{code_block(item.get("summary_response_text", ""))}</div>'
            "</section>"
            f'<div class="panel"><h4>Raw Visit Content</h4>{code_block(item.get("raw_content", ""), "compact")}</div>'
            "</details>"
        )

    return (
        '<section class="panel">'
        "<h4>Summary Model Trace</h4>"
        f'{"".join(cards)}'
        "</section>"
    )


def step_image_paths(record: dict, step_record: dict, output_path: Path) -> tuple[str, str]:
    img_path = Path(step_record.get("img_path", "")) if step_record.get("img_path") else None
    full_src = ""
    if img_path and img_path.exists():
        full_src = str(img_path.relative_to(output_path.parent)).replace("\\", "/")
    visit_path = (
        Path(step_record.get("visit_screenshot_path", ""))
        if step_record.get("visit_screenshot_path")
        else None
    )
    visit_src = ""
    if visit_path and visit_path.exists():
        visit_src = str(visit_path.relative_to(output_path.parent)).replace("\\", "/")
    return full_src, visit_src


def render_step(record: dict, step_record: dict, output_path: Path) -> str:
    tool_call = step_record.get("tool_call")
    answer = step_record.get("answer", "")
    full_src, visit_src = step_image_paths(record, step_record, output_path)

    summary_bits = [f"Step {step_record.get('step', '?')}"]
    if tool_call:
        summary_bits.append(f"Tool: {tool_call.get('name', 'unknown')}")
    if answer:
        summary_bits.append("Final Answer")
    current_url = step_record.get("current_url", "")
    if current_url:
        summary_bits.append(current_url)

    usage_sections = [
        '<section class="step-top">',
        image_card("Screenshot", full_src),
        image_card("Visit Screenshot", visit_src),
        "</section>",
        '<section class="meta-grid">',
        f'<div class="panel"><h4>URL</h4>{code_block(step_record.get("current_url", ""), "compact")}</div>',
        f'<div class="panel"><h4>Title</h4>{code_block(step_record.get("current_title", ""), "compact")}</div>',
        "</section>",
        '<section class="meta-grid">',
        f'<div class="panel"><h4>Prompt</h4>{code_block(step_record.get("prompt_text", ""))}</div>',
        f'<div class="panel"><h4>Model Output</h4>{code_block(step_record.get("raw_response", ""))}</div>',
        "</section>",
    ]

    if step_record.get("report"):
        usage_sections.append(
            f'<section class="panel"><h4>Parsed Report</h4>{code_block(step_record.get("report", ""))}</section>'
        )

    if tool_call:
        usage_sections.extend(
            [
                '<section class="meta-grid">',
                f'<div class="panel"><h4>Tool Call</h4>{code_block(json.dumps(tool_call, ensure_ascii=False, indent=2))}</div>',
                f'<div class="panel"><h4>Observation</h4>{code_block(step_record.get("observation", ""))}</div>',
                "</section>",
            ]
        )
    else:
        usage_sections.append(
            f'<section class="panel"><h4>Observation</h4>{code_block(step_record.get("observation", ""))}</section>'
        )

    summary_trace = step_record.get("summary_trace") if "summary_trace" in step_record else None
    summary_trace_section = render_summary_trace(summary_trace, tool_call)
    if summary_trace_section:
        usage_sections.append(summary_trace_section)

    if answer:
        usage_sections.append(
            f'<section class="panel"><h4>Final Answer</h4>{code_block(answer)}</section>'
        )

    return (
        '<details class="step-card">'
        f'<summary>{" | ".join(html.escape(item) for item in summary_bits)}</summary>'
        f'{"".join(usage_sections)}'
        "</details>"
    )


def render_record(record: dict, index: int, output_path: Path) -> str:
    records = record.get("records", [])
    usage = record.get("usage", {})
    usage_text = json.dumps(usage, ensure_ascii=False, indent=2)
    body = [
        '<article class="run-card">',
        f'<h2>Sample {index} | {html.escape(record.get("termination", "unknown"))} | {len(records)} steps</h2>',
        '<section class="meta-grid">',
        f'<div class="panel"><h4>Task ID</h4>{code_block(record.get("task_id", ""), "compact")}</div>',
        f'<div class="panel"><h4>Question</h4>{code_block(record.get("question", ""), "compact")}</div>',
        "</section>",
        '<section class="meta-grid">',
        f'<div class="panel"><h4>Prediction</h4>{code_block(record.get("prediction", ""))}</div>',
        f'<div class="panel"><h4>Usage</h4>{code_block(usage_text)}</div>',
        "</section>",
        '<section class="step-list">',
        "".join(render_step(record, step_record, output_path) for step_record in records),
        "</section>",
        "</article>",
    ]
    return "".join(body)


def build_html(records: list[dict], input_path: Path, output_path: Path) -> str:
    body = "".join(render_record(record, index + 1, output_path) for index, record in enumerate(records))
    answer_count = sum(1 for record in records if record.get("termination") == "answer")
    title = f"HybridWebAgent Trajectory Viewer - {input_path.name}"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f5f2ea;
      --panel: #fffdf8;
      --line: #d6cdbf;
      --ink: #1f1c18;
      --muted: #6a6257;
      --accent: #9e3323;
      --accent-soft: #f3d7cf;
      --blue: #295dcc;
      --green: #0f766e;
      --shadow: 0 14px 36px rgba(57, 42, 16, 0.08);
      --radius: 18px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(158, 51, 35, 0.16), transparent 26%),
        linear-gradient(180deg, #fbf8f1 0%, var(--bg) 65%, #efe6da 100%);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    main {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 32px 24px 56px;
    }}
    header {{
      padding: 24px 28px;
      border-radius: 28px;
      background: rgba(255, 253, 248, 0.88);
      border: 1px solid rgba(255,255,255,0.52);
      box-shadow: var(--shadow);
      margin-bottom: 24px;
    }}
    h1, h2, h3, h4 {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      font-weight: 600;
    }}
    h1 {{ font-size: 2rem; margin-bottom: 10px; }}
    h2 {{ font-size: 1.35rem; margin-bottom: 16px; }}
    h4 {{ font-size: 1rem; margin-bottom: 10px; }}
    p {{ margin: 0; color: var(--muted); }}
    .run-card, .panel, .step-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }}
    .run-card {{
      padding: 22px;
      margin-bottom: 24px;
    }}
    .meta-grid, .step-top {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 16px;
    }}
    .panel {{
      padding: 16px;
      overflow: hidden;
    }}
    .step-list {{
      display: grid;
      gap: 14px;
    }}
    .step-card summary {{
      cursor: pointer;
      padding: 16px 18px;
      font-weight: 600;
      list-style: none;
    }}
    .step-card[open] summary {{
      border-bottom: 1px solid var(--line);
      background: linear-gradient(90deg, rgba(158, 51, 35, 0.08), rgba(41, 93, 204, 0.05));
      border-top-left-radius: var(--radius);
      border-top-right-radius: var(--radius);
    }}
    .step-card > section {{
      padding: 16px 18px 0;
    }}
    .step-card > section:last-child {{
      padding-bottom: 18px;
    }}
    .nested-card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fcfaf4;
      margin-top: 12px;
      overflow: hidden;
    }}
    .nested-card summary {{
      cursor: pointer;
      padding: 12px 14px;
      font-weight: 600;
      list-style: none;
      background: #f3ede3;
    }}
    .nested-card > section {{
      padding: 14px;
    }}
    .code {{
      margin: 0;
      padding: 14px;
      border-radius: 14px;
      background: #171512;
      color: #f9f6ee;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 0.9rem;
    }}
    .compact {{
      max-height: 220px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.82rem;
      font-weight: 600;
      margin-right: 8px;
    }}
    .badge-default {{ background: #ece7dd; color: #4d4538; }}
    .badge-answer {{ background: #d7f3ed; color: #0f766e; }}
    .badge-run {{ background: var(--accent-soft); color: var(--accent); }}
    .image-card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      background: #f9f4ea;
      padding: 14px;
    }}
    .image-card img {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 12px;
      border: 1px solid #d0c4b0;
      background: white;
    }}
    .empty-card {{
      display: flex;
      flex-direction: column;
      justify-content: center;
      min-height: 160px;
    }}
    .empty {{
      color: var(--muted);
      font-style: italic;
    }}
    @media (max-width: 960px) {{
      .meta-grid, .step-top {{
        grid-template-columns: 1fr;
      }}
      main {{
        padding: 20px 14px 40px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>{html.escape(title)}</h1>
      <p>{html.escape(str(input_path))}</p>
      <p>{badge(f"{len(records)} samples", "run")} {badge(f"{answer_count} answered", "answer")}</p>
    </header>
    {body}
  </main>
</body>
</html>"""


def main():
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    output_path = resolve_output_path(input_path, args.output_path)
    records = load_jsonl(input_path)
    if args.limit > 0:
        records = records[: args.limit]
    html_text = build_html(records, input_path, output_path)
    output_path.write_text(html_text, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()

import argparse
import html
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render an Online-Mind2Web evaluation result into a standalone HTML page."
    )
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", default="")
    return parser.parse_args()


def resolve_output_path(input_path: Path, output_path: str) -> Path:
    if output_path:
        return Path(output_path).resolve()
    return input_path.with_suffix(".html")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def maybe_load_agent_result(eval_record: dict) -> dict | None:
    task_dir = eval_record.get("task_dir", "")
    if not task_dir:
        return None
    result_path = Path(task_dir) / "result.json"
    if not result_path.exists():
        return None
    with result_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def code_block(text: str, extra_class: str = "") -> str:
    safe_text = html.escape("" if text is None else str(text))
    return f'<pre class="code {extra_class}">{safe_text}</pre>'


def badge(text: str, kind: str = "default") -> str:
    return f'<span class="badge badge-{kind}">{html.escape(str(text))}</span>'


def infer_step_kind(action: str) -> str:
    name = ((action or "").strip().split(" ", 1)[0] or "").lower()
    if name in {"search", "visit"}:
        return name
    if name == "answer":
        return "answer"
    if not name or name == "{}":
        return "browser"
    return "browser"


def rel_src(path_str: str, output_path: Path) -> str:
    if not path_str:
        return ""
    path = Path(path_str)
    if not path.exists():
        return ""
    return str(path.relative_to(output_path.parent)).replace("\\", "/")


def image_card(title: str, img_src: str) -> str:
    if not img_src:
        return (
            '<div class="image-card empty-card">'
            f"<h4>{html.escape(title)}</h4>"
            '<div class="empty">No image attached.</div>'
            "</div>"
        )
    return (
        '<div class="image-card">'
        f"<h4>{html.escape(title)}</h4>"
        f'<img src="{html.escape(img_src)}" alt="{html.escape(title)}" loading="lazy" />'
        "</div>"
    )


def selected_final_images(eval_record: dict, output_path: Path) -> list[tuple[str, str]]:
    images = []
    seen = set()
    for item in eval_record.get("step_judge_record", []):
        if infer_step_kind(item.get("action", "")) != "browser":
            continue
        step = item.get("step", "?")
        action = item.get("action", "")
        for index, path_str in enumerate(item.get("image_paths") or [], 1):
            src = rel_src(path_str, output_path)
            if not src or src in seen:
                continue
            seen.add(src)
            images.append((f"Final Prompt Image | Step {step} | {action[:60] or '[no action]'} | Img {index}", src))
    return images


def parse_status(eval_record: dict) -> str:
    response = eval_record.get("evaluation_response", "")
    match = re.search(r'Status:\s*"?(success|failure)"?', response, flags=re.IGNORECASE)
    return (match.group(1).lower() if match else "unknown")


def summary_block(eval_record: dict, agent_record: dict | None) -> str:
    status = parse_status(eval_record)
    kind = "success" if status == "success" else "danger" if status == "failure" else "default"
    final_answer = ""
    if agent_record:
        final_answer = (
            agent_record.get("prediction")
            or agent_record.get("final_result_response")
            or agent_record.get("answer")
            or ""
        )
    fields = {
        "Task Dir": eval_record.get("task_dir", ""),
        "Model": eval_record.get("model", ""),
        "Mode": eval_record.get("mode", ""),
        "Predicted Label": eval_record.get("predicted_label", ""),
        "Selected Steps": eval_record.get("selected_step_count", 0),
        "Score Threshold": eval_record.get("score_threshold", ""),
    }
    field_html = "".join(
        f'<div class="panel"><h4>{html.escape(key)}</h4>{code_block(value, "compact")}</div>'
        for key, value in fields.items()
    )
    extra = ""
    if final_answer:
        extra = f'<section class="panel"><h4>Agent Final Answer</h4>{code_block(final_answer)}</section>'
    return (
        '<section class="hero">'
        f"<h1>Evaluation Viewer {badge(status, kind)}</h1>"
        f"<p>{html.escape(eval_record.get('task', ''))}</p>"
        '<section class="meta-grid">'
        f"{field_html}"
        "</section>"
        f"{extra}"
        "</section>"
    )


def render_top_sections(eval_record: dict, agent_record: dict | None, output_path: Path) -> str:
    final_prompt_images = selected_final_images(eval_record, output_path)
    final_prompt_image_section = ""
    if final_prompt_images:
        cards = "".join(image_card(title, src) for title, src in final_prompt_images)
        final_prompt_image_section = (
            '<section class="panel">'
            '<h3>Final Judge Attached Images</h3>'
            '<section class="step-top">'
            f'{cards}'
            '</section>'
            '</section>'
        )
    parts = [
        summary_block(eval_record, agent_record),
        '<section class="meta-grid">',
        f'<div class="panel"><h3>Key Points</h3>{code_block(eval_record.get("key_points", ""))}</div>',
        f'<div class="panel"><h3>Final Judge Output</h3>{code_block(eval_record.get("evaluation_response", ""))}</div>',
        "</section>",
        '<section class="meta-grid">',
        f'<div class="panel"><h3>Final Judge System Prompt</h3>{code_block(eval_record.get("system_msg", ""))}</div>',
        f'<div class="panel"><h3>Final Judge User Input</h3>{code_block(eval_record.get("input_text", ""))}</div>',
        "</section>",
        final_prompt_image_section,
    ]
    return "".join(parts)


def render_step(step_record: dict, output_path: Path) -> str:
    step = step_record.get("step", "?")
    action = step_record.get("action", "")
    score = step_record.get("Score", 0)
    step_kind = infer_step_kind(action)
    image_paths = step_record.get("image_paths") or []
    cards = []
    for index, path_str in enumerate(image_paths[:3], 1):
        cards.append(image_card(f"Image {index}", rel_src(path_str, output_path)))
    if not cards:
        cards.append(image_card("Step Image", ""))

    return (
        '<details class="step-card">'
        f'<summary>Step {html.escape(str(step))} | {html.escape(step_kind)} | score={html.escape(str(score))} | {html.escape(action[:120] or "[no action]")}</summary>'
        '<section class="step-top">'
        f'{"".join(cards)}'
        "</section>"
        '<section class="meta-grid">'
        f'<div class="panel"><h4>Action</h4>{code_block(action, "compact")}</div>'
        f'<div class="panel"><h4>Judge Score</h4>{code_block(score, "compact")}</div>'
        "</section>"
        '<section class="meta-grid">'
        f'<div class="panel"><h4>Observation</h4>{code_block(step_record.get("observation", ""))}</div>'
        f'<div class="panel"><h4>Agent Report</h4>{code_block(step_record.get("thought", ""))}</div>'
        "</section>"
        f'<section class="panel"><h4>Step Judge Output</h4>{code_block(step_record.get("Response", ""))}</section>'
        "</details>"
    )


def build_html(eval_record: dict, output_path: Path) -> str:
    agent_record = maybe_load_agent_result(eval_record)
    step_html = "".join(render_step(item, output_path) for item in eval_record.get("step_judge_record", []))
    title = f"Eval Viewer - {output_path.stem}"
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
      --red: #b42318;
      --shadow: 0 14px 36px rgba(57, 42, 16, 0.08);
      --radius: 18px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top left, #fef7ed, #f5f2ea 44%, #efe7da);
      color: var(--ink);
      line-height: 1.5;
    }}
    .page {{
      width: min(1500px, calc(100vw - 40px));
      margin: 0 auto;
      padding: 28px 0 48px;
    }}
    .hero, .panel, .step-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }}
    .hero {{
      padding: 24px;
      margin-bottom: 18px;
    }}
    h1, h2, h3, h4 {{
      margin: 0 0 10px;
      line-height: 1.2;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}
    .panel {{
      padding: 16px;
      min-width: 0;
    }}
    .step-list {{
      display: grid;
      gap: 14px;
    }}
    .step-card {{
      overflow: hidden;
    }}
    .step-card > summary {{
      cursor: pointer;
      list-style: none;
      padding: 16px 18px;
      font-weight: 700;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(120deg, rgba(158, 51, 35, 0.06), rgba(41, 93, 204, 0.04));
    }}
    .step-card[open] > summary {{
      border-bottom-color: var(--accent-soft);
    }}
    .step-top {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      padding: 16px;
      border-bottom: 1px solid var(--line);
    }}
    .image-card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      background: #fff;
    }}
    .image-card h4 {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
      color: var(--muted);
    }}
    .image-card img {{
      display: block;
      width: 100%;
      height: auto;
      background: #f2ede4;
    }}
    .empty-card .empty {{
      padding: 24px 16px;
      color: var(--muted);
      text-align: center;
    }}
    .code {{
      margin: 0;
      padding: 12px 14px;
      background: #171717;
      color: #f9fafb;
      border-radius: 14px;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 13px;
      max-height: 680px;
    }}
    .compact {{
      max-height: none;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      margin-left: 10px;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      background: #ede7dc;
      color: var(--muted);
    }}
    .badge-success {{ background: rgba(15, 118, 110, 0.12); color: var(--green); }}
    .badge-danger {{ background: rgba(180, 35, 24, 0.12); color: var(--red); }}
    .badge-default {{ background: #ede7dc; color: var(--muted); }}
    @media (max-width: 720px) {{
      .page {{ width: min(100vw - 20px, 100%); padding-top: 18px; }}
      .hero {{ padding: 18px; }}
      .step-card > summary {{ padding: 14px; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    {render_top_sections(eval_record, agent_record, output_path)}
    <section class="panel">
      <h2>Step Judge Trace</h2>
      <div class="step-list">
        {step_html}
      </div>
    </section>
  </main>
</body>
</html>"""


def main():
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    output_path = resolve_output_path(input_path, args.output_path)
    eval_record = load_json(input_path)
    html_text = build_html(eval_record, output_path)
    output_path.write_text(html_text, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()

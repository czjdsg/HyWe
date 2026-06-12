import argparse
import html
from pathlib import Path

from visualize_trajectory import (
    badge,
    load_jsonl,
    render_record,
    resolve_output_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render paired HybridWebAgent trajectories for with-web vs no-web comparison."
    )
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", default="")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def pair_records(records: list[dict]) -> list[tuple[str, dict | None, dict | None]]:
    grouped: dict[str, dict[str, dict]] = {}
    for record in records:
        task_id = record.get("task_id", "")
        base_id = task_id[:-7] if task_id.endswith("_no_web") else task_id
        variant = "no_web" if task_id.endswith("_no_web") else "with_web"
        grouped.setdefault(base_id, {})[variant] = record

    paired = []
    for base_id in sorted(grouped):
        paired.append((base_id, grouped[base_id].get("with_web"), grouped[base_id].get("no_web")))
    return paired


def build_html(records: list[dict], input_path: Path, output_path: Path) -> str:
    pairs = pair_records(records)
    if len(pairs) == 0:
        body = '<div class="empty">No paired records found.</div>'
    else:
        sections = []
        for index, (base_id, with_web, no_web) in enumerate(pairs, 1):
            sections.append(
                '<section class="pair-card">'
                f'<div class="pair-head"><h2>Pair {index} | {html.escape(base_id)}</h2></div>'
                '<div class="pair-grid">'
                '<div class="pair-col">'
                f'<h3>With Web</h3>{render_record(with_web, index, output_path) if with_web else "<div class=\"empty\">Missing with-web result.</div>"}'
                '</div>'
                '<div class="pair-col">'
                f'<h3>No Web</h3>{render_record(no_web, index, output_path) if no_web else "<div class=\"empty\">Missing no-web result.</div>"}'
                '</div>'
                '</div>'
                '</section>'
            )
        body = "".join(sections)

    with_web_answer = sum(
        1
        for _, with_web, _ in pairs
        if with_web and with_web.get("termination") == "answer"
    )
    no_web_answer = sum(
        1
        for _, _, no_web in pairs
        if no_web and no_web.get("termination") == "answer"
    )
    title = f"HybridWebAgent Paired Viewer - {input_path.name}"

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
      max-width: 1900px;
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
    h1, h2, h3 {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      font-weight: 600;
    }}
    h1 {{ font-size: 2rem; margin-bottom: 10px; }}
    h2 {{ font-size: 1.25rem; }}
    h3 {{ font-size: 1.05rem; margin-bottom: 12px; }}
    p {{ margin: 0; color: var(--muted); }}
    .pair-card, .run-card, .panel, .step-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }}
    .pair-card {{
      padding: 18px;
      margin-bottom: 24px;
    }}
    .pair-head {{
      margin-bottom: 14px;
    }}
    .pair-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      align-items: start;
    }}
    .pair-col {{
      min-width: 0;
    }}
    .empty {{
      color: var(--muted);
      font-style: italic;
      padding: 16px;
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
    .run-card {{
      padding: 18px;
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
      background: linear-gradient(90deg, rgba(158, 51, 35, 0.08), rgba(15, 118, 110, 0.05));
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
    @media (max-width: 1200px) {{
      .pair-grid, .meta-grid, .step-top {{
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
      <p>
        {badge(f"{len(pairs)} paired tasks", "run")}
        {badge(f"with-web answered {with_web_answer}", "answer")}
        {badge(f"no-web answered {no_web_answer}", "answer")}
      </p>
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

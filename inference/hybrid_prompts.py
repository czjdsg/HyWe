import copy
import json


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Run a web search; return result snippets (title, url, snippet, source, date).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "One or more search queries.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scholar",
            "description": "Search academic sources; return titles, publication info, citation counts, snippets, source URL, and PDF URL when available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "One or more academic queries.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visit",
            "description": "Read one or more webpages or PDFs via a reader-style extractor; return goal-relevant text. Does not change browser state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "oneOf": [
                            {"type": "string", "description": "A webpage URL."},
                            {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "description": "A list of webpage URLs.",
                            },
                        ],
                        "description": "URL or list of URLs to read. If omitted, reads the current browser URL.",
                    },
                    "goal": {
                        "type": "string",
                        "description": "What to extract from the page(s).",
                    },
                    "parse_type": {
                        "type": "string",
                        "enum": ["auto", "html", "pdf"],
                        "default": "auto",
                        "description": "auto = detect, html = webpage, pdf = PDF.",
                    },
                },
                "required": ["goal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_sandbox",
            "description": "Run self-contained code in an isolated sandbox; returns stdout, stderr, status, timing. For arithmetic, parsing, validation, or attachment processing. Does not access the browser.",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "default": "python",
                        "description": "python (default), bash, nodejs, cpp, java, rust, go, pytest, sql, lean, verilog, etc.",
                    },
                    "code": {
                        "type": "string",
                        "description": "Self-contained code to execute.",
                    },
                    "stdin": {
                        "type": "string",
                        "description": "Optional stdin.",
                    },
                    "run_timeout": {
                        "type": "number",
                        "default": 10,
                        "description": "Max runtime seconds (default 10).",
                    },
                    "compile_timeout": {
                        "type": "number",
                        "default": 10,
                        "description": "Max compile seconds (default 10).",
                    },
                    "files": {
                        "type": "object",
                        "additionalProperties": {
                            "oneOf": [{"type": "string"}, {"type": "null"}]
                        },
                        "description": "Optional {path: base64} mapping. Task attachments are auto-injected; do not inline them here.",
                    },
                    "fetch_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File paths to fetch back after execution.",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click at (x, y) on the visible target element.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X coordinate of the target element."},
                    "y": {"type": "number", "description": "Y coordinate of the target element."},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "double_click",
            "description": "Double-click at (x, y). Use for opening files in lists, editing spreadsheet-style cells, expanding tree nodes, zooming images, or any UI where a single click is insufficient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X coordinate of the target element."},
                    "y": {"type": "number", "description": "Y coordinate of the target element."},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_option",
            "description": "For a native HTML <select> dropdown at (x, y), set its value to the given option label/text WITHOUT needing to open the dropdown popup. Faster and more reliable than click+click for native selects. If the element at (x, y) is NOT a native <select>, the runtime falls back to a regular click so you can then click the option from the open menu.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X coordinate of the <select> element."},
                    "y": {"type": "number", "description": "Y coordinate of the <select> element."},
                    "value": {"type": "string", "description": "The visible option label/text to select (matched case-insensitively, then by substring)."},
                },
                "required": ["x", "y", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hover",
            "description": "Hover at (x, y) to reveal menus, tooltips, or hidden controls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X coordinate of the target element."},
                    "y": {"type": "number", "description": "Y coordinate of the target element."},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fill",
            "description": "Click at (x, y) to focus an input/textarea, then type the given text. Set submit=true to press Enter at the end.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X coordinate of the target element."},
                    "y": {"type": "number", "description": "Y coordinate of the target element."},
                    "text": {"type": "string", "description": "Text to type."},
                    "submit": {"type": "boolean", "description": "Press Enter after typing. Defaults to false."},
                },
                "required": ["x", "y", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Scroll the main page (omit x/y) or scroll a scrollable region containing (x, y).",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down"],
                        "description": "Scroll direction.",
                    },
                    "x": {"type": "number", "description": "Omit for page scroll; provide X coordinate to scroll the scrollable region at that point."},
                    "y": {"type": "number", "description": "Omit for page scroll; provide Y coordinate to scroll the scrollable region at that point."},
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drag",
            "description": "Press at (from_x, from_y), hold, move to (to_x, to_y), then release. Use for scrollbar dragging, sliders/range inputs, list/card reordering, map panning, and drag-and-drop file upload targets — anything click/scroll cannot express.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_x": {"type": "number", "description": "Start X within viewport."},
                    "from_y": {"type": "number", "description": "Start Y within viewport."},
                    "to_x":   {"type": "number", "description": "End X within viewport."},
                    "to_y":   {"type": "number", "description": "End Y within viewport."},
                    "hold_ms": {"type": "integer", "description": "Optional ms to hold the mouse down before moving (default 0). Useful for elements with delayed drag-handles."},
                },
                "required": ["from_x", "from_y", "to_x", "to_y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait for page state to settle.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {
                        "type": "integer",
                        "description": "Wait seconds (default 5, max 30).",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "go_back",
            "description": "Navigate back in browser history.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "goto",
            "description": "Navigate the active browser tab to a URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Destination URL.",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "new_tab",
            "description": "Open a URL in a new browser tab and switch to it. Subsequent actions apply to the new tab.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Destination URL to open in a new tab.",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_tab",
            "description": "Switch the active tab to the given index. Subsequent screenshots and actions apply to the new tab. Tab indices appear in the Current Browser State block.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "0-based tab index.",
                    }
                },
                "required": ["index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_tab",
            "description": "Close the tab at the given index. If active, the most recent remaining tab becomes active. Cannot close the last tab.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "0-based tab index.",
                    }
                },
                "required": ["index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "press_key",
            "description": "Press a key (e.g. \"Escape\", \"Enter\", \"Tab\", \"ArrowDown\") or a chord (e.g. [\"Control\", \"F\"]) on the focused element. Names follow Playwright key conventions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "Key names. The last is the trigger; preceding keys are held as modifiers.",
                    }
                },
                "required": ["keys"],
            },
        },
    },
]


TOOL_STR = json.dumps(TOOLS, ensure_ascii=False, indent=2)


SWITCH_TOOL = {
    "type": "function",
    "function": {
        "name": "switch",
        "description": (
            "Hand off control to the OTHER path (search-based <-> operation-based). "
            "Consumes this turn -- NO browser action is performed. Use ONLY when the other path is clearly required for the next remaining_subgoal. "
            "Do NOT call switch as default exploration. Do NOT call switch just because you are uncertain. "
            "cumulative_progress and remaining_subgoals carry across the switch verbatim, so DO NOT restate context in the reason."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "One sentence: why the current path is stuck or why the other path is faster for the next remaining_subgoal.",
                },
            },
            "required": ["reason"],
        },
    },
}


SYSTEM_PROMPT_TEMPLATE = """You are a general-purpose web agent.

Available tools and JSON schemas:
{tool_str}

You can solve:
- information-seeking tasks
- website-operation tasks
- mixed tasks that require both reasoning and interaction

Coordinate-based interaction:
- For click, double_click, hover, fill, select_option, scroll, and drag: read the screenshot and output (x, y) coordinates pointing to the visual center of the target element (drag uses from_x/from_y/to_x/to_y).

Action descriptions are mandatory:
- Every tool_call you emit MUST be paired with a short `action_description` (5-15 words) naming the visible target element and what you expect to happen (e.g. "Click the red 'GET MY RATES' button at the top-right of the page" or "Type 'home workout' in the search box and submit"). The description is appended to the action history; future turns and reviewers use it to detect loops and recover from mistakes.

Before each step, decide whether to keep or switch the current route:
- search-based path, or
- operation-based path, or
- completed.

Path semantics:
- The search-based path is for efficient information gathering: search, scholar, and visit.
- The operation-based path is for browser interaction that reveals deeper webpage information unavailable through search-based tools: goto, new_tab, hover, click, double_click, fill, select_option, scroll, drag, wait, go_back, press_key, switch_tab, and close_tab.
- code_sandbox is path-neutral. You may use it from either path for calculations, parsing, data processing, attachment inspection, or validation. Using code_sandbox alone does not mean the path switched.

After choosing a path, the next tool must obey that path:
- If you choose search-based, use only search, scholar, or visit, unless using code_sandbox for computation/parsing/validation.
- If you choose operation-based, use only goto, new_tab, hover, click, double_click, fill, select_option, scroll, drag, wait, go_back, press_key, switch_tab, or close_tab, unless using code_sandbox for computation/parsing/validation.
- If you choose completed, do not call a tool; provide the answer.
- Re-evaluate whether to keep or switch paths at every step. Do not lock yourself into one route.

Path switching criteria:
- Prefer starting or staying on search-based when the next unknown can be found efficiently through search snippets, scholarly metadata, page/PDF text, or cross-source lookup.
- Switch from search-based to operation-based when you have identified the relevant site/page but the needed information is hidden behind dynamic UI, filters, forms, pagination, menus, map/list interactions, or other browser-only state.
- Prefer starting or staying on operation-based when the current page can reveal the needed information through interaction.
- Switch from operation-based to search-based when browser actions are not making progress, the current site is unusable, or the missing fact is likely available faster from search/visit.
- If code_sandbox is needed, still state which path is governing the task; code_sandbox itself is only an auxiliary tool.

Tool semantics matter:
- search finds candidate pages or external evidence.
- scholar finds academic papers, publication metadata, citation counts, and PDF links.
- visit reads webpage or PDF content for information gathering and does not change browser state.
- code_sandbox runs small self-contained code for computation, data processing, parsing, or code validation and does not change browser state. Task-provided attachment files are automatically available in the sandbox under the sandbox paths shown in the task; read those files directly instead of inlining file contents into the tool_call JSON.
- goto opens a URL in the active tab when you want page state, screenshots, or future interaction.
- new_tab opens a URL in a new tab and switches to it; use it when you want to keep the current page alive for later switch_tab.
- hover, click, double_click, fill, select_option, scroll, drag, wait, go_back, and press_key operate on the current browser page using (x, y) coordinates (where applicable).
- drag presses at (from_x, from_y), holds, moves to (to_x, to_y), then releases — use it for scrollbar dragging, slider/range inputs, list/card reordering, map panning, and drag-and-drop file upload areas.
- For native <select> dropdowns there is no dedicated tool: click the dropdown to open it, then click the option, or use press_key with ArrowDown/Enter.
- switch_tab and close_tab manage the open browser tabs. The full tab list is shown in the Current Browser State block on every turn, so you do not need a separate listing tool.

search, scholar, and visit are the search-based tools for broad information gathering and shallow webpage
or PDF content extraction. Browser-operation tools are for interacting with a specific
webpage and for collecting information that is only revealed through browser interaction.
code_sandbox is path-neutral and does not interact with the browser. search, scholar, and visit also do not interact with the browser; browser-operation tools do.
Therefore, when the
browser is still blank and no page screenshot is provided, no page-interaction tool is valid.
Use goto or new_tab first. Do not use hover, click, double_click, fill, select_option, scroll, drag, wait, go_back, or press_key
until the browser provides a current page screenshot.
switch_tab and close_tab may be used at any time, but are only useful when more
than one tab is open. The list of open tabs is shown in the Current Browser State block.

Your report must follow this decision flow:
1. First synthesize the current webpage state, visible evidence, previous observations, and any confirmed information.
2. Summarize task progress: what is already done and what remains.
3. Analyze whether to keep or switch the current path, then choose exactly one of: search-based, operation-based, or completed.
4. Only after that, choose the next tool and describe how to call it with concrete arguments allowed by the selected path. Always include an `action_description` describing the target element and expected outcome.

Do not decide the path from the page state alone.
Do not decide the path from prior facts alone.
Always combine both the accumulated information and the current web state before choosing whether to keep or switch paths.
Tip: Before choosing the next tool, inspect Recent Actions. If the recent actions repeat the same tool, same arguments, same page, or same query without progress, treat it as a loop and try a different method.

The finish condition depends on the task:
- For information-seeking tasks, you may finish when the collected evidence is sufficient to answer accurately.
- For website-operation tasks, you may finish when the required target page, target result, or target state has already been reached, even if no extra information gathering is needed.
- For mixed tasks, finish only when both the required interaction outcome and the required answer have been completed.

Use the search-based path when the next progress requires efficient information gathering:
- search
- scholar
- visit

Use the operation-based path when the next progress requires browser interaction, page-state change, keyboard action, or tab management:
- goto
- new_tab
- hover
- click
- fill
- scroll
- wait
- go_back
- press_key
- switch_tab
- close_tab

Use code_sandbox from either path only when the next progress requires calculation, local attachment parsing, data processing, or code validation.

You will receive:
- the task
- the previous report, previous observation, and recent actions
- the current browser URL and title
- the current screenshot

""".strip()


INITIAL_USER_PROMPT = """## Current Date
{current_date}

## Task
{task}

## Current Browser State
- URL: {current_url}
- Title: {current_title}
{tabs_text}
- Screenshot: attached below

## Required Output Format
Return EXACTLY ONE top-level JSON object as your entire response. No XML tags. No markdown fences. No prose outside the JSON.

This is the FIRST turn. Analyze the task, decompose it into sub-goals, then choose the first action.

Schema (every required key MUST be present; extra keys are forbidden):

{{
  "task_analysis":          "<Analyze the task and identify concrete sub-goals.>",
  "remaining_subgoals":     ["<atomic open item still required to finish the task>", "..."],
  "path_switch":            null | {{"to": "search-based" | "operation-based", "reason": "<short>"}},
  "next_action_rationale":  "<one sentence: WHY the chosen tool will close some remaining_subgoal>",
  "tool_call":              {{"name": "<tool name from the schemas above>", "arguments": {{...}}}},
  "action_description":     "<5-15 word short description of the action: name the visible target element and expected outcome>"
}}

Field rules:
- `task_analysis` is a paragraph decomposing the task into sub-goals. Required and non-empty.
- `remaining_subgoals` is a JSON array of strings listing all open items.
- `path_switch` is null when keeping the current path. Set it when switching.
- `next_action_rationale` is a non-empty one-sentence string stating why the chosen tool advances a remaining_subgoal.
- `tool_call` is required — choose the first tool to invoke based on the task analysis.
- `action_description` is a non-empty string (5-15 words).
- `tool_call.arguments` must use the exact English argument keys / enum values from the tool schema.

Rules:
- Once a path is selected, use only tools from that path: search/scholar/visit for search-based, or goto/new_tab/hover/click/fill/scroll/drag/wait/go_back/press_key/switch_tab/close_tab for operation-based. code_sandbox is path-neutral.
- Use exactly one tool call.
- For hover, click, fill, regional scroll, and drag, output (x, y) coordinates of the target element identified visually from the screenshot.
""".strip()


INITIAL_USER_PROMPT_NO_BROWSER = """## Current Date
{current_date}

## Task
{task}

No initial browser URL or page state is provided for this task.

## Required Output Format
Return EXACTLY ONE top-level JSON object as your entire response. No XML tags. No markdown fences. No prose outside the JSON.

This is the FIRST turn. Analyze the task, decompose it into sub-goals, then choose the first action.

Schema (every required key MUST be present; extra keys are forbidden):

{{
  "task_analysis":          "<Analyze the task and identify concrete sub-goals.>",
  "remaining_subgoals":     ["<atomic open item still required to finish the task>", "..."],
  "path_switch":            null | {{"to": "search-based" | "operation-based", "reason": "<short>"}},
  "next_action_rationale":  "<one sentence: WHY the chosen tool will close some remaining_subgoal>",
  "tool_call":              {{"name": "<tool name from the schemas above>", "arguments": {{...}}}},
  "action_description":     "<5-15 word short description of the action: name the target and expected outcome>"
}}

Field rules:
- `task_analysis` is a paragraph decomposing the task into sub-goals. Required and non-empty.
- `remaining_subgoals` is a JSON array of strings listing all open items.
- `path_switch` is null when keeping the current path. Set it when switching.
- `next_action_rationale` is a non-empty one-sentence string stating why the chosen tool advances a remaining_subgoal.
- `tool_call` is required — choose the first tool to invoke based on the task analysis.
- `action_description` is a non-empty string (5-15 words).
- `tool_call.arguments` must use the exact English argument keys / enum values from the tool schema.

Rules:
- Since no page is provided, do not assume any current page state.
- Select search-based if you need candidate pages, academic evidence, or page/PDF text.
- Select operation-based if you already know a page URL and need browser interaction; since the browser is blank, use goto first.
- Use code_sandbox from either path for calculation, parsing, data processing, or attachment inspection.
- Use exactly one tool call.
""".strip()


TURN_USER_PROMPT_NO_BROWSER = """## Current Date
{current_date}

## Task
{task}

## Previous Thought (the JSON object emitted last turn)
```json
{last_report}
```

## Recent Actions (up to 10 most recent tool calls)
{recent_actions}


## Previous Observation
<tool_response>
{last_observation}
</tool_response>

The browser does not currently have a useful loaded page. The current page is blank, so no screenshot or current page state is provided for this turn.

## Required Output Format
Return EXACTLY ONE top-level JSON object as your entire response. No XML tags. No markdown fences. No prose outside the JSON.

Generate the keys in the order shown — `this_turn` is your fresh analysis of THIS turn's observation, and `cumulative_progress` is built by READING the prior turn's `cumulative_progress` (from "Previous Thought" above) and walking through it POINT BY POINT — explicitly KEEPING items still true, MODIFYING/CORRECTING items now refined or superseded (state both old and new), ADDING new items contributed by `this_turn`, and DELETING items no longer true. Format is flexible prose or short bullet-like lines.

Schema (every required key MUST be present; extra keys are forbidden):

{{
  "this_turn":              "<one paragraph: what THIS turn newly contributed. May be a fact surfaced from the latest tool result, a state change just produced (e.g. \"clicked 'Add to bag', cart count 0→1\", \"page navigated from /search to /product/123\"), a confirmation, or a failure. Be specific.>",
  "cumulative_progress":    "<RUNNING summary of EVERYTHING accomplished so far that is relevant to the task. Carried across turns. ONLY write progress that has actually been made (facts confirmed, values collected, URLs identified, state changes that took effect). DO NOT write what is still missing, what remains to do, blockers, plans, or guesses — those belong in `remaining_subgoals` or `this_turn`, not here. Be COMPREHENSIVE — capture every task-relevant concrete fact and every confirmed state change, including small intermediate wins (filter applied successfully, correct page reached, dropdown configured, login completed). Build it by taking the prior turn's `cumulative_progress` (from \"Previous Thought\") and walking through it POINT BY POINT against `this_turn`: KEEP each prior item that is still true; MODIFY/CORRECT items now refined or superseded (state both the old and the new value); ADD any NEW items confirmed in `this_turn`; DELETE items the new evidence shows were wrong. Cover BOTH information progress (concrete facts/values/URLs collected) AND operation progress (cart additions, sign-ins, forms submitted, filters applied, bookings confirmed, navigation milestones reached). Each entry must describe something already done, not something pending.>",
  "remaining_subgoals":     ["<atomic open item still required to finish the task. Can be a fact to find OR a state target not yet reached (e.g. \"item still missing from cart\", \"booking form not yet submitted\")>", "..."],
  "path_switch":            null | {{"to": "search-based" | "operation-based", "reason": "<short>"}},
  "next_action_rationale":  "<one sentence: WHY the chosen tool will close some remaining_subgoal>",
  "tool_call":              {{"name": "<tool name from the schemas above>", "arguments": {{...}}}} OR null,
  "action_description":     "<5-15 word short description of the action: name the visible target element and the expected outcome. REQUIRED when tool_call is non-null; use null when answer is set instead.>",
  "answer":                 null OR "<final answer string if you are confident the task is solved/blocked and want to terminate THIS TURN early>"
}}

Field rules:
- `this_turn` and `cumulative_progress` are paragraph-style strings (free-form prose, may include short bullet-like lines). Both are required and non-empty.
- `remaining_subgoals` is a JSON array of strings (may be the empty array `[]` only when the task is fully solved).
- `path_switch` is null when keeping the current path. Set it to `{{"to": "search-based" | "operation-based", "reason": "<short>"}}` only when actually switching paths this turn. Do NOT default to a non-null value if you're staying.
- EXACTLY ONE of `tool_call` (object) and `answer` (non-empty string) must be non-null; the other MUST be null.
- Set `answer` (and `tool_call`=null) ONLY when you are confident the task is fully solved (or definitively blocked) and want to terminate the trajectory THIS turn. Otherwise, set `tool_call` (the next tool to invoke) and `answer`=null.
- When `tool_call` is set, it must be the structured tool-call JSON for ONE tool from the schemas listed in the system prompt.
- `tool_call.arguments` must use the exact English argument keys / enum values from the tool schema.
- When `tool_call` is non-null, `action_description` MUST be a non-empty string of 5-15 words naming the visible target element and the expected outcome. When `answer` is set (and `tool_call` is null), `action_description` MUST be null.

Rules:
- Since the current page is blank, do not assume any useful current page state.
- Since there is no loaded page, base the synthesis on previous progress and the absence of browser state.
- Select search-based if you need candidate pages, academic evidence, or page/PDF text via search, scholar, or visit.
- Select operation-based if you already know a page and need future browser interaction or page-state changes; since the browser is blank, the operation-based tool must be goto first.
- Use code_sandbox from either path only for calculation, parsing, data processing, attachment inspection, or code validation.
- search, scholar, visit, and code_sandbox do not interact with the browser. Browser operations require a loaded page with a current screenshot.
- Since the browser is blank, do not use hover, click, double_click, fill, select_option, scroll, drag, wait, go_back, or press_key. Use goto or new_tab first if page interaction is needed. (switch_tab and close_tab remain available but are only useful when more than one tab is open; the open-tab list is shown in the Current Browser State block.)
- Use exactly one tool call if not answering.
- For hover, click, fill, regional scroll, and drag, output (x, y) coordinates of the target element identified visually from the screenshot. drag takes (from_x, from_y, to_x, to_y).
""".strip()


TURN_USER_PROMPT = """## Current Date
{current_date}

## Task
{task}

## Previous Thought (the JSON object emitted last turn)
```json
{last_report}
```

## Recent Actions (up to 10 most recent tool calls)
{recent_actions}


## Previous Observation
<tool_response>
{last_observation}
</tool_response>

## Current Browser State
- URL: {current_url}
- Title: {current_title}
{tabs_text}
- Screenshot: attached below

## Required Output Format
Return EXACTLY ONE top-level JSON object as your entire response. No XML tags. No markdown fences. No prose outside the JSON.

Generate the keys in the order shown — `this_turn` is your fresh analysis of THIS turn's observation, and `cumulative_progress` is built by READING the prior turn's `cumulative_progress` (from "Previous Thought" above) and walking through it POINT BY POINT — explicitly KEEPING items still true, MODIFYING/CORRECTING items now refined or superseded (state both old and new), ADDING new items contributed by `this_turn`, and DELETING items no longer true. Format is flexible prose or short bullet-like lines.

Schema (every required key MUST be present; extra keys are forbidden):

{{
  "this_turn":              "<one paragraph: what THIS turn newly contributed. May be a fact surfaced from the latest tool result, a state change just produced (e.g. \"clicked 'Add to bag', cart count 0→1\", \"page navigated from /search to /product/123\"), a confirmation, or a failure. Be specific.>",
  "cumulative_progress":    "<RUNNING summary of EVERYTHING accomplished so far that is relevant to the task. Carried across turns. ONLY write progress that has actually been made (facts confirmed, values collected, URLs identified, state changes that took effect). DO NOT write what is still missing, what remains to do, blockers, plans, or guesses — those belong in `remaining_subgoals` or `this_turn`, not here. Be COMPREHENSIVE — capture every task-relevant concrete fact and every confirmed state change, including small intermediate wins (filter applied successfully, correct page reached, dropdown configured, login completed). Build it by taking the prior turn's `cumulative_progress` (from \"Previous Thought\") and walking through it POINT BY POINT against `this_turn`: KEEP each prior item that is still true; MODIFY/CORRECT items now refined or superseded (state both the old and the new value); ADD any NEW items confirmed in `this_turn`; DELETE items the new evidence shows were wrong. Cover BOTH information progress (concrete facts/values/URLs collected) AND operation progress (cart additions, sign-ins, forms submitted, filters applied, bookings confirmed, navigation milestones reached). Each entry must describe something already done, not something pending.>",
  "remaining_subgoals":     ["<atomic open item still required to finish the task. Can be a fact to find OR a state target not yet reached (e.g. \"item still missing from cart\", \"booking form not yet submitted\")>", "..."],
  "path_switch":            null | {{"to": "search-based" | "operation-based", "reason": "<short>"}},
  "next_action_rationale":  "<one sentence: WHY the chosen tool will close some remaining_subgoal>",
  "tool_call":              {{"name": "<tool name from the schemas above>", "arguments": {{...}}}} OR null,
  "action_description":     "<5-15 word short description of the action: name the visible target element and the expected outcome. REQUIRED when tool_call is non-null; use null when answer is set instead.>",
  "answer":                 null OR "<final answer string if you are confident the task is solved/blocked and want to terminate THIS TURN early>"
}}

Field rules:
- `this_turn` and `cumulative_progress` are paragraph-style strings (free-form prose, may include short bullet-like lines). Both are required and non-empty.
- `remaining_subgoals` is a JSON array of strings (may be the empty array `[]` only when the task is fully solved).
- `path_switch` is null when keeping the current path. Set it to `{{"to": "search-based" | "operation-based", "reason": "<short>"}}` only when actually switching paths this turn. Do NOT default to a non-null value if you're staying.
- EXACTLY ONE of `tool_call` (object) and `answer` (non-empty string) must be non-null; the other MUST be null.
- Set `answer` (and `tool_call`=null) ONLY when you are confident the task is fully solved (or definitively blocked) and want to terminate the trajectory THIS turn. Otherwise, set `tool_call` (the next tool to invoke) and `answer`=null.
- When `tool_call` is set, it must be the structured tool-call JSON for ONE tool from the schemas listed in the system prompt.
- `tool_call.arguments` must use the exact English argument keys / enum values from the tool schema.
- When `tool_call` is non-null, `action_description` MUST be a non-empty string of 5-15 words naming the visible target element and the expected outcome. When `answer` is set (and `tool_call` is null), `action_description` MUST be null.

Rules:
- Re-evaluate whether to keep or switch paths every step.
- Once a path is selected, use only tools from that path: search/scholar/visit for search-based, or goto/new_tab/hover/click/fill/scroll/drag/wait/go_back/press_key/switch_tab/close_tab for operation-based. code_sandbox is path-neutral and may be used from either path.
- Use exactly one tool call if not answering.
- Use search-based when efficient information gathering is needed.
- Use operation-based when browser interaction is needed to reveal information that search-based tools cannot obtain.
- visit does not change browser state; use goto when you need browser interaction on a URL.
- Do not keep retrying the same failed action without a real change in strategy.
- For hover, click, fill, regional scroll, and drag, output (x, y) coordinates of the target element identified visually from the screenshot. drag takes (from_x, from_y, to_x, to_y).
""".strip()


FINAL_USER_PROMPT_NO_BROWSER = """## Current Date
{current_date}

## Task
{task}

## Previous Thought (the JSON object emitted last turn)
```json
{last_report}
```

## Recent Actions (up to 10 most recent tool calls)
{recent_actions}


## Previous Observation
<tool_response>
{last_observation}
</tool_response>

The browser does not currently have a useful loaded page. The current page is blank, so no screenshot or current page state is provided for this turn.

You have reached the final step budget. You are not allowed to take more tool actions.

## Required Output Format
This is the FINAL turn. Return EXACTLY ONE top-level JSON object as your entire response. No XML tags. No markdown fences. No prose outside the JSON.

Generate the keys in the order shown — `this_turn` is your fresh analysis; `cumulative_progress` is built by walking the prior `cumulative_progress` POINT BY POINT (keep / modify / add / delete) so the result is a clean updated running KB, then derive `answer_plan` and the final `answer`.

Schema (every required key MUST be present; extra keys are forbidden):

{{
  "this_turn":            "<one paragraph: what the final turn newly shows. Could be a final value confirmation, a final state observation, or a blocker preventing completion.>",
  "cumulative_progress":  "<RUNNING summary of EVERYTHING accomplished so far that is relevant to the task. ONLY write progress that has actually been made (facts confirmed, values collected, URLs identified, state changes that took effect). DO NOT write what is still missing, what remains, blockers, plans, or guesses — those belong elsewhere. Be COMPREHENSIVE — capture every task-relevant concrete fact and every confirmed state change, including small intermediate wins. Build it by taking the prior turn's `cumulative_progress` and walking through it POINT BY POINT against `this_turn`: KEEP items still true; MODIFY/CORRECT items now refined or superseded (state both old and new); ADD new items confirmed in `this_turn`; DELETE items the new evidence shows were wrong. This is your authoritative record of what was accomplished and what `answer` will be composed from.>",
  "remaining_subgoals":   ["<sub-goal NOT met; describe the blocker. Use [] if the task is fully solved.>", "..."],
  "answer_plan":          "<one or two sentences: how the answer string is composed from cumulative_progress; explicitly call out any uncertainty or unresolved blocker>",
  "answer":               "<the final answer string. For information-seeking tasks, give the concrete answer. For operation tasks, describe the completed result/state (or the blocker if it was not reached).>"
}}

Field rules:
- `this_turn`, `cumulative_progress`, `answer_plan`, `answer` are all required non-empty strings.
- `remaining_subgoals` is a JSON array of strings. Use `[]` when the task is fully solved.
- Do NOT include a `tool_call` field on the final turn.

""".strip()


FINAL_USER_PROMPT = """## Current Date
{current_date}

## Task
{task}

## Previous Thought (the JSON object emitted last turn)
```json
{last_report}
```

## Recent Actions (up to 10 most recent tool calls)
{recent_actions}


## Previous Observation
<tool_response>
{last_observation}
</tool_response>

## Current Browser State
- URL: {current_url}
- Title: {current_title}
{tabs_text}
- Screenshot: attached below

You have reached the final step budget. You are not allowed to take more tool actions.

## Required Output Format
This is the FINAL turn. Return EXACTLY ONE top-level JSON object as your entire response. No XML tags. No markdown fences. No prose outside the JSON.

Generate the keys in the order shown — `this_turn` is your fresh analysis; `cumulative_progress` is built by walking the prior `cumulative_progress` POINT BY POINT (keep / modify / add / delete) so the result is a clean updated running KB, then derive `answer_plan` and the final `answer`.

Schema (every required key MUST be present; extra keys are forbidden):

{{
  "this_turn":            "<one paragraph: what the final turn newly shows. Could be a final value confirmation, a final state observation, or a blocker preventing completion.>",
  "cumulative_progress":  "<RUNNING summary of EVERYTHING accomplished so far that is relevant to the task. ONLY write progress that has actually been made (facts confirmed, values collected, URLs identified, state changes that took effect). DO NOT write what is still missing, what remains, blockers, plans, or guesses — those belong elsewhere. Be COMPREHENSIVE — capture every task-relevant concrete fact and every confirmed state change, including small intermediate wins. Build it by taking the prior turn's `cumulative_progress` and walking through it POINT BY POINT against `this_turn`: KEEP items still true; MODIFY/CORRECT items now refined or superseded (state both old and new); ADD new items confirmed in `this_turn`; DELETE items the new evidence shows were wrong. This is your authoritative record of what was accomplished and what `answer` will be composed from.>",
  "remaining_subgoals":   ["<sub-goal NOT met; describe the blocker. Use [] if the task is fully solved.>", "..."],
  "answer_plan":          "<one or two sentences: how the answer string is composed from cumulative_progress; explicitly call out any uncertainty or unresolved blocker>",
  "answer":               "<the final answer string. For information-seeking tasks, give the concrete answer. For operation tasks, describe the completed result/state (or the blocker if it was not reached).>"
}}

Field rules:
- `this_turn`, `cumulative_progress`, `answer_plan`, `answer` are all required non-empty strings.
- `remaining_subgoals` is a JSON array of strings. Use `[]` when the task is fully solved.
- Do NOT include a `tool_call` field on the final turn.

""".strip()


AGENT_MODE_TOOL_NAMES = {
    "hybrid": [
        "search",
        "scholar",
        "visit",
        "code_sandbox",
        "click",
        "double_click",
        "hover",
        "fill",
        "select_option",
        "scroll",
        "drag",
        "wait",
        "go_back",
        "goto",
        "new_tab",
        "press_key",
        "switch_tab",
        "close_tab",
    ],
    "operation": [
        "goto",
        "new_tab",
        "click",
        "double_click",
        "hover",
        "fill",
        "select_option",
        "scroll",
        "drag",
        "wait",
        "go_back",
        "press_key",
        "switch_tab",
        "close_tab",
        "code_sandbox",
    ],
    # B-stage probe (fact_prober_v2 runs --agent-mode search): text-only.
    # The blind-answer agent's job is to find a fact on the web, not to
    # compute. code_sandbox here only burns turns building shell strings;
    # remove it so the probe stays focused on search / scholar / visit.
    "search": ["search", "scholar", "visit"],
    # hybrid_routed (路由版本): runtime tracks active side; per-side tool list
    # comes from ROUTED_SIDE_TOOL_NAMES below. The "hybrid_routed" entry here
    # is intentionally empty -- never read directly.
    "hybrid_routed": [],
    # hybrid_guided_routed: runtime/router tracks active side; decision model
    # never sees the switch tool and only acts on the current routed subgoal.
    "hybrid_guided_routed": [],
}


# Routed mode per-side tool lists. The `switch` tool is appended at both sides
# so the model can always invoke it; the runtime intercepts switch and flips
# active_mode WITHOUT executing any browser action.
ROUTED_SIDE_TOOL_NAMES = {
    "S": ["search", "scholar", "visit", "code_sandbox", "switch"],
    "O": [
        "goto",
        "new_tab",
        "click",
        "double_click",
        "hover",
        "fill",
        "select_option",
        "scroll",
        "drag",
        "wait",
        "go_back",
        "press_key",
        "switch_tab",
        "close_tab",
        "code_sandbox",
        "switch",
    ],
}

GUIDED_ROUTED_SIDE_TOOL_NAMES = {
    "S": ["search", "scholar", "visit", "code_sandbox"],
    "O": [
        "goto",
        "new_tab",
        "click",
        "double_click",
        "hover",
        "fill",
        "select_option",
        "scroll",
        "drag",
        "wait",
        "go_back",
        "press_key",
        "switch_tab",
        "close_tab",
        "code_sandbox",
    ],
}


def normalize_agent_mode(agent_mode: str | None) -> str:
    mode = (agent_mode or "hybrid").strip().lower()
    if mode not in AGENT_MODE_TOOL_NAMES:
        return "hybrid"
    return mode


def normalize_routed_side(active_mode: str | None) -> str:
    s = (active_mode or "").strip().upper()
    if s not in ("S", "O"):
        raise ValueError(f"routed mode requires active_mode 'S' or 'O', got {active_mode!r}")
    return s


MODE_TOOL_DESCRIPTIONS = {
    "operation": {},
    "search": {},
}


def get_tools_for_mode(
    agent_mode: str | None,
    active_mode: str | None = None,
    expose_switch: bool = False,
) -> list[dict]:
    mode = normalize_agent_mode(agent_mode)
    # Per-side tool catalog source: standard TOOLS plus the SWITCH_TOOL
    # (only needed when assembling routed-mode tool lists, but harmless to
    # include in the lookup map for all modes).
    tool_by_name = {tool["function"]["name"]: tool for tool in TOOLS}
    tool_by_name["switch"] = SWITCH_TOOL

    if mode in {"hybrid_routed", "hybrid_guided_routed"}:
        side = normalize_routed_side(active_mode)
        names = (
            ROUTED_SIDE_TOOL_NAMES[side]
            if mode == "hybrid_routed"
            else GUIDED_ROUTED_SIDE_TOOL_NAMES[side]
        )
        # Guided mode normally hides `switch` (the router decides). When the
        # router has decided to hand off, the runtime exposes ONLY `switch` for
        # this one turn (business tools removed) so the decision model is forced
        # to emit a real switch tool_call and cannot keep acting on this side or
        # answer early.
        if expose_switch and mode == "hybrid_guided_routed":
            names = ["switch"]
        tools = []
        for name in names:
            tool = copy.deepcopy(tool_by_name[name])
            # Tighten `visit` to require an explicit url (same tweak applied to
            # plain `search` mode below) so routed S-side behaves identically to
            # the dedicated search mode.
            if side == "S" and name == "visit":
                parameters = tool["function"]["parameters"]
                parameters["required"] = ["url", "goal"]
                parameters["properties"]["url"]["oneOf"][0]["description"] = "A webpage URL to read."
                parameters["properties"]["url"]["oneOf"][1]["description"] = (
                    "One or more webpage URLs to read."
                )
                parameters["properties"]["url"]["description"] = (
                    "Required URL or URL list to read. Provide the URL explicitly."
                )
            tools.append(tool)
        return tools

    tools = []
    for name in AGENT_MODE_TOOL_NAMES[mode]:
        tool = copy.deepcopy(tool_by_name[name])
        description = MODE_TOOL_DESCRIPTIONS.get(mode, {}).get(name)
        if description:
            tool["function"]["description"] = description
        if mode == "search" and name == "visit":
            parameters = tool["function"]["parameters"]
            parameters["required"] = ["url", "goal"]
            parameters["properties"]["url"]["oneOf"][0]["description"] = "A webpage URL to read."
            parameters["properties"]["url"]["oneOf"][1]["description"] = (
                "One or more webpage URLs to read."
            )
            parameters["properties"]["url"]["description"] = (
                "Required URL or URL list to read. Provide the URL explicitly."
            )
        tools.append(tool)
    return tools


def get_tool_str(
    agent_mode: str | None,
    active_mode: str | None = None,
    expose_switch: bool = False,
) -> str:
    return json.dumps(
        get_tools_for_mode(agent_mode, active_mode, expose_switch=expose_switch),
        ensure_ascii=False,
        indent=2,
    )


def get_allowed_tool_names(agent_mode: str | None, active_mode: str | None = None) -> set[str]:
    mode = normalize_agent_mode(agent_mode)
    if mode == "hybrid_routed":
        return set(ROUTED_SIDE_TOOL_NAMES[normalize_routed_side(active_mode)])
    if mode == "hybrid_guided_routed":
        return set(GUIDED_ROUTED_SIDE_TOOL_NAMES[normalize_routed_side(active_mode)])
    return set(AGENT_MODE_TOOL_NAMES[mode])


OPERATION_SYSTEM_PROMPT_TEMPLATE = """You are a web agent that works through browser interaction and local computation.

Available tools and JSON schemas:
{tool_str}

You can solve tasks by operating the browser and by using code_sandbox for task-provided local attachments or computations. Use only the tools listed in the schemas above.

Coordinate-based interaction:
- For click, double_click, hover, fill, select_option, scroll, and drag: read the screenshot and output (x, y) coordinates pointing to the visual center of the target element (drag uses from_x/from_y/to_x/to_y).

Action descriptions are mandatory:
- Every tool_call MUST include a short `action_description` (5-15 words) naming the visible target element and what you expect to happen. The description is appended to the action history and is the primary signal for loop detection.

Working mechanism:
- Use goto to navigate the current tab to a known URL; use new_tab to open a URL in a fresh tab and switch to it.
- Use hover, click, double_click, fill, select_option, scroll, drag, wait, go_back, and press_key to change page state and inspect visible browser evidence.
- For native <select> dropdowns there is no dedicated tool: click the dropdown to open it, then click the option, or use press_key with ArrowDown/Enter.
- Use switch_tab and close_tab to manage browser tabs when popups, OAuth flows, or target=_blank links open additional tabs. The list of currently open tabs is shown in the Current Browser State block on every turn, so you do not need a separate listing tool.
- Use code_sandbox only for calculations, parsing task-provided local attachment files, or validating data-derived results. It does not browse the web or change browser state.
- Task-provided attachment files are automatically available in code_sandbox under the sandbox paths shown in the task; read those files directly instead of inlining file contents into the tool_call JSON.
- The browser is available. If the loaded page is unavailable or unusable, use goto to navigate to a useful page.
- Do not invent information that has not been observed in the browser state or task-provided attachments.
- Re-check the current URL, title, and screenshot after every browser action.
- Before choosing the next tool, inspect Recent Actions. If the recent browser actions repeat without progress, treat it as a loop and try a different browser path or finish with the blocker.
- In each report, synthesize webpage evidence, task information, and code_sandbox results when used; summarize progress; then plan the next browser or code_sandbox tool.

Finish condition:
- For tasks requiring website operation, finish when the requested page, result, form state, submitted state, or other target state is reached.
- For information requested from a site, answer only from information visible or reached through browser operation, plus any task-provided local attachment data inspected with code_sandbox.
- If the needed URL, page, or attachment computation cannot be completed using the listed tools, answer with the final reached state and the blocker.

You will receive:
- the task
- the previous report, previous observation, and recent actions
- the current browser URL and title when available
- the current screenshot when available

The `tool_call` object inside the response JSON must use the exact English tool names, argument keys, and enum values from the schema above.
""".strip()


SEARCH_SYSTEM_PROMPT_TEMPLATE = """You are a research web agent that gathers textual evidence and performs local computation.

Available tools and JSON schemas:
{tool_str}

You can solve tasks through search results, scholar results, reader-style page/PDF visits, and code sandbox computation. Use only the tools listed in the schemas above.

Action descriptions are mandatory:
- Every tool_call MUST include a short `action_description` (5-15 words) summarizing the intent of the call (e.g. "Search for 'tomorrow weather forecast SF'" or "Visit IEEE PDF to extract abstract"). The description is appended to the action history and is the primary signal for loop detection.

Working mechanism:
- Use search to find candidate pages, sources, and query refinements.
- Use scholar to find academic papers, scholarly sources, publication metadata, citation counts, and PDF links.
- Use visit to read one or more pages or PDF documents and extract goal-relevant evidence.
- Use code_sandbox for calculations, parsing, data processing, simulations, and code validation after or during evidence gathering.
- Task-provided attachment files are automatically available in code_sandbox under the sandbox paths shown in the task; read those files directly instead of inlining file contents into the tool_call JSON.
- Treat search snippets and scholar results as leads. Use visit with an explicit url for important facts whenever possible before finalizing.
- Before choosing the next tool, inspect Recent Actions. If the recent searches or visits repeat without progress, treat it as a loop and try a different query, source, URL, or answer with the remaining uncertainty.
- If an initial URL is provided, it appears as text context. Pass it explicitly as the visit url when you need to read it.
- In each report, synthesize gathered textual evidence, code results, and task information, summarize progress, then plan the next search, scholar, visit, or code_sandbox tool.

Finish condition:
- Finish when the gathered textual evidence is sufficient to answer the task.
- If the needed information or computation cannot be completed with search, scholar, visit, and code_sandbox, answer with the best supported result and clearly state what remains missing.

You will receive:
- the task
- the previous report, previous observation, and recent actions
- the initial URL as text context when provided

The `tool_call` object inside the response JSON must use the exact English tool names and argument keys from the schema above.
""".strip()


# ----------------------------------------------------------------------------
# Routed hybrid (路由版本): two prompts (S-side / O-side) extending the existing
# SEARCH/OPERATION templates with a hand-off section that explains the path
# roles and when to invoke the `switch` tool. The model only sees the active
# side's tool list (plus `switch`) each turn.
# ----------------------------------------------------------------------------

ROUTED_SWITCH_BLOCK_FOR_SEARCH = """
## Path roles (routed hybrid)

- **search-based path (currently active)**: efficient INFORMATION gathering -- web search, scholar, reader-style page/PDF visit. Use when the next progress is to find a fact, a URL, a reference, or external evidence that is fast to look up.
- **operation-based path (available via `switch`)**: browser INTERACTION and STATE CHANGE -- navigate to URLs, click, fill forms, scroll, manage tabs, log in, submit, book, add to cart. Also better for extracting information that is only revealed through interactive page state (filters, dropdowns, dynamic lists, map views, sorted/paginated results).

## When to call `switch`

You are currently on the search-based path. Call `switch` when the operation-based path would be more effective for the next remaining_subgoal:

**Information behind interaction:**
- the needed information is only visible after applying filters, sorting, selecting dropdown options, clicking tabs, expanding sections, or navigating paginated/infinite-scroll lists -- search snippets and visit text cannot capture this dynamic content
- the task involves complex filtering conditions (e.g. "find a hotel under $200 with pool and free parking near downtown") that require interacting with a website's filter UI rather than hoping search results match all criteria
- you already located the target URL, but extracting the answer requires clicking, scrolling, or filling a form (search snippets / visit text are insufficient)

**State-changing actions:**
- the next remaining_subgoal requires a browser action that changes site state (submit a form, add to cart, book, register, set a preference, complete a checkout flow)

**Reactive recovery:**
- recent search/visit actions are looping on the same query/URL without new progress, and a known page interaction would unblock the next step

IMPORTANT: Do NOT set `answer` while `remaining_subgoals` still contains state-changing items (book, rent, buy, submit, register, add to cart, search on a specific site). You must call `switch` first to complete those subgoals via browser interaction, then answer only after ALL subgoals are resolved. Answering early with unfinished operation subgoals is incorrect — the task is not complete until the actions are performed, not just described.

`cumulative_progress` and `remaining_subgoals` carry across the switch verbatim, so DO NOT restate task context inside the switch `reason`. One sentence stating why operation-based is needed is sufficient.
""".strip()


ROUTED_SWITCH_BLOCK_FOR_OPERATION = """
## Path roles (routed hybrid)

- **operation-based path (currently active)**: browser INTERACTION and STATE CHANGE -- navigate to URLs, click, fill forms, scroll, manage tabs, log in, submit, book, add to cart. Use when the next progress requires manipulating a page or changing site state.
- **search-based path (available via `switch`)**: efficient INFORMATION gathering -- web search, scholar, reader-style page/PDF visit. Use when the next progress is to find a fact, a URL, a reference, or external evidence. Search-based tools are inherently faster and more reliable for information gathering than navigating a website manually.

## When to call `switch`

You are currently on the operation-based path. Call `switch` when the search-based path would be more effective for the next remaining_subgoal:

**Proactive efficiency (switch BEFORE getting stuck):**
- the next remaining_subgoal is primarily about FINDING or VERIFYING information (a fact, a price, a date, a name, a reference) -- search/visit can extract this directly without navigating menus, filters, or pagination
- even though a URL is provided, the needed information can be read from that URL via `visit` (text extraction) rather than interacted with via browser -- switch to search and use visit(url=..., goal=...)
- the task requires looking up information from an external source not on the current page

**Reactive recovery (switch AFTER getting stuck):**
- the current page is unhelpful and the needed fact is faster to look up via search/visit than via further browser interaction
- recent operation actions are looping on the same tool/page/coordinates without new progress, and a fresh search would surface a better starting page

IMPORTANT: If the task goal is information-seeking (find, look up, check, compare, identify, verify) and not state-changing (book, buy, submit, register, set, add to cart), prefer switching to search-based immediately -- even on the very first step.

Do NOT stay on the operation-based path just because a URL was provided. A URL is a starting point, not a commitment to browser interaction. If search/visit can answer the question without clicking, switch.

IMPORTANT: Do NOT set `answer` while `remaining_subgoals` still contains information-gathering items that require search/visit (find a fact, look up a value, compare options). You must call `switch` first to gather that information, then answer only after ALL subgoals are resolved.

`cumulative_progress` and `remaining_subgoals` carry across the switch verbatim, so DO NOT restate task context inside the switch `reason`. One sentence stating why search-based is faster is sufficient.
""".strip()


ROUTED_SEARCH_SYSTEM_PROMPT_TEMPLATE = (
    SEARCH_SYSTEM_PROMPT_TEMPLATE + "\n\n" + ROUTED_SWITCH_BLOCK_FOR_SEARCH
)

ROUTED_OPERATION_SYSTEM_PROMPT_TEMPLATE = (
    OPERATION_SYSTEM_PROMPT_TEMPLATE + "\n\n" + ROUTED_SWITCH_BLOCK_FOR_OPERATION
)

GUIDED_ROUTED_SYSTEM_BLOCK = """
## Guided routing

The active path is selected by an external router. Do not decide whether to
switch paths, and do not mention hidden routing guidance. Focus only on the
current subgoal shown in the user message. Do not include any field for
remaining or future subgoals; the runtime records the active current subgoal
separately.

Your action this turn MUST advance ONLY the current subgoal shown in the user
message. Do NOT work ahead on goals that belong to a later phase, and in
particular do NOT perform work that belongs to the OTHER path:
- On the search-based path, do NOT search for things whose answer is meant to
  come from browser interaction in a later operation phase (for example,
  computing a map/transit travel time, a route, or a price that the route plan
  expects you to read off an interactive page). Gather only the information the
  current subgoal asks for.
- On the operation-based path, do NOT navigate/click to re-find a fact the
  current subgoal does not ask you to obtain.
If the current subgoal already appears satisfied by what you have, take the
smallest confirming action or a no-op-style action that lets the router advance;
do not invent extra work to fill the turn. The router — not you — will move to
the next subgoal and switch paths when appropriate.
""".strip()


# Shown ONLY on the turn the router has decided to hand off. Overrides the
# standing "the router decides switches, not you" instruction for this one turn:
# the current_subgoal now explicitly asks for a `switch`, and the switch tool is
# in the schema. The decision model should emit a real switch tool_call with a
# continuous report (carry cumulative_progress forward verbatim).
GUIDED_ROUTED_SWITCH_NOW_BLOCK = """
## Hand-off this turn

The routing controller has determined the next subgoal belongs to the OTHER
path. Your current_subgoal this turn is to hand off: call the `switch` tool now.
Do NOT perform a search/scholar/visit or a browser action this turn — emit
`switch` with a one-sentence `reason` matching the current_subgoal. Carry your
`cumulative_progress` forward verbatim in the report; the switch does not change
the browser or lose any progress. After the switch, you will continue on the
other path.
""".strip()


def get_system_prompt(
    agent_mode: str | None,
    active_mode: str | None = None,
    expose_switch: bool = False,
) -> str:
    mode = normalize_agent_mode(agent_mode)
    if mode == "hybrid_routed":
        side = normalize_routed_side(active_mode)
        tool_str = get_tool_str(mode, side)
        if side == "S":
            return ROUTED_SEARCH_SYSTEM_PROMPT_TEMPLATE.format(tool_str=tool_str)
        return ROUTED_OPERATION_SYSTEM_PROMPT_TEMPLATE.format(tool_str=tool_str)
    if mode == "hybrid_guided_routed":
        side = normalize_routed_side(active_mode)
        tool_str = get_tool_str(mode, side, expose_switch=expose_switch)
        guided_block = GUIDED_ROUTED_SYSTEM_BLOCK
        if expose_switch:
            guided_block = guided_block + "\n\n" + GUIDED_ROUTED_SWITCH_NOW_BLOCK
        base = SEARCH_SYSTEM_PROMPT_TEMPLATE if side == "S" else OPERATION_SYSTEM_PROMPT_TEMPLATE
        return (base + "\n\n" + guided_block).format(tool_str=tool_str)
    tool_str = get_tool_str(mode)
    if mode == "operation":
        return OPERATION_SYSTEM_PROMPT_TEMPLATE.format(tool_str=tool_str)
    if mode == "search":
        return SEARCH_SYSTEM_PROMPT_TEMPLATE.format(tool_str=tool_str)
    return SYSTEM_PROMPT_TEMPLATE.format(tool_str=tool_str)


OPERATION_INITIAL_USER_PROMPT = """## Current Date
{current_date}

## Task
{task}

## Current Browser State
- URL: {current_url}
- Title: {current_title}
{tabs_text}
- Screenshot: attached below

## Required Output Format
Return EXACTLY ONE top-level JSON object as your entire response. No XML tags. No markdown fences. No prose outside the JSON.

This is the FIRST turn. Analyze the task, decompose it into sub-goals, then choose the first action.

Schema (every required key MUST be present; extra keys are forbidden):

{{
  "task_analysis":          "<Analyze the task: (1) identify all concrete sub-goals — facts to find, values to collect, actions to perform; (2) determine which sub-goals to tackle first based on the current browser state.>",
  "remaining_subgoals":     ["<atomic open item still required to finish the task>", "..."],
  "next_action_rationale":  "<one sentence: WHY the chosen tool will close some remaining_subgoal>",
  "tool_call":              {{"name": "<tool name from the schemas above>", "arguments": {{...}}}},
  "action_description":     "<5-15 word short description of the action: name the visible target element and expected outcome>"
}}

Field rules:
- `task_analysis` is a paragraph decomposing the task into sub-goals. Required and non-empty.
- `remaining_subgoals` is a JSON array of strings listing all open items.
- `next_action_rationale` is a non-empty one-sentence string stating why the chosen tool advances a remaining_subgoal.
- `tool_call` is required — choose the first tool to invoke based on the task analysis.
- `action_description` is a non-empty string (5-15 words).
- `tool_call.arguments` must use the exact English argument keys / enum values from the tool schema.

Rules:
- Use only goto, new_tab, hover, click, double_click, fill, select_option, scroll, drag, wait, go_back, press_key, switch_tab, close_tab, or code_sandbox.
- Use exactly one tool call.
- For hover, click, fill, regional scroll, and drag, output (x, y) coordinates of the target element identified visually from the screenshot.
""".strip()


OPERATION_TURN_USER_PROMPT = """## Current Date
{current_date}

## Task
{task}

## Previous Thought (the JSON object emitted last turn)
```json
{last_report}
```

## Recent Actions (up to 10 most recent tool calls)
{recent_actions}


## Previous Observation
<tool_response>
{last_observation}
</tool_response>

## Current Browser State
- URL: {current_url}
- Title: {current_title}
{tabs_text}
- Screenshot: attached below

## Required Output Format
Return EXACTLY ONE top-level JSON object as your entire response. No XML tags. No markdown fences. No prose outside the JSON.

Generate the keys in the order shown — `this_turn` is your fresh analysis of THIS turn's observation, and `cumulative_progress` is built by READING the prior turn's `cumulative_progress` (from "Previous Thought" above) and walking through it POINT BY POINT — explicitly KEEPING items still true, MODIFYING/CORRECTING items now refined or superseded (state both old and new), ADDING new items contributed by `this_turn`, and DELETING items no longer true. Format is flexible prose or short bullet-like lines.

Schema (every required key MUST be present; extra keys are forbidden):

{{
  "this_turn":              "<one paragraph: what THIS turn newly contributed. May be a fact surfaced from the latest tool result, a state change just produced (e.g. \"clicked 'Add to bag', cart count 0→1\", \"page navigated from /search to /product/123\"), a confirmation, or a failure. Be specific.>",
  "cumulative_progress":    "<RUNNING summary of EVERYTHING accomplished so far that is relevant to the task. Carried across turns. ONLY write progress that has actually been made (facts confirmed, values collected, URLs identified, state changes that took effect). DO NOT write what is still missing, what remains to do, blockers, plans, or guesses — those belong in `remaining_subgoals` or `this_turn`, not here. Be COMPREHENSIVE — capture every task-relevant concrete fact and every confirmed state change, including small intermediate wins (filter applied successfully, correct page reached, dropdown configured, login completed). Build it by taking the prior turn's `cumulative_progress` (from \"Previous Thought\") and walking through it POINT BY POINT against `this_turn`: KEEP each prior item that is still true; MODIFY/CORRECT items now refined or superseded (state both the old and the new value); ADD any NEW items confirmed in `this_turn`; DELETE items the new evidence shows were wrong. Cover BOTH information progress (concrete facts/values/URLs collected) AND operation progress (cart additions, sign-ins, forms submitted, filters applied, bookings confirmed, navigation milestones reached). Each entry must describe something already done, not something pending.>",
  "remaining_subgoals":     ["<atomic open item still required to finish the task. Can be a fact to find OR a state target not yet reached (e.g. \"item still missing from cart\", \"booking form not yet submitted\")>", "..."],
  "next_action_rationale":  "<one sentence: WHY the chosen tool will close some remaining_subgoal>",
  "tool_call":              {{"name": "<tool name from the schemas above>", "arguments": {{...}}}} OR null,
  "action_description":     "<5-15 word short description of the action: name the visible target element and the expected outcome. REQUIRED when tool_call is non-null; use null when answer is set instead.>",
  "answer":                 null OR "<final answer string if you are confident the task is solved/blocked and want to terminate THIS TURN early>"
}}

Field rules:
- `this_turn` and `cumulative_progress` are paragraph-style strings (free-form prose, may include short bullet-like lines). Both are required and non-empty.
- `remaining_subgoals` is a JSON array of strings (may be the empty array `[]` only when the task is fully solved).
- EXACTLY ONE of `tool_call` (object) and `answer` (non-empty string) must be non-null; the other MUST be null.
- Set `answer` (and `tool_call`=null) ONLY when you are confident the task is fully solved (or definitively blocked) and want to terminate the trajectory THIS turn. Otherwise, set `tool_call` (the next tool to invoke) and `answer`=null.
- When `tool_call` is set, it must be the structured tool-call JSON for ONE tool from the schemas listed in the system prompt.
- `tool_call.arguments` must use the exact English argument keys / enum values from the tool schema.
- When `tool_call` is non-null, `action_description` MUST be a non-empty string of 5-15 words naming the visible target element and the expected outcome. When `answer` is set (and `tool_call` is null), `action_description` MUST be null.

Rules:
- Use only goto, new_tab, hover, click, double_click, fill, select_option, scroll, drag, wait, go_back, press_key, switch_tab, close_tab, or code_sandbox.
- Use code_sandbox only for calculations, parsing task-provided local attachments, or validating data-derived results; do not use it to browse the web.
- Do not repeat the same failed browser action without changing strategy.
- Use exactly one tool call if not answering.
- For hover, click, fill, regional scroll, and drag, output (x, y) coordinates of the target element identified visually from the screenshot. drag takes (from_x, from_y, to_x, to_y).
""".strip()


OPERATION_FINAL_USER_PROMPT = """## Current Date
{current_date}

## Task
{task}

## Previous Thought (the JSON object emitted last turn)
```json
{last_report}
```

## Recent Actions (up to 10 most recent tool calls)
{recent_actions}


## Previous Observation
<tool_response>
{last_observation}
</tool_response>

## Current Browser State
- URL: {current_url}
- Title: {current_title}
{tabs_text}
- Screenshot: attached below

You have reached the final step budget. You are not allowed to take more tool actions.

## Required Output Format
This is the FINAL turn. Return EXACTLY ONE top-level JSON object as your entire response. No XML tags. No markdown fences. No prose outside the JSON.

Generate the keys in the order shown — `this_turn` is your fresh analysis; `cumulative_progress` is built by walking the prior `cumulative_progress` POINT BY POINT (keep / modify / add / delete) so the result is a clean updated running KB, then derive `answer_plan` and the final `answer`.

Schema (every required key MUST be present; extra keys are forbidden):

{{
  "this_turn":            "<one paragraph: what the final turn newly shows. Could be a final value confirmation, a final state observation, or a blocker preventing completion.>",
  "cumulative_progress":  "<RUNNING summary of EVERYTHING accomplished so far that is relevant to the task. ONLY write progress that has actually been made (facts confirmed, values collected, URLs identified, state changes that took effect). DO NOT write what is still missing, what remains, blockers, plans, or guesses — those belong elsewhere. Be COMPREHENSIVE — capture every task-relevant concrete fact and every confirmed state change, including small intermediate wins. Build it by taking the prior turn's `cumulative_progress` and walking through it POINT BY POINT against `this_turn`: KEEP items still true; MODIFY/CORRECT items now refined or superseded (state both old and new); ADD new items confirmed in `this_turn`; DELETE items the new evidence shows were wrong. This is your authoritative record of what was accomplished and what `answer` will be composed from.>",
  "remaining_subgoals":   ["<sub-goal NOT met; describe the blocker. Use [] if the task is fully solved.>", "..."],
  "answer_plan":          "<one or two sentences: how the answer string is composed from cumulative_progress; explicitly call out any uncertainty or unresolved blocker>",
  "answer":               "<the final answer string. For information-seeking tasks, give the concrete answer. For operation tasks, describe the completed result/state (or the blocker if it was not reached).>"
}}

Field rules:
- `this_turn`, `cumulative_progress`, `answer_plan`, `answer` are all required non-empty strings.
- `remaining_subgoals` is a JSON array of strings. Use `[]` when the task is fully solved.
- Do NOT include a `tool_call` field on the final turn.

""".strip()


SEARCH_INITIAL_USER_PROMPT = """## Current Date
{current_date}

## Task
{task_with_web}

## Required Output Format
Return EXACTLY ONE top-level JSON object as your entire response. No XML tags. No markdown fences. No prose outside the JSON.

This is the FIRST turn. Analyze the task, decompose it into sub-goals, then choose the first action.

Schema (every required key MUST be present; extra keys are forbidden):

{{
  "task_analysis":          "<Analyze the task: (1) identify all concrete sub-goals — facts to find, values to collect, sources to check; (2) determine which sub-goals to tackle first.>",
  "remaining_subgoals":     ["<atomic open item still required to finish the task>", "..."],
  "next_action_rationale":  "<one sentence: WHY the chosen tool will close some remaining_subgoal>",
  "tool_call":              {{"name": "<tool name from the schemas above>", "arguments": {{...}}}},
  "action_description":     "<5-15 word short description of the action: name the target and expected outcome>"
}}

Field rules:
- `task_analysis` is a paragraph decomposing the task into sub-goals. Required and non-empty.
- `remaining_subgoals` is a JSON array of strings listing all open items.
- `next_action_rationale` is a non-empty one-sentence string stating why the chosen tool advances a remaining_subgoal.
- `tool_call` is required — choose the first tool to invoke based on the task analysis.
- `action_description` is a non-empty string (5-15 words).
- `tool_call.arguments` must use the exact English argument keys / enum values from the tool schema.

Rules:
- Use only search, scholar, visit, or code_sandbox.
- Use scholar for academic papers, scholarly sources, citation-rich evidence, and PDF links.
- Use exactly one tool call.
""".strip()


SEARCH_TURN_USER_PROMPT = """## Current Date
{current_date}

## Task
{task_with_web}

## Previous Thought (the JSON object emitted last turn)
```json
{last_report}
```

## Recent Actions (up to 10 most recent tool calls)
{recent_actions}


## Previous Observation
<tool_response>
{last_observation}
</tool_response>

## Required Output Format
Return EXACTLY ONE top-level JSON object as your entire response. No XML tags. No markdown fences. No prose outside the JSON.

Generate the keys in the order shown — `this_turn` is your fresh analysis of THIS turn's observation, and `cumulative_progress` is built by READING the prior turn's `cumulative_progress` (from "Previous Thought" above) and walking through it POINT BY POINT — explicitly KEEPING items still true, MODIFYING/CORRECTING items now refined or superseded (state both old and new), ADDING new items contributed by `this_turn`, and DELETING items no longer true. Format is flexible prose or short bullet-like lines.

Schema (every required key MUST be present; extra keys are forbidden):

{{
  "this_turn":              "<one paragraph: what THIS turn newly contributed. May be a fact surfaced from the latest tool result, a state change just produced (e.g. \"clicked 'Add to bag', cart count 0→1\", \"page navigated from /search to /product/123\"), a confirmation, or a failure. Be specific.>",
  "cumulative_progress":    "<RUNNING summary of EVERYTHING accomplished so far that is relevant to the task. Carried across turns. ONLY write progress that has actually been made (facts confirmed, values collected, URLs identified, state changes that took effect). DO NOT write what is still missing, what remains to do, blockers, plans, or guesses — those belong in `remaining_subgoals` or `this_turn`, not here. Be COMPREHENSIVE — capture every task-relevant concrete fact and every confirmed state change, including small intermediate wins (filter applied successfully, correct page reached, dropdown configured, login completed). Build it by taking the prior turn's `cumulative_progress` (from \"Previous Thought\") and walking through it POINT BY POINT against `this_turn`: KEEP each prior item that is still true; MODIFY/CORRECT items now refined or superseded (state both the old and the new value); ADD any NEW items confirmed in `this_turn`; DELETE items the new evidence shows were wrong. Cover BOTH information progress (concrete facts/values/URLs collected) AND operation progress (cart additions, sign-ins, forms submitted, filters applied, bookings confirmed, navigation milestones reached). Each entry must describe something already done, not something pending.>",
  "remaining_subgoals":     ["<atomic open item still required to finish the task. Can be a fact to find OR a state target not yet reached (e.g. \"item still missing from cart\", \"booking form not yet submitted\")>", "..."],
  "next_action_rationale":  "<one sentence: WHY the chosen tool will close some remaining_subgoal>",
  "tool_call":              {{"name": "<tool name from the schemas above>", "arguments": {{...}}}} OR null,
  "action_description":     "<5-15 word short description of the action: name the visible target element and the expected outcome. REQUIRED when tool_call is non-null; use null when answer is set instead.>",
  "answer":                 null OR "<final answer string if you are confident the task is solved/blocked and want to terminate THIS TURN early>"
}}

Field rules:
- `this_turn` and `cumulative_progress` are paragraph-style strings (free-form prose, may include short bullet-like lines). Both are required and non-empty.
- `remaining_subgoals` is a JSON array of strings (may be the empty array `[]` only when the task is fully solved).
- EXACTLY ONE of `tool_call` (object) and `answer` (non-empty string) must be non-null; the other MUST be null.
- Set `answer` (and `tool_call`=null) ONLY when you are confident the task is fully solved (or definitively blocked) and want to terminate the trajectory THIS turn. Otherwise, set `tool_call` (the next tool to invoke) and `answer`=null.
- When `tool_call` is set, it must be the structured tool-call JSON for ONE tool from the schemas listed in the system prompt.
- `tool_call.arguments` must use the exact English argument keys / enum values from the tool schema.
- When `tool_call` is non-null, `action_description` MUST be a non-empty string of 5-15 words naming the visible target element and the expected outcome. When `answer` is set (and `tool_call` is null), `action_description` MUST be null.

Rules:
- Use only search, scholar, visit, or code_sandbox.
- If visit fails for a URL, change source, query, or URL rather than repeating the same failed visit.
- Search snippets and scholar results are leads; use visit with an explicit url for important facts whenever possible.
- Use scholar for academic papers, scholarly sources, citation-rich evidence, and PDF links.
- Use parse_type="pdf" when reading PDF papers or documents.
- Use code_sandbox for calculations, parsing, data processing, simulations, and code validation. Do not use it to browse the web. For task-provided attachments, read the sandbox path shown in the task instead of inlining file contents.
- Use exactly one tool call if not answering.
""".strip()


SEARCH_FINAL_USER_PROMPT = """## Current Date
{current_date}

## Task
{task_with_web}

## Previous Thought (the JSON object emitted last turn)
```json
{last_report}
```

## Recent Actions (up to 10 most recent tool calls)
{recent_actions}


## Previous Observation
<tool_response>
{last_observation}
</tool_response>

You have reached the final step budget. You are not allowed to take more tool actions.

## Required Output Format
This is the FINAL turn. Return EXACTLY ONE top-level JSON object as your entire response. No XML tags. No markdown fences. No prose outside the JSON.

Generate the keys in the order shown — `this_turn` is your fresh analysis; `cumulative_progress` is built by walking the prior `cumulative_progress` POINT BY POINT (keep / modify / add / delete) so the result is a clean updated running KB, then derive `answer_plan` and the final `answer`.

Schema (every required key MUST be present; extra keys are forbidden):

{{
  "this_turn":            "<one paragraph: what the final turn newly shows. Could be a final value confirmation, a final state observation, or a blocker preventing completion.>",
  "cumulative_progress":  "<RUNNING summary of EVERYTHING accomplished so far that is relevant to the task. ONLY write progress that has actually been made (facts confirmed, values collected, URLs identified, state changes that took effect). DO NOT write what is still missing, what remains, blockers, plans, or guesses — those belong elsewhere. Be COMPREHENSIVE — capture every task-relevant concrete fact and every confirmed state change, including small intermediate wins. Build it by taking the prior turn's `cumulative_progress` and walking through it POINT BY POINT against `this_turn`: KEEP items still true; MODIFY/CORRECT items now refined or superseded (state both old and new); ADD new items confirmed in `this_turn`; DELETE items the new evidence shows were wrong. This is your authoritative record of what was accomplished and what `answer` will be composed from.>",
  "remaining_subgoals":   ["<sub-goal NOT met; describe the blocker. Use [] if the task is fully solved.>", "..."],
  "answer_plan":          "<one or two sentences: how the answer string is composed from cumulative_progress; explicitly call out any uncertainty or unresolved blocker>",
  "answer":               "<the final answer string. For information-seeking tasks, give the concrete answer. For operation tasks, describe the completed result/state (or the blocker if it was not reached).>"
}}

Field rules:
- `this_turn`, `cumulative_progress`, `answer_plan`, `answer` are all required non-empty strings.
- `remaining_subgoals` is a JSON array of strings. Use `[]` when the task is fully solved.
- Do NOT include a `tool_call` field on the final turn.

""".strip()


USER_PROMPTS_BY_MODE = {
    "hybrid": {
        "initial": INITIAL_USER_PROMPT,
        "initial_no_browser": INITIAL_USER_PROMPT_NO_BROWSER,
        "turn": TURN_USER_PROMPT,
        "turn_no_browser": TURN_USER_PROMPT_NO_BROWSER,
        "final": FINAL_USER_PROMPT,
        "final_no_browser": FINAL_USER_PROMPT_NO_BROWSER,
    },
    "operation": {
        "initial": OPERATION_INITIAL_USER_PROMPT,
        "initial_no_browser": OPERATION_INITIAL_USER_PROMPT,
        "turn": OPERATION_TURN_USER_PROMPT,
        "turn_no_browser": OPERATION_TURN_USER_PROMPT,
        "final": OPERATION_FINAL_USER_PROMPT,
        "final_no_browser": OPERATION_FINAL_USER_PROMPT,
    },
    "search": {
        "initial": SEARCH_INITIAL_USER_PROMPT,
        "initial_no_browser": SEARCH_INITIAL_USER_PROMPT,
        "turn": SEARCH_TURN_USER_PROMPT,
        "turn_no_browser": SEARCH_TURN_USER_PROMPT,
        "final": SEARCH_FINAL_USER_PROMPT,
        "final_no_browser": SEARCH_FINAL_USER_PROMPT,
    },
}


ROUTED_SEARCH_INITIAL_USER_PROMPT = """## Current Date
{current_date}

## Task
{task_with_web}

## Required Output Format
Return EXACTLY ONE top-level JSON object as your entire response. No XML tags. No markdown fences. No prose outside the JSON.

This is the FIRST turn. Analyze the task, decompose it into sub-goals, then choose the first action.

Schema (every required key MUST be present; extra keys are forbidden):

{{
  "task_analysis":          "<Analyze the task and identify concrete sub-goals.>",
  "remaining_subgoals":     ["<atomic open item still required to finish the task>", "..."],
  "next_action_rationale":  "<one sentence: WHY the chosen tool will close some remaining_subgoal>",
  "tool_call":              {{"name": "<tool name from the schemas above>", "arguments": {{...}}}},
  "action_description":     "<5-15 word short description of the action: name the target and expected outcome>"
}}

Field rules:
- `task_analysis` is a paragraph decomposing the task into concrete sub-goals. Required and non-empty.
- `remaining_subgoals` is a JSON array of strings listing all open items. Must not be empty on the first turn.
- `next_action_rationale` is a non-empty one-sentence string stating why the chosen tool advances a remaining_subgoal.
- `tool_call` is required — choose the first tool to invoke based on the task analysis.
- `action_description` is a non-empty string (5-15 words).
- `tool_call.arguments` must use the exact English argument keys / enum values from the tool schema.

Rules:
- Use only search, scholar, visit, code_sandbox, or switch.
- Use exactly one tool call.
""".strip()


ROUTED_OPERATION_INITIAL_USER_PROMPT = """## Current Date
{current_date}

## Task
{task}

## Current Browser State
- URL: {current_url}
- Title: {current_title}
{tabs_text}
- Screenshot: attached below

## Required Output Format
Return EXACTLY ONE top-level JSON object as your entire response. No XML tags. No markdown fences. No prose outside the JSON.

This is the FIRST turn. Analyze the task, decompose it into sub-goals, then choose the first action.

Schema (every required key MUST be present; extra keys are forbidden):

{{
  "task_analysis":          "<Analyze the task and identify concrete sub-goals.>",
  "remaining_subgoals":     ["<atomic open item still required to finish the task>", "..."],
  "next_action_rationale":  "<one sentence: WHY the chosen tool will close some remaining_subgoal>",
  "tool_call":              {{"name": "<tool name from the schemas above>", "arguments": {{...}}}},
  "action_description":     "<5-15 word short description of the action: name the visible target element and expected outcome>"
}}

Field rules:
- `task_analysis` is a paragraph decomposing the task into concrete sub-goals. Required and non-empty.
- `remaining_subgoals` is a JSON array of strings listing all open items. Must not be empty on the first turn.
- `next_action_rationale` is a non-empty one-sentence string stating why the chosen tool advances a remaining_subgoal.
- `tool_call` is required — choose the first tool to invoke based on the task analysis.
- `action_description` is a non-empty string (5-15 words).
- `tool_call.arguments` must use the exact English argument keys / enum values from the tool schema.

Rules:
- Use only goto, new_tab, hover, click, double_click, fill, select_option, scroll, drag, wait, go_back, press_key, switch_tab, close_tab, code_sandbox, or switch.
- Use exactly one tool call.
- For hover, click, fill, regional scroll, and drag, output (x, y) coordinates of the target element identified visually from the screenshot.
""".strip()


USER_PROMPTS_BY_ROUTED_SIDE = {
    "S": {
        "initial": ROUTED_SEARCH_INITIAL_USER_PROMPT,
        "initial_no_browser": ROUTED_SEARCH_INITIAL_USER_PROMPT,
        "turn": SEARCH_TURN_USER_PROMPT,
        "turn_no_browser": SEARCH_TURN_USER_PROMPT,
        "final": SEARCH_FINAL_USER_PROMPT,
        "final_no_browser": SEARCH_FINAL_USER_PROMPT,
    },
    "O": {
        "initial": ROUTED_OPERATION_INITIAL_USER_PROMPT,
        "initial_no_browser": ROUTED_OPERATION_INITIAL_USER_PROMPT,
        "turn": OPERATION_TURN_USER_PROMPT,
        "turn_no_browser": OPERATION_TURN_USER_PROMPT,
        "final": OPERATION_FINAL_USER_PROMPT,
        "final_no_browser": OPERATION_FINAL_USER_PROMPT,
    },
}


def _guided_prompt_without_remaining_subgoals(template: str) -> str:
    text = template
    text = text.replace(
        "those belong in `remaining_subgoals` or `this_turn`, not here",
        "those belong in `this_turn` or the next action rationale, not here",
    )
    text = text.replace(
        "WHY the chosen tool will close some remaining_subgoal",
        "WHY the chosen tool advances the current guided subgoal",
    )
    text = text.replace(
        "WHY the chosen tool advances a remaining_subgoal",
        "WHY the chosen tool advances the current guided subgoal",
    )
    # The initial-turn schema note phrases this in lowercase prose, which the
    # uppercase replacements above miss. Rewrite it so guided prompts never
    # mention remaining_subgoals (the decision model neither sees nor emits it).
    text = text.replace(
        "stating why the chosen tool advances a remaining_subgoal",
        "stating why the chosen tool advances the current guided subgoal",
    )
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if '"remaining_subgoals":' in line:
            continue
        if stripped.startswith("- `remaining_subgoals`"):
            continue
        lines.append(line)
    return "\n".join(lines)


GUIDED_SEARCH_INITIAL_USER_PROMPT = _guided_prompt_without_remaining_subgoals(
    SEARCH_INITIAL_USER_PROMPT
)
GUIDED_SEARCH_TURN_USER_PROMPT = _guided_prompt_without_remaining_subgoals(
    SEARCH_TURN_USER_PROMPT
)
GUIDED_SEARCH_FINAL_USER_PROMPT = _guided_prompt_without_remaining_subgoals(
    SEARCH_FINAL_USER_PROMPT
)
GUIDED_OPERATION_INITIAL_USER_PROMPT = _guided_prompt_without_remaining_subgoals(
    OPERATION_INITIAL_USER_PROMPT
)
GUIDED_OPERATION_TURN_USER_PROMPT = _guided_prompt_without_remaining_subgoals(
    OPERATION_TURN_USER_PROMPT
)
GUIDED_OPERATION_FINAL_USER_PROMPT = _guided_prompt_without_remaining_subgoals(
    OPERATION_FINAL_USER_PROMPT
)

USER_PROMPTS_BY_GUIDED_ROUTED_SIDE = {
    "S": {
        "initial": GUIDED_SEARCH_INITIAL_USER_PROMPT,
        "initial_no_browser": GUIDED_SEARCH_INITIAL_USER_PROMPT,
        "turn": GUIDED_SEARCH_TURN_USER_PROMPT,
        "turn_no_browser": GUIDED_SEARCH_TURN_USER_PROMPT,
        "final": GUIDED_SEARCH_FINAL_USER_PROMPT,
        "final_no_browser": GUIDED_SEARCH_FINAL_USER_PROMPT,
    },
    "O": {
        "initial": GUIDED_OPERATION_INITIAL_USER_PROMPT,
        "initial_no_browser": GUIDED_OPERATION_INITIAL_USER_PROMPT,
        "turn": GUIDED_OPERATION_TURN_USER_PROMPT,
        "turn_no_browser": GUIDED_OPERATION_TURN_USER_PROMPT,
        "final": GUIDED_OPERATION_FINAL_USER_PROMPT,
        "final_no_browser": GUIDED_OPERATION_FINAL_USER_PROMPT,
    },
}


def get_user_prompt_template(
    agent_mode: str | None,
    prompt_kind: str,
    active_mode: str | None = None,
) -> str:
    mode = normalize_agent_mode(agent_mode)
    if mode == "hybrid_routed":
        side = normalize_routed_side(active_mode)
        return USER_PROMPTS_BY_ROUTED_SIDE[side][prompt_kind]
    if mode == "hybrid_guided_routed":
        side = normalize_routed_side(active_mode)
        return USER_PROMPTS_BY_GUIDED_ROUTED_SIDE[side][prompt_kind]
    return USER_PROMPTS_BY_MODE[mode][prompt_kind]

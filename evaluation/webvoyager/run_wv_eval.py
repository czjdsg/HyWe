#!/usr/bin/env python3
"""WebVoyager 官方 auto_eval 协议 × HybridWebAgent 轨迹适配版。

官方协议(MinorJerry/WebVoyager evaluation/auto_eval.py): 单次调用 =
  任务指令 + 末尾 k(=15) 张截图 + 最终文本答案(Result Response) → SUCCESS / NOT SUCCESS。
适配(与 M2W 的做法一致): 非浏览器工具(search/visit/code/scholar)的文本返回是 S 侧
步骤的"页面观测", 与截图同等地位 —— 与截图合并成按步排序的观测时间线, 取末尾 k 条;
文本观测以文本块进入消息。答案在官方协议中本就是一等输入, 无需额外处理。
SYSTEM_PROMPT/USER_PROMPT 沿用官方原文, 仅追加一段说明文本观测的地位。
"""
import argparse, base64, json, glob, os, re, time
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

SYSTEM_PROMPT = """As an evaluator, you will be presented with three primary components to assist you in your role:

1. Web Task Instruction: This is a clear and specific directive provided in natural language, detailing the online activity to be carried out. These requirements may include conducting searches, verifying information, comparing prices, checking availability, or any other action relevant to the specified web service (such as Amazon, Apple, ArXiv, BBC News, Booking etc).

2. Result Screenshots: This is a visual representation of the screen showing the result or intermediate state of performing a web task. It serves as visual proof of the actions taken in response to the instruction.

3. Result Response: This is a textual response obtained after the execution of the web task. It serves as textual result in response to the instruction.

-- You DO NOT NEED to interact with web pages or perform actions such as booking flights or conducting searches on websites.
-- You SHOULD NOT make assumptions based on information not presented in the screenshot when comparing it to the instructions.
-- Your primary responsibility is to conduct a thorough assessment of the web task instruction against the outcome depicted in the screenshot and in the response, evaluating whether the actions taken align with the given instructions.
-- NOTE that the instruction may involve more than one task, for example, locating the garage and summarizing the review. Failing to complete either task, such as not providing a summary, should be considered unsuccessful.
-- NOTE that the screenshot is authentic, but the response provided by LLM is generated at the end of web browsing, and there may be discrepancies between the text and the screenshots.
-- Note the difference: 1) Result response may contradict the screenshot, then the content of the screenshot prevails, 2) The content in the Result response is not mentioned on the screenshot, choose to believe the content.

Additional note for this hybrid agent: besides browser screenshots, the agent can also use non-browser tools (web search, webpage reader, code sandbox). The textual observations returned by those tools are provided in the trajectory below, in chronological order together with the screenshots. Treat such textual observations as page-content evidence with the same status as screenshots.

You should elaborate on how you arrived at your final evaluation and then provide a definitive verdict on whether the task has been successfully accomplished, either as 'SUCCESS' or 'NOT SUCCESS'.

## Access blocking (report this separately from success)

Independently of whether the task succeeded, report whether the **site refused access** and
that is why something could not be done. This is about the environment turning the agent
away, not about the agent doing the wrong thing.

Counts as blocked: a CAPTCHA or "verify you are human" gate, a bot-detection or anti-scraping
challenge, a login/paywall used as a gate the agent has no credentials for, an IP, region or
country block, rate limiting or "too many requests", a certificate or connection error the
agent cannot get past, or a page that simply refuses to serve content to this client.

Does NOT count as blocked: the agent clicking the wrong thing, mis-reading a page, running out
of steps, giving up on its own, a search returning no matches, or a page that loaded fine but
did not contain what was wanted.

Blocked is not a verdict. Answer it even when the task succeeded — the agent may have been
turned away on one sub-goal and still reached the required end state another way.

End your reply with this line, on its own:
Blocked: 'YES' or 'NO'   (did a site refuse access, per the Access blocking criterion above; answer independently of the verdict)"""

USER_PROMPT = """TASK: <task>
Result Response: <answer>
<num> observations (screenshots and textual tool returns) at the end: """

NONBROWSER_TOOLS = {'search', 'visit', 'code', 'scholar', 'code_sandbox', 'python'}

# 官方口径是判官末尾吐大写的 SUCCESS / NOT SUCCESS。实测 kimi-k3 有 2.6% 的回复
# (base 643 题里 17 条)判词写成散文 —— "The task was completed successfully."、
# "the task was not completed" —— 大写 token 一个没有,严格解析就判成不可判,
# 而这些题的结论其实毫不含糊。补捞只读判官已经写下的文本,**不重新调用判官**,
# 所以不改变任何一条已有判定,只把漏掉的那些捞回来(base 上 15 成功 / 2 失败,
# 成功率 78.3% -> 78.5%,几乎不动 —— 补捞是为了完整,不是为了抬分)。
# 顺序要紧:先否定后肯定,否则 "was not completed successfully" 会被判成功。
_NEG = re.compile(r'\b(?:not\s+success|unsuccessful|did\s+not\s+(?:succeed|complete)|'
                  r'was\s+not\s+(?:completed|accomplished|successful)|'
                  r'has\s+not\s+been\s+(?:completed|accomplished)|'
                  r'task\s+(?:was|is)\s+not\b|failed\s+to\s+(?:complete|accomplish)|'
                  r'prevented\s+any\s+task\s+completion)\b', re.I)
_POS = re.compile(r'\b(?:task\s+(?:was|is|has\s+been)\s+(?:successfully\s+)?'
                  r'(?:completed|accomplished)|completed\s+successfully|'
                  r'successfully\s+(?:completed|accomplished|planned))\b', re.I)
# 只删 `Blocked:` **那一行**。写成 re.S 会让 . 吃掉换行,从第一行 Blocked 起把后面
# 全删光 —— 而判官经常把 SUCCESS 写在 Blocked 行之后(实测 3 条被这么误伤成不可判)。
_BLOCKED_LINE = re.compile(r'^[ \t]*Blocked\s*:.*$', re.M | re.I)


def parse_verdict(text):
    """判官回复 -> 1 / 0 / None。官方 token 优先,散文判词兜底。"""
    if not text:
        return None
    # 末尾那行 `Blocked: 'YES'` 是另一个轴,不参与成败解析
    body = _BLOCKED_LINE.sub('', text)
    if 'NOT SUCCESS' in body:
        return 0
    if 'SUCCESS' in body:
        return 1
    if _NEG.search(body):
        return 0
    if _POS.search(body):
        return 1
    return None


_URL_CACHE = {}


def _tunnel_ready():
    """本地图床 + Cloudflare 隧道模式;由 run_with_image_url_transport.py 注入。"""
    return (os.environ.get("HYBRID_IMAGE_TRANSPORT", "").strip().lower() == "url"
            and os.environ.get("HYBRID_IMAGE_URL_BASE", "").strip()
            and os.environ.get("HYBRID_IMAGE_PUBLISH_DIR", "").strip())


def image_content(p, maxw=1280, quality=72):
    """返回一个 image_url 片段:优先隧道 URL,否则 base64 内联。

    两条路的**像素完全一致**(同样 resize 到 maxw、同样 JPEG quality),
    只有传输方式不同 —— 换传输不会改变判官看到的画面,结果可与旧跑对比。
    """
    if _tunnel_ready():
        hit = _URL_CACHE.get(p)
        if hit:
            return {'type': 'image_url', 'image_url': {'url': hit}}
        try:
            import io, sys as _sys
            from PIL import Image
            im = Image.open(p).convert('RGB')
            if im.width > maxw:
                im = im.resize((maxw, int(im.height * maxw / im.width)))
            buf = io.BytesIO()
            im.save(buf, 'JPEG', quality=quality)
            _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "inference"))
            from hybrid_runtime import _image_publish_config, _publish_image_bytes
            base, pub = _image_publish_config()
            url = _publish_image_bytes(buf.getvalue(), "image/jpeg", base, pub)
            _URL_CACHE[p] = url
            return {'type': 'image_url', 'image_url': {'url': url}}
        except Exception as exc:
            print(f"[img] URL 发布失败,退回 base64: {type(exc).__name__}: {str(exc)[:90]}", flush=True)
    return {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{encode_image(p, maxw, quality)}'}}


def encode_image(p, maxw=1280, quality=72):
    # 压缩截图: 原始PNG 15张合计~6MB, 退化网关会黑洞大载荷。降到JPEG(~1280px, q72)
    # 后单题载荷~1MB, 判分所需的页面文字/布局仍清晰可读。
    import io
    from PIL import Image
    im = Image.open(p).convert('RGB')
    if im.width > maxw:
        im = im.resize((maxw, int(im.height * maxw / im.width)))
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def build_timeline(r):
    """按步排序的观测时间线。浏览器步取**动作后**那一帧。

    harness 每个浏览器步写两张图:`img_path` 是决策前的页面,`post_shot_path` 是
    动作执行后的页面。原来只取 img_path —— 于是末尾 k 帧里最后一次点击的结果永远
    看不到,判官只能看到"点之前"。WebVoyager 官方 auto_eval 判的是 *result*
    screenshots,M2W 那侧的 WebJudge 也是整个 trajectory/ 目录(两种帧都在)一起打分,
    所以这里改成优先 post_shot_path、缺了才退回 img_path。一步仍只出一帧,k=15
    的含义不变。2026-09-11 改;此前的 65.6%(kimi-k2.6)是旧口径,不可与新数直接比。
    """
    tl = []
    for s in r.get('records', []):
        tool = str((s.get('tool_call') or {}).get('name') or '')
        img = ''
        for cand in (s.get('post_shot_path'), s.get('img_path')):
            if cand and os.path.exists(cand):
                img = cand
                break
        if img:
            tl.append(('img', s.get('step'), None, img))
        elif tool in NONBROWSER_TOOLS:
            obs = str(s.get('observation') or '').strip()
            if obs:
                tl.append(('text', s.get('step'), tool, obs[:3000]))
    return tl


def eval_one(f, client, model, k, outdir):
    r = json.load(open(f))
    tid = r['task_id']
    tl = build_timeline(r)[-k:]
    content = [{'type': 'text', 'text': USER_PROMPT.replace('<task>', r['question'])
                .replace('<answer>', (r.get('prediction') or '')[:4000]).replace('<num>', str(len(tl)))}]
    for kind, step, tool, payload in tl:
        if kind == 'img':
            content.append(image_content(payload))
        else:
            content.append({'type': 'text', 'text': f'[Textual observation at step {step}, tool `{tool}` — same status as a screenshot]\n{payload}'})
    content.append({'type': 'text', 'text': 'Your verdict:\n'})
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': content}]
    last_err = ''
    for attempt in range(6):
        try:
            # 推理模型开关同 baselines/m2w_official/src/utils.py:o 系列预算含推理 token、不收 temperature
            min_comp = int(os.environ.get('JUDGE_MIN_COMPLETION_TOKENS', '0') or 0)
            if min_comp > 0:
                resp = client.chat.completions.create(model=model, messages=messages,
                                                      max_completion_tokens=max(2500, min_comp))
            else:
                resp = client.chat.completions.create(model=model, messages=messages, max_tokens=2500, temperature=0)
            text = resp.choices[0].message.content or ''
            if not text.strip():
                raise RuntimeError(f'empty response finish_reason={resp.choices[0].finish_reason}')
            label = parse_verdict(text)
            # 反爬阻断:独立于 SUCCESS/NOT SUCCESS 的第三个轴。判据与另两套判官共用
            # (evaluation/hywebench/blocked_criterion.py),抠不到记 None 而不是 False ——
            # 把"判官没写"和"没被挡"混成一个数会系统性低估阻断率。
            try:
                import sys as _s
                _s.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hywebench'))
                from blocked_criterion import parse_blocked as _pb
                blocked = _pb(text)
            except Exception:
                blocked = None
            return {'task_id': tid, 'predicted_label': label, 'blocked': blocked, 'response': text,
                    'n_obs': len(tl), 'n_text_obs': sum(1 for x in tl if x[0] == 'text')}
        except Exception as e:
            last_err = f'{type(e).__name__}: {e}'
            time.sleep(10 * (attempt + 1))
    return {'task_id': tid, 'predicted_label': None, 'blocked': None, 'response': f'EVAL_ERROR {last_err}', 'n_obs': len(tl)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', default='runs/base_wv643')
    ap.add_argument('--out', default='runs/base_wv643/eval_wv_official/results.jsonl')
    ap.add_argument('--model', default='gpt-4o')
    ap.add_argument('--base-url', default=os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1'))
    ap.add_argument('--api-key', default=os.environ.get('OPENAI_API_KEY', 'EMPTY'))
    ap.add_argument('--k', type=int, default=15)
    ap.add_argument('--workers', type=int, default=6)
    # 判官端点是共享的,同时挂着 M2W 那边的判分时排队很深:2026-09-11 02:10 实测
    # 一个 8-token 的 `hi` 都要 105 秒。原来写死 timeout=90,于是**每一次调用都超时**,
    # 6 次重试全废、最后写 EVAL_ERROR —— 判分不会报错,只会全成不可判。
    # 单次请求本身也重(15 张图),90 秒即使端点空闲也偏紧。
    ap.add_argument('--timeout', type=float, default=900)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                rr = json.loads(l)
                if rr.get('predicted_label') is not None:   # None-verdict(judge被截断)也要重判
                    done.add(rr['task_id'])
            except Exception:
                pass
    # 只判**终态**轨迹。rollout 边跑边判是允许的(下面有 task_id 断点,后面再跑只补新的),
    # 但半截的必须挡在外面:result.json 是边跑边写的(termination="running"),判了它等于
    # 给一条没做完的轨迹盖章,而且断点之后不会再重判,错误会固化。M2W 那侧靠
    # runs/webjudge_m2w.sh 里的剔除步骤做同一件事,这里直接在源头判。
    files, half = [], 0
    for f in sorted(glob.glob(f'{args.run_dir}/*/result.json')):
        try:
            r = json.load(open(f))
        except Exception:
            half += 1
            continue
        if r.get('termination') in (None, 'running', 'task_error'):
            half += 1
            continue
        if r['task_id'] not in done:
            files.append(f)
    print(f'待判 {len(files)} (已完成 {len(done)},跳过半截/失败 {half})', flush=True)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)
    import threading
    lock = threading.Lock()
    from concurrent.futures import as_completed
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(eval_one, f, client, args.model, args.k, args.out) for f in files]
        done_n = 0
        for fut in as_completed(futs):
            res = fut.result()
            with lock:
                with open(args.out, 'a') as fo:
                    fo.write(json.dumps(res, ensure_ascii=False) + '\n')
            done_n += 1
            if done_n % 20 == 0:
                print(f'progress {done_n}/{len(files)}', flush=True)
    ok = n = unj = 0
    for l in open(args.out):
        rr = json.loads(l); n += 1
        if rr.get('predicted_label') == 1: ok += 1
        elif rr.get('predicted_label') is None: unj += 1
    print(f'总{n} 成功{ok} 不可判{unj} -> 成功率(剔除不可判) {ok/max(n-unj,1):.1%}', flush=True)


if __name__ == '__main__':
    main()

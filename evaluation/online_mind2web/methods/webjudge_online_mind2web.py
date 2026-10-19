from utils import encode_image, image_part
from PIL import Image
import re
import asyncio
MAX_IMAGE =50

# ── 严格来源口径(WEBJUDGE_REQUIRE_SITE=1,2026-09-15 起 insta 单侧对照用)──────────────────
# 实测宽松口径下,S 侧会从目标站以外拿答案也被判成功:insta 判成功的 S 轨迹里 10 道全程没打开过
# 目标站(sportshop.com 的价格取自 passasports.nl、data.fis-ski.com 的赛程取自维基百科……),另有
# 50 道同时访问了目标站和第三方。"只有 O 能做"的实验里这等于允许 S 侧绕过站点,结论就不成立了。
# 不设这个变量时三段文字都为空,M2W / WV 的官方口径判分完全不受影响。
import os as _os
_SITE_STRICT = _os.environ.get("WEBJUDGE_REQUIRE_SITE", "") == "1"
_SITE_CRITERION = """
8: Source of the answer. The task names a target website (the `Web:` line of the task, or the site named in the task text). Every value, item or fact the final answer relies on must be established ON THAT WEBSITE: a page of that website (its registrable domain, its subdomains, or an official domain it redirects to) that the agent actually opened — in the browser or with a webpage reader — or a result the website itself produced in response to the agent's input. Information taken from anywhere else does NOT count, even if it is correct and even if it claims to describe the target website: third-party retailers, marketplaces, aggregators, price-comparison or review sites, news articles, Wikipedia, cached or mirrored copies, and search-engine result snippets (a snippet is the search engine's page, not the website, even when it quotes the website). If any value the task asks for is supported only by off-site evidence, the task is a failure. Do not assume the target website would show the same thing as another source. Check the URLs in the action history and in the textual observations to see where each value really came from.
""" if _SITE_STRICT else ""
_SITE_OBS_NOTE = """
**Source rule** (does not apply to an `answer` observation — that is the agent's own reply; score it only on whether it contains what the task asks for): The task names a target website (the `Web:` line or the site named in the task). Only content that comes from that website can count as evidence. If this observation's content comes from any other source — a search-engine results list (even when its snippets quote the target website), a third-party site, an aggregator, a news article, Wikipedia — it cannot establish anything the task asks for: score it 1 or 2 no matter how relevant its content looks. A webpage-reader result counts only if the URL it read is on the target website.
""" if _SITE_STRICT else ""
_SITE_IMG_NOTE = """
**Source rule**: The task names a target website (the `Web:` line or the site named in the task). If the page in the image is clearly not on that website (e.g. a search engine, a third-party shop, an aggregator, Wikipedia), it cannot serve as evidence: score it 1 or 2.
""" if _SITE_STRICT else ""


def _parse_score(response):
    """取判词里**最后一个** Score 后面的 1-5。

    官方实现是 response.split("Score")[1] —— 切在**第一次**出现 "Score" 的地方。判官在推理里
    引用网页上的字样(如排序选项 "Critic Score")时,切出来的那段里没有 1-5,抛 IndexError,
    被 except 记成 0 分,一条本该 >=3、进终判的证据就被静默扔掉。实测 gpt-4o 0.58% 的逐项打分
    中招(它爱原样引用页面文字),o4-mini 0.04%、kimi-k3 0.03%。格式要求的 **Score** 行在判词
    末尾,取最后一处匹配即可。2026-09-14 改。
    """
    # Score 与数字之间只允许标点/空白/markdown 符号(可跨行):"**Score**: 3"、"### Score:\n**2**" 都能取到;
    # "Critic Score, 95"、"Score of 4 stars" 这类夹着字母或非 1-5 数字的不会误匹配。
    # (第一版写成 [^0-9\n] 不许跨行,gpt-4o 把分数放在标题下一行的 94 条被记成 0 分,已改。)
    hits = re.findall(r"Score[^0-9A-Za-z]{0,20}([1-5])", response)
    if not hits:
        raise ValueError("no score found")
    return int(hits[-1])


async def identify_key_points(task, model):
    system_msg = """You are an expert tasked with analyzing a given task to identify the key points explicitly stated in the task description.

**Objective**: Carefully analyze the task description and extract the critical elements explicitly mentioned in the task for achieving its goal.

**Instructions**:
1. Read the task description carefully.
2. Identify and extract **key points** directly stated in the task description.
   - A **key point** is a critical element, condition, or step explicitly mentioned in the task description.
   - Do not infer or add any unstated elements.
   - Words such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" must go through the sort function(e.g., the key point should be "Filter by highest").

**Respond with**:
- **Key Points**: A numbered list of the explicit key points for completing this task, one per line, without explanations or additional details."""
    prompt = """Task: {task}"""
    text = prompt.format(task=task)
    messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text}
                ],
            }
        ]
    responses = await asyncio.to_thread(model.generate, messages)
    return responses[0]

async def judge_image(task, image_path, key_points, model):
    system_msg = """You are an expert evaluator tasked with determining whether an image contains information about the necessary steps to complete a task.

**Objective**: Analyze the provided image and decide if it shows essential steps or evidence required for completing the task. Use your reasoning to explain your decision before assigning a score.

**Instructions**:
1. Provide a detailed description of the image, including its contents, visible elements, text (if any), and any notable features.

2. Carefully examine the image and evaluate whether it contains necessary steps or evidence crucial to task completion:  
- Identify key points that could be relevant to task completion, such as actions, progress indicators, tool usage, applied filters, or step-by-step instructions.  
- Does the image show actions, progress indicators, or critical information directly related to completing the task?  
- Is this information indispensable for understanding or ensuring task success?
- If the image contains partial but relevant information, consider its usefulness rather than dismissing it outright.

3. Provide your response in the following format:  
- **Reasoning**: Explain your thought process and observations. Mention specific elements in the image that indicate necessary steps, evidence, or lack thereof.  
- **Score**: Assign a score based on the reasoning, using the following scale:  
    - **1**: The image does not contain any necessary steps or relevant information.  
    - **2**: The image contains minimal or ambiguous information, unlikely to be essential.  
    - **3**: The image includes some relevant steps or hints but lacks clarity or completeness.  
    - **4**: The image contains important steps or evidence that are highly relevant but not fully comprehensive.  
    - **5**: The image clearly displays necessary steps or evidence crucial for completing the task.

Respond with:  
1. **Reasoning**: [Your explanation]  
2. **Score**: [1-5]""" + _SITE_IMG_NOTE

    img_part = image_part(image_path)

    prompt = """**Task**: {task}

**Key Points for Task Completion**: {key_points}

The snapshot of the web page is shown in the image."""
    text = prompt.format(task=task,key_points=key_points)

    messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    img_part,
                ],
            }
        ]

    responses = await asyncio.to_thread(model.generate, messages)
    return responses[0]

async def judge_text_obs(task, obs, key_points, model):
    """与 judge_image 同 rubric 的文本观测打分: hybrid agent 的 search/visit/code 等
    非浏览器工具返回, 在我们的范式里与网页截图同等地位的观测证据。"""
    system_msg = """You are an expert evaluator tasked with determining whether a textual observation contains information about the necessary steps to complete a task. The observation is either the returned result of a non-browser tool (a web search, a webpage reader, or a code sandbox), or — when the tool is `answer` — the agent's final textual reply to the user.

**Objective**: Analyze the provided observation text and decide if it shows essential steps or evidence required for completing the task. Use your reasoning to explain your decision before assigning a score.

**Instructions**:
1. Briefly summarize the observation's contents and notable elements.
2. Evaluate whether it contains necessary steps or evidence crucial to task completion (relevant facts, retrieved values, search hits, page content, computation results). For an `answer` observation, evaluate whether the reply contains the concrete information or outcome the task asks for.
3. Provide your response in the following format:
- **Reasoning**: [Your explanation]
- **Score**: [1-5]  (1 = irrelevant; 3 = some relevant hints; 5 = crucial evidence for completing the task)""" + _SITE_OBS_NOTE
    prompt = """**Task**: {task}

**Key Points for Task Completion**: {key_points}

The observation returned by the agent's tool (step {step}, tool `{tool}`):
{text}"""
    text = prompt.format(task=task, key_points=key_points, step=obs.get("step"), tool=obs.get("tool"), text=str(obs.get("text") or "")[:3000])
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]
    responses = await asyncio.to_thread(model.generate, messages)
    return responses[0]


async def WebJudge_Online_Mind2Web_eval(task, last_actions, images_path, model, score_threshold, final_answer=None, text_observations=None):
    system_msg = """You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's task, the agent's action history, key points for task completion, some potentially important web pages in the agent's trajectory and their reasons, your goal is to determine whether the agent has completed the task and achieved all requirements.

Your response must strictly follow the following evaluation criteria!
*Important Evaluation Criteria*:
1: The filtered results must be displayed correctly. If filters were not properly applied (i.e., missing selection, missing confirmation, or no visible effect in results), the task is not considered successful.
2: You must carefully check whether these snapshots and action history meet these key points. Ensure that specific filter conditions, such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" are correctly applied using the filter function(e.g., sort function).
3: Certain key points or requirements should be applied by the filter. Otherwise, a search with all requirements as input will be deemed a failure since it cannot guarantee that all results meet the requirements!
4: If the task requires filtering by a specific range of money, years, or the number of beds and bathrooms, the applied filter must exactly match the given requirement. Any deviation results in failure. To ensure the task is successful, the applied filter must precisely match the specified range without being too broad or too narrow.
Examples of Failure Cases:
- If the requirement is less than $50, but the applied filter is less than $25, it is a failure.
- If the requirement is $1500-$2500, but the applied filter is $2000-$2500, it is a failure.
- If the requirement is $25-$200, but the applied filter is $0-$200, it is a failure.
- If the required years are 2004-2012, but the filter applied is 2001-2012, it is a failure.
- If the required years are before 2015, but the applied filter is 2000-2014, it is a failure.
- If the task requires exactly 2 beds, but the filter applied is 2+ beds, it is a failure.
5: Some tasks require a submission action or a display of results to be considered successful.
6: If the retrieved information is invalid or empty(e.g., No match was found), but the agent has correctly performed the required action, it should still be considered successful.
7: If the current page already displays all available items, then applying a filter is not necessary. As long as the agent selects items that meet the requirements (e.g., the cheapest or lowest price), the task is still considered successful.
""" + _SITE_CRITERION + """
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

*IMPORTANT*
Format your response into three lines as shown below:

Thoughts: <your thoughts and reasoning process based on double-checking each key points and the evaluation criteria>
Status: "success" or "failure"
Blocked: "yes" or "no"   (did a site refuse access, per the Access blocking criterion above; answer independently of Status)
"""
    prompt = """User Task: {task}

Key Points: {key_points}

Action History:
{last_actions}

The potentially important snapshots of the webpage in the agent's trajectory and their reasons:
{thoughts}"""


    key_points = await identify_key_points(task, model)
    key_points = key_points.replace("\n\n", "\n")

    try:
        key_points = key_points.split("**Key Points**:")[1]
        key_points = "\n".join(line.lstrip() for line in key_points.splitlines())
    except:
        key_points = key_points.split("Key Points:")[-1]
        key_points = "\n".join(line.lstrip() for line in key_points.splitlines())
    
    tasks = [judge_image(task, image_path, key_points, model) for image_path in images_path]
    image_responses = await asyncio.gather(*tasks)
    # 文本观测(search/visit/code返回)与截图同权重参与关键观测筛选
    text_observations = text_observations or []
    obs_tasks = [judge_text_obs(task, obs, key_points, model) for obs in text_observations]
    obs_responses = await asyncio.gather(*obs_tasks) if obs_tasks else []

    whole_content_img = []
    whole_thoughts = []
    record = []
    pattern = r"[1-5]"
    for response, image_path in zip(image_responses, images_path):
        try:
            score = _parse_score(response)
            thought = response.split("**Reasoning**:")[-1].strip().lstrip("\n").split("\n\n")[0].replace('\n',' ')
            record.append({"Response": response, "Score": int(score)})
        except Exception as e:
            print(f"Error processing response: {e}")
            score = 0
            record.append({"Response": response, "Score": 0})

        if int(score) >= score_threshold:
            whole_content_img.append(image_part(image_path))
            if thought != "":
                whole_thoughts.append(thought)

    whole_content_img = whole_content_img[:MAX_IMAGE]
    whole_thoughts = whole_thoughts[:MAX_IMAGE]
    # 文本观测按同一阈值入选(与截图同等地位), 记入record便于审计
    selected_obs = []
    for response, obs in zip(obs_responses, text_observations):
        try:
            obs_score = _parse_score(response)
        except Exception:
            obs_score = 0
        record.append({"Response": response, "Score": obs_score, "obs_step": obs.get("step"), "obs_tool": obs.get("tool")})
        if obs_score >= score_threshold:
            selected_obs.append((obs, obs_score))
    selected_obs = selected_obs[:MAX_IMAGE]
    if len(whole_content_img) == 0:
        prompt = """User Task: {task}

Key Points: {key_points}

Action History:
{last_actions}"""
    text = prompt.format(task=task, last_actions="\n".join(f"{i+1}. {action}" for i, action in enumerate(last_actions)), key_points=key_points, thoughts = "\n".join(f"{i+1}. {thought}" for i, thought in enumerate(whole_thoughts)))
    # Patch: 入选的非浏览器工具观测(search/visit/code返回)作为与截图同等地位的页面证据。
    if selected_obs:
        obs_blocks = "\n\n".join(
            f"Step {o.get('step')} [{o.get('tool')}] (relevance {s}/5): {str(o.get('text') or '')[:1500]}"
            for o, s in selected_obs)
        text += ("\n\nKey textual observations from the agent's non-browser tools (web search / page reader / code results). "
                 "Treat them as page-content evidence with the same status as the snapshots:\n" + obs_blocks)
    # (最终答案已统一为 tool='answer' 的观测, 走上面同一条打分-入选通道; final_answer
    # 参数保留兼容, 仅在调用方未提供 text_observations 时兜底使用。)
    if final_answer and not text_observations:
        text += f"\n\nAgent's Final Answer (the agent's textual reply to the user; treat it as part of the outcome evidence):\n{final_answer}"

    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text}]
                + whole_content_img
        }
    ]
    return messages, text, system_msg, record, key_points
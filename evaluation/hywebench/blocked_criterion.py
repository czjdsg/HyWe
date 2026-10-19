#!/usr/bin/env python3
"""三套判官共用的「反爬阻断」判据与解析。

为什么要单独抽一份:阻断率要在 hybrid_traj_judge / WebJudge(Online-M2W) / auto_eval(WebVoyager)
三套判官之间横比,判据措辞必须逐字相同 —— 各写各的,差异里就会混进措辞差异。

**阻断是独立于成功/失败的第三个轴**,不是第三种判决:
  * 一条轨迹可以「被阻断且失败」(最常见)、也可以「被阻断但仍完成」
    (某一跳被挡住,agent 绕开或从别处拿到了同样的结果);
  * 所以判官照常给 success / failure,再**另外**回答一次阻断与否;
  * 汇报成功率时按这个轴分开统计,而不是把阻断从分母里悄悄删掉 ——
    删掉会让"难题被挡得更多"变成"成功率更高"(实测 MMInA 上 multi567 有 67% 被挡,
    直接剔除会把成功率抬高 7 个点,且组成完全变形)。
"""

# 判据正文。三套判官原样嵌进各自的 system prompt,不改一个字。
CRITERION = """## Access blocking (report this separately from success)

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
turned away on one sub-goal and still reached the required end state another way."""

# 各判官在输出格式段追加的一行(措辞随各自的输出风格,语义完全一致)
LINE_M2W = ('Blocked: "yes" or "no"   '
            '(did a site refuse access, per the Access blocking criterion above; '
            'answer independently of Status)')
LINE_WV = ("Blocked: 'YES' or 'NO'   "
           "(did a site refuse access, per the Access blocking criterion above; "
           "answer independently of the verdict)")


def parse_blocked(text: str):
    """从判词里抠 Blocked 行。找不到返回 None —— **不能默认 False**:

    默认 False 会把"判官没写这一行"和"判官说没被挡"混成一个数,阻断率就被系统性低估。
    没写就是没测到,单独计一类。
    """
    if not text:
        return None
    low = str(text).lower()
    i = low.rfind("blocked:")
    if i < 0:
        return None
    tail = low[i + len("blocked:"): i + len("blocked:") + 40]
    for tok, val in (("yes", True), ("true", True), ("no", False), ("false", False)):
        j = tail.find(tok)
        if j >= 0:
            # 取最先出现的那个词
            others = [(tail.find(t), v) for t, v in
                      (("yes", True), ("true", True), ("no", False), ("false", False))
                      if tail.find(t) >= 0]
            return min(others)[1]
    return None


def split_report(rows, label_key="predicted_label", blocked_key="blocked"):
    """按阻断轴分开统计成功率。rows: [{label_key: 0/1/None, blocked_key: True/False/None}]"""
    def rate(sub):
        ok = [r for r in sub if r.get(label_key) is not None]
        s = sum(1 for r in ok if r.get(label_key) == 1)
        return s, len(ok), (s / len(ok) if ok else 0.0)
    blocked = [r for r in rows if r.get(blocked_key) is True]
    clean = [r for r in rows if r.get(blocked_key) is False]
    unknown = [r for r in rows if r.get(blocked_key) is None]
    out = {"n": len(rows)}
    for nm, sub in (("all", rows), ("clean", clean), ("blocked", blocked), ("unknown", unknown)):
        s, n, r = rate(sub)
        out[nm] = {"succ": s, "n": n, "rate": r}
    return out


def format_report(st, title=""):
    L = [f"{title}  n={st['n']}"] if title else [f"n={st['n']}"]
    for nm, zh in (("all", "全量"), ("clean", "未被阻断"), ("blocked", "被阻断"), ("unknown", "阻断未测到")):
        d = st[nm]
        if not d["n"] and nm != "all": continue
        L.append(f"  {zh:10s} {d['succ']:4d}/{d['n']:<4d} = {d['rate']:6.1%}")
    return "\n".join(L)

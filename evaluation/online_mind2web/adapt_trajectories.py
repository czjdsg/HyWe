#!/usr/bin/env python3
"""把 HybridWebAgent 的 M2W 轨迹适配成 Online-Mind2Web 官方评测(v1)布局:
  <out>/<task_id>/result.json  {task, action_history, thoughts, final_result_response,
                                text_observations}
  <out>/<task_id>/trajectory/  编号截图(软链原目录)
text_observations: S侧非浏览器工具(search/visit/code/scholar)的返回观测, 在我们的
混合范式里与截图同等地位, 由打过补丁的 WebJudge 以同一 rubric 打分入选终判。
"""
import argparse
import json, glob, os, sys

NONBROWSER_TOOLS = {'search', 'visit', 'code', 'scholar', 'code_sandbox', 'python'}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run_dir', help='HyWe run directory')
parser.add_argument('out_dir', help='Adapted WebJudge trajectories')
args = parser.parse_args()
src_root = os.path.abspath(args.run_dir)
out_root = os.path.abspath(args.out_dir)
os.makedirs(out_root, exist_ok=True)
n = obs_total = 0
for f in sorted(glob.glob(f'{src_root}/*/result.json')):
    r = json.load(open(f))
    if r.get('termination') in (None, 'running', 'task_error'):
        continue
    tid = r['task_id']
    recs = r.get('records', [])
    traj_src = os.path.join(os.path.dirname(f), 'trajectory')
    # 默认丢掉没有截图的轨迹(M2W 里那是环境报错、0 步就死的题)。
    # 但**锁 S 侧的纯搜索跑一张截图都没有**(实测 insta100_S:100 题全 0 张,
    # 每题 3 次 search/visit),那不是失败而是这一侧的正常形态 —— 打过补丁的
    # WebJudge 会把文本观测与截图同等打分,还有 whole_content_img 为空时的纯文本
    # 分支。所以给个开关:KEEP_EMPTY_TRAJ=1 时保留,交给文本观测去判。
    _keep_empty = os.environ.get('KEEP_EMPTY_TRAJ', '') == '1'
    if not os.path.isdir(traj_src) or not os.listdir(traj_src):
        if not _keep_empty:
            continue
        traj_src = None
    actions, thoughts, text_obs = [], [], []
    for s in recs:
        tc = s.get('tool_call') or {}
        tool = str(tc.get('name') or '')
        desc = (s.get('action_description') or '').strip()
        if not desc:
            desc = f"{tool}({json.dumps(tc.get('arguments') or {}, ensure_ascii=False)[:150]})"
        mode = s.get('active_mode_before') or ''
        actions.append(f"[{mode}] {desc}" if mode else desc)
        rep = s.get('report') or ''
        try:
            rep = json.loads(rep).get('next_action_rationale') or rep
        except Exception:
            pass
        thoughts.append(str(rep)[:500])
        if tool in NONBROWSER_TOOLS:
            obs = str(s.get('observation') or '').strip()
            if obs:
                text_obs.append({'step': s.get('step'), 'tool': tool, 'text': obs[:3000]})
    # 答案步同样作为一条观测(tool='answer'), 与其他观测完全统一走打分-入选通道
    pred = (r.get('prediction') or '').strip()
    if pred:
        text_obs.append({'step': len(recs) + 1, 'tool': 'answer', 'text': pred[:3000]})
    obs_total += len(text_obs)
    od = os.path.join(out_root, tid)
    os.makedirs(od, exist_ok=True)
    dst = os.path.join(od, 'trajectory')
    if os.path.islink(dst):
        os.unlink(dst)
    if traj_src is None:
        os.makedirs(dst, exist_ok=True)          # 空目录:run.py 会列出 0 张图,走纯文本分支
    elif not os.path.exists(dst):
        os.symlink(os.path.abspath(traj_src), dst)
    json.dump({
        # WEBJUDGE_TASK_WITH_WEB=1:与 agent 看到的任务文本一致,末尾带 `Web: <目标站>`
        # (harness 对有 web 的题都这样拼,见 hybrid_agent._v2_public_task_text)。严格来源口径必须开,
        # 否则题面没写站名的题(如 insta "在柏林找二手自行车")判官不知道目标站是哪个。
        'task': (r['question'] + (f"\nWeb: {r.get('web')}" if os.environ.get('WEBJUDGE_TASK_WITH_WEB') == '1'
                                  and str(r.get('web') or '').strip() else '')),
        'action_history': actions,
        'thoughts': thoughts,
        'final_result_response': r.get('prediction') or '',
        'text_observations': text_obs,
    }, open(os.path.join(od, 'result.json'), 'w'), ensure_ascii=False, indent=1)
    n += 1
print(f'adapted {n} tasks -> {out_root} | 非浏览器观测总数: {obs_total}')

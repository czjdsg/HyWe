import sys, os
sys.path.insert(0,'inference')
import failure_summary as f
# 抽取:剥 think/answer 外壳
assert f.extract("<think>x</think>\n<answer>\nFailure type: incomplete\nWhat happened: a\nUseful findings: b\n</answer>").startswith("Failure type")
# 可用性判据
good="<answer>Failure type: incomplete\nWhat happened: a\nUseful findings: b</answer>"
assert f.is_usable(good)
assert not f.is_usable("<answer>Failure type: incomplete</answer>")           # 缺小节
assert not f.is_usable(good.replace("<answer>","<answer><tool_call>{}</tool_call>"))  # 带工具调用
assert not f.is_usable("<answer>依嘉</answer>")                                # 直接吐答案
# 拼装
t=f.compose_task_text("原题", ["Failure type: incomplete\nWhat happened: a\nUseful findings: b"])
assert t.startswith("原题") and "[Attempt 1]" in t and "try a different strategy" in t
assert f.compose_task_text("原题", []) == "原题"                              # 没后验就是原题
t2=f.compose_task_text("原题", ["S1","S2"])
assert "[Attempt 1]" in t2 and "[Attempt 2]" in t2
print("failure_summary 单元校验全过")
print("--- 拼装样例 ---"); print(t[:260])

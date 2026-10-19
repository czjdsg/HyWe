#!/usr/bin/env python3
"""把 O 侧的像素坐标归一化到 Qwen 的 0-1000 网格。

MolmoWeb 的动作是按 1280x720 视口记录的原始像素;Qwen-VL 系列的坐标约定是
0-1000 整数网格。推理端早就支持了(hybrid_utils.coord_scale_for_model 对 qwen*
返回 1000,hybrid_agent._scale_xy 按 `x * vw / scale` 换回真实像素),缺的只是
训练数据这一侧。

两处都要改,少一处就错位:
  1. tool_call 的参数 —— x/y 和 drag 的 from_x/from_y/to_x/to_y;
  2. **CoT 正文里写的坐标** —— 实测 13.7% 的步在思考里点名了坐标
     ("I will click at (739, 131)")。只改 tool_call 不改正文,think 说一个数、
     动作是另一个数,自相矛盾。

正文里三种写法都出现过:(x, y) 11331 步 / (x: a, y: b) 649 步 / x=a y=b 604 步。
判据是**落在视口范围内的数对**,而不是"和本步动作坐标相等" —— 实测有 914 步
提到的是别的元素或先前位置的坐标,同样得换算。
"""
from __future__ import annotations

import re

SRC_W, SRC_H = 1280, 720
GRID = 1000


def norm_x(v, w: int = SRC_W) -> int:
    return max(0, min(GRID - 1, round(float(v) * GRID / w)))


def norm_y(v, h: int = SRC_H) -> int:
    return max(0, min(GRID - 1, round(float(v) * GRID / h)))


_XY = ("x", "y")
_DRAG = (("from_x", "from_y"), ("to_x", "to_y"))


def norm_tool_call(tc: dict, w: int = SRC_W, h: int = SRC_H) -> dict:
    """返回坐标已归一化的 tool_call 副本;没有坐标的原样返回。"""
    if not isinstance(tc, dict):
        return tc
    args = tc.get("arguments")
    if not isinstance(args, dict):
        return tc
    out = dict(args)
    touched = False
    if isinstance(args.get("x"), (int, float)) and isinstance(args.get("y"), (int, float)):
        out["x"], out["y"] = norm_x(args["x"], w), norm_y(args["y"], h)
        touched = True
    for kx, ky in _DRAG:
        if isinstance(args.get(kx), (int, float)) and isinstance(args.get(ky), (int, float)):
            out[kx], out[ky] = norm_x(args[kx], w), norm_y(args[ky], h)
            touched = True
    if not touched:
        return tc
    return {**tc, "arguments": out}


# (739, 131) —— 最常见
_PAREN = re.compile(r"\(\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\)")
# (x: 739, y: 131) / x=739, y=131 —— 允许可选括号与 : 或 =
_LABELED = re.compile(r"(\(?)\s*x\s*[:=]\s*(\d{1,4})\s*,\s*y\s*[:=]\s*(\d{1,4})\s*(\)?)", re.I)
# at 739, 131
_AT = re.compile(r"\b(at\s+)(\d{2,4})\s*,\s*(\d{2,4})\b", re.I)


def _in_view(a: int, b: int, w: int, h: int) -> bool:
    return 0 <= a <= w and 0 <= b <= h


def norm_text(s: str, w: int = SRC_W, h: int = SRC_H) -> str:
    """把正文里的视口坐标换成 0-1000。范围外的数对原样保留 —— 年份、金额、
    尺寸这类括号数对不会落在 [0,1280]x[0,720] 里,顺带就滤掉了。"""
    if not s:
        return s

    def paren(m):
        a, b = int(m.group(1)), int(m.group(2))
        return f"({norm_x(a, w)}, {norm_y(b, h)})" if _in_view(a, b, w, h) else m.group(0)

    def labeled(m):
        a, b = int(m.group(2)), int(m.group(3))
        if not _in_view(a, b, w, h):
            return m.group(0)
        return f"{m.group(1)}x: {norm_x(a, w)}, y: {norm_y(b, h)}{m.group(4)}"

    def at(m):
        a, b = int(m.group(2)), int(m.group(3))
        return f"{m.group(1)}{norm_x(a, w)}, {norm_y(b, h)}" if _in_view(a, b, w, h) else m.group(0)

    s = _LABELED.sub(labeled, s)     # 先做带标签的,免得 (x: 1, y: 2) 被 _PAREN 咬掉半截
    s = _PAREN.sub(paren, s)
    s = _AT.sub(at, s)
    return s

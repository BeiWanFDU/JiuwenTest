"""
0311 模型生成测试用例 —— 全流程 Runner

步骤：
  1. 压缩 user/assistant 原文 → user_compressed / assistant_compressed
  2. 生成各策略答案（jiuwen / no_context / 5+15 / 5+15_3000 / 20_compressed / min_set）
  3. 对各策略答案独立打分
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_core import (
    compress_context,
    answer_by_jiuwen,
    answer_by_no_context,
    answer_by_5_15,
    answer_by_5_15_3000,
    answer_by_20_compressed,
    answer_by_min_set,
    judge_single,
)

XLSX = "../测试用例/0311模型生成测试用例.xlsx"

if __name__ == "__main__":
    # ── 第一阶段：压缩 ────────────────────────────
    # print("====== 压缩上下文 ======")
    # compress_context(XLSX, XLSX)

    # ── 第二阶段：生成答案 ────────────────────────
    # print("====== 策略1：九问上下文 ======")
    # answer_by_jiuwen(XLSX, XLSX)

    # print("====== 策略2：无上下文基线 ======")
    # answer_by_no_context(XLSX, XLSX)

    # print("====== 策略3：5+15 滑动窗口 ======")
    # answer_by_5_15(XLSX, XLSX)

    # print("====== 策略4：5+15（3000 截断）======")
    # answer_by_5_15_3000(XLSX, XLSX)

    # print("====== 策略5：20 轮全压缩 ======")
    # answer_by_20_compressed(XLSX, XLSX)

    print("====== 策略6：一轮原始对话及选择 ======")
    answer_by_min_set(XLSX, XLSX)

    # ── 第三阶段：评分 ────────────────────────────
    # print("====== 开始评分 ======")
    # judge_single(XLSX, XLSX, answer_col="answer_jiuwen",        score_col_prefix="score_jiuwen")
    # judge_single(XLSX, XLSX, answer_col="answer_no_context",    score_col_prefix="score_no_context")
    # judge_single(XLSX, XLSX, answer_col="answer_5+15",          score_col_prefix="score_5+15")
    # judge_single(XLSX, XLSX, answer_col="answer_5+15_3000",     score_col_prefix="score_5+15_3000")
    # judge_single(XLSX, XLSX, answer_col="answer_20_compressed", score_col_prefix="score_20_compressed")

    print("====== 开始评分：一轮原始对话及选择 ======")
    judge_single(XLSX, XLSX, answer_col="answer_1_and_select", score_col_prefix="score_1_and_select")

    print("**********************************")
    print("全部完成")
    print("**********************************")
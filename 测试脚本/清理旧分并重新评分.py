import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from 单答案评分 import judge_single

XLSX = "../测试用例/0311模型生成测试用例.xlsx"

# 清除旧打分列
df = pd.read_excel(XLSX, sheet_name='Sheet1', engine='openpyxl')
old_score_prefixes = ["score_jiuwen", "score_5+15", "score_no_context",
                      "score_jiuwen_10轮", "score_jiuwen_20轮", "score_5+15_3000"]
old_dims = ["_事实一致性", "_无矛盾性", "_意图覆盖度", "_上下文利用准确性",
            "_回答完整性", "_回答准确性", "_reason"]
old_cols = [p + d for p in old_score_prefixes for d in old_dims]
df = df.drop(columns=[c for c in old_cols if c in df.columns])
df.to_excel(XLSX, sheet_name='Sheet1', index=False)
print("旧打分列已清除")

# 重新打分
judge_single(XLSX, XLSX, answer_col="answer_jiuwen_10轮", score_col_prefix="score_jiuwen_10轮")
judge_single(XLSX, XLSX, answer_col="answer_jiuwen_20轮", score_col_prefix="score_jiuwen_20轮")
judge_single(XLSX, XLSX, answer_col="answer_no_context",  score_col_prefix="score_no_context")
judge_single(XLSX, XLSX, answer_col="answer_5+15",        score_col_prefix="score_5+15")
judge_single(XLSX, XLSX, answer_col="answer_5+15_3000",   score_col_prefix="score_5+15_3000")

print("**********************************")
print("全部完成")
print("**********************************")
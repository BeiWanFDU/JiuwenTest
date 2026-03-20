"""
0307AISF精选测试用例 全流程脚本
策略：jiuwen / no_context / 5+15 / 5+15_3000 / 20_compressed
流程：生成答案 → 打分（均支持断点续跑）

与 0311 的主要差异：
  - 上下文列为 context_content_jiuwen（单列，无10/20轮拆分）
  - assistant_compressed 列已预先填充
"""

import os
import sys

import pandas as pd
import tiktoken

import 模型调用脚本_外部模型

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from 单答案评分 import judge_single

XLSX = "../测试用例/0307AISF精选测试用例.xlsx"

# ─────────────────────────────────────────────
# 上下文构建辅助函数
# ─────────────────────────────────────────────

def build_5_15_context(df, current_index):
    """按 Excel 行顺序（跨对话）取前最多20轮：近5轮原文，6-20轮压缩"""
    prev_indices = [idx for idx in df.index if idx < current_index]
    context = []
    n = len(prev_indices)
    for i, row_idx in enumerate(prev_indices):
        distance = n - i
        if distance > 20:
            continue
        user_content = str(df.at[row_idx, "user"])
        if distance <= 5:
            assistant_content = str(df.at[row_idx, "assistant"])
        else:
            assistant_content = str(df.at[row_idx, "assistant_compressed"])
        context.append({"user": user_content, "assistant": assistant_content})

    token_count = len(tiktoken.encoding_for_model("gpt-4").encode(str(context)))
    return context, token_count


def build_5_15_context_3000(df, current_index):
    """同 build_5_15_context，但从最近一轮往前累加 token，超过 3000 则截断"""
    prev_indices = [idx for idx in df.index if idx < current_index]
    candidates = []
    n = len(prev_indices)
    for i, row_idx in enumerate(prev_indices):
        distance = n - i
        if distance > 20:
            continue
        user_content = str(df.at[row_idx, "user"])
        if distance <= 5:
            assistant_content = str(df.at[row_idx, "assistant"])
        else:
            assistant_content = str(df.at[row_idx, "assistant_compressed"])
        candidates.append({"user": user_content, "assistant": assistant_content})

    encoding = tiktoken.encoding_for_model("gpt-4")
    token_budget = 3000
    selected = []
    for item in reversed(candidates):
        item_tokens = len(encoding.encode(str(item)))
        if token_budget - item_tokens < 0:
            break
        token_budget -= item_tokens
        selected.insert(0, item)

    token_count = len(encoding.encode(str(selected)))
    return selected, token_count


def build_20_compressed_context(df, current_index):
    """按 Excel 行顺序（跨对话）取前最多20轮，assistant 全部使用压缩摘要"""
    prev_indices = [idx for idx in df.index if idx < current_index]
    context = []
    n = len(prev_indices)
    for i, row_idx in enumerate(prev_indices):
        distance = n - i
        if distance > 20:
            continue
        user_content = str(df.at[row_idx, "user"])
        assistant_content = str(df.at[row_idx, "assistant_compressed"])
        context.append({"user": user_content, "assistant": assistant_content})

    token_count = len(tiktoken.encoding_for_model("gpt-4").encode(str(context)))
    return context, token_count


# ─────────────────────────────────────────────
# 答案生成
# ─────────────────────────────────────────────

def _load_system_prompt():
    with open("../prompts/调用qwen_32b回答.txt", "r", encoding="utf-8") as f:
        return f.read()


def answer_jiuwen(input_path: str, output_path: str):
    df = pd.read_excel(input_path, sheet_name="Sheet1", engine="openpyxl")
    system_prompt = _load_system_prompt()

    for index, row in df.iterrows():
        user_prompt = f"""
上下文：{df.at[index, "context_content_jiuwen"]}
用户当前query：{df.at[index, "user"]}
        """
        response = 模型调用脚本_外部模型.invoke_model(system_prompt, user_prompt, "qwen3-32b")

        df.at[index, "answer_jiuwen"] = response
        df.at[index, "tokens_jiuwen"] = len(
            tiktoken.encoding_for_model("gpt-4").encode(str(df.at[index, "context_content_jiuwen"]))
        )
        print(f"user: {df.at[index, 'user']}")
        print(f"answer_jiuwen: {response}")
        df.to_excel(output_path, sheet_name="Sheet1", index=False)
        print("**********************************")
        print(f"第{index}行处理完成")
        print("**********************************")


def answer_no_context(input_path: str, output_path: str):
    df = pd.read_excel(input_path, sheet_name="Sheet1", engine="openpyxl")
    system_prompt = _load_system_prompt()

    for index, row in df.iterrows():

        user_prompt = f"""
上下文：[]
用户当前query：{df.at[index, "user"]}
        """
        response = 模型调用脚本_外部模型.invoke_model(system_prompt, user_prompt, "qwen3-32b")

        df.at[index, "answer_no_context"] = response
        print(f"user: {df.at[index, 'user']}")
        print(f"answer_no_context: {response}")
        df.to_excel(output_path, sheet_name="Sheet1", index=False)
        print("**********************************")
        print(f"第{index}行处理完成")
        print("**********************************")


def answer_5_and_15(input_path: str, output_path: str):
    df = pd.read_excel(input_path, sheet_name="Sheet1", engine="openpyxl")
    system_prompt = _load_system_prompt()

    for index, row in df.iterrows():
        context, token_count = build_5_15_context(df, index)
        user_prompt = f"""
上下文：{str(context)}
用户当前query：{df.at[index, "user"]}
        """
        response = 模型调用脚本_外部模型.invoke_model(system_prompt, user_prompt, "qwen3-32b")

        df.at[index, "answer_5+15"] = response
        df.at[index, "tokens_5+15"] = token_count
        print(f"user: {df.at[index, 'user']}")
        print(f"context轮数: {len(context)} | tokens: {token_count}")
        print(f"answer_5+15: {response}")
        df.to_excel(output_path, sheet_name="Sheet1", index=False)
        print("**********************************")
        print(f"第{index}行处理完成")
        print("**********************************")


def answer_5_and_15_3000(input_path: str, output_path: str):
    df = pd.read_excel(input_path, sheet_name="Sheet1", engine="openpyxl")
    system_prompt = _load_system_prompt()

    for index, row in df.iterrows():
        context, token_count = build_5_15_context_3000(df, index)
        user_prompt = f"""
上下文：{str(context)}
用户当前query：{df.at[index, "user"]}
        """
        response = 模型调用脚本_外部模型.invoke_model(system_prompt, user_prompt, "qwen3-32b")

        df.at[index, "answer_5+15_3000"] = response
        df.at[index, "tokens_5+15_3000"] = token_count
        print(f"user: {df.at[index, 'user']}")
        print(f"context轮数: {len(context)} | tokens: {token_count}")
        print(f"answer_5+15_3000: {response}")
        df.to_excel(output_path, sheet_name="Sheet1", index=False)
        print("**********************************")
        print(f"第{index}行处理完成")
        print("**********************************")


def answer_20_compressed(input_path: str, output_path: str):
    df = pd.read_excel(input_path, sheet_name="Sheet1", engine="openpyxl")
    if "answer_20_compressed" in df.columns:
        df["answer_20_compressed"] = df["answer_20_compressed"].astype(object)
    system_prompt = _load_system_prompt()

    for index, row in df.iterrows():
        context, token_count = build_20_compressed_context(df, index)
        user_prompt = f"""
上下文：{str(context)}
用户当前query：{df.at[index, "user"]}
        """
        response = 模型调用脚本_外部模型.invoke_model(system_prompt, user_prompt, "qwen3-32b")

        df.at[index, "answer_20_compressed"] = response
        df.at[index, "tokens_20_compressed"] = token_count
        print(f"user: {df.at[index, 'user']}")
        print(f"context轮数: {len(context)} | tokens: {token_count}")
        print(f"answer_20_compressed: {response}")
        df.to_excel(output_path, sheet_name="Sheet1", index=False)
        print("**********************************")
        print(f"第{index}行处理完成")
        print("**********************************")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 第一阶段：生成各策略回答
    print("====== 策略1：jiuwen上下文 ======")
    answer_jiuwen(XLSX, XLSX)

    print("====== 策略2：无上下文基线 ======")
    answer_no_context(XLSX, XLSX)

    print("====== 策略3：5+15滑动窗口 ======")
    answer_5_and_15(XLSX, XLSX)

    print("====== 策略4：5+15（3000截断）======")
    answer_5_and_15_3000(XLSX, XLSX)

    print("====== 策略5：20轮全压缩 ======")
    answer_20_compressed(XLSX, XLSX)

    # 第二阶段：打分（复用 单答案评分.judge_single，参考答案列已统一为 assistant）
    print("====== 开始评分 ======")
    judge_single(XLSX, XLSX, answer_col="answer_jiuwen",        score_col_prefix="score_jiuwen")
    judge_single(XLSX, XLSX, answer_col="answer_no_context",    score_col_prefix="score_no_context")
    judge_single(XLSX, XLSX, answer_col="answer_5+15",          score_col_prefix="score_5+15")
    judge_single(XLSX, XLSX, answer_col="answer_5+15_3000",     score_col_prefix="score_5+15_3000")
    judge_single(XLSX, XLSX, answer_col="answer_20_compressed", score_col_prefix="score_20_compressed")

    print("**********************************")
    print("全部完成")
    print("**********************************")
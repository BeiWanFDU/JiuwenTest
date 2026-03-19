"""
重新生成压缩相关策略的全流程脚本（不跳过已有内容，全量覆盖更新）
流程：
  1. 重新生成 user_compressed / assistant_compressed（全量覆盖）
  2. 重新生成 answer_5+15 / answer_20_compressed（全量覆盖）
  3. 重新对这两个策略打分（全量覆盖）
"""

import json
import os
import sys

import pandas as pd
import tiktoken

import 模型调用脚本_外部模型

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from 单答案评分 import judge_single

XLSX = "../测试用例/0311模型生成测试用例.xlsx"


# ─────────────────────────────────────────────
# 第一步：重新压缩（全量覆盖）
# ─────────────────────────────────────────────

def recompress_all(input_path: str, output_path: str):
    df = pd.read_excel(input_path, sheet_name='Sheet1', engine='openpyxl')

    with open("../prompts/压缩.txt", 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    for index, row in df.iterrows():
        user_prompt = json.dumps([
            {"role": "user", "content": df.at[index, "user"]},
            {"role": "assistant", "content": df.at[index, "assistant"]}
        ], ensure_ascii=False)

        response = 模型调用脚本_外部模型.invoke_model(system_prompt, user_prompt, "qwen3.5-27b")

        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            print(f"JSON解析失败，跳过index {index}，原始response：{response}")
            continue
        except Exception as e:
            print(f"解析失败，跳过index {index}，错误：{e}，原始response：{response}")
            continue

        abstract = response.get("abstract", [])
        user_compressed = next((item["content"] for item in abstract if item["role"] == "user"), "")
        assistant_compressed = next((item["content"] for item in abstract if item["role"] == "assistant"), "")

        df.at[index, "user_compressed"] = user_compressed
        df.at[index, "assistant_compressed"] = assistant_compressed

        print(f"user_compressed: {user_compressed}")
        print(f"assistant_compressed: {assistant_compressed}")
        df.to_excel(output_path, sheet_name='Sheet1', index=False)
        print("**********************************")
        print(f"第{index}轮处理完成")
        print("**********************************")


# ─────────────────────────────────────────────
# 第二步：重新生成答案（全量覆盖）
# ─────────────────────────────────────────────

def build_5_15_context(df, current_index):
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


def build_20_compressed_context(df, current_index):
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


def reanswer_5_and_15(input_path: str, output_path: str):
    df = pd.read_excel(input_path, sheet_name='Sheet1', engine='openpyxl')

    with open("../prompts/调用qwen_32b回答.txt", 'r', encoding='utf-8') as f:
        system_prompt = f.read()

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
        df.to_excel(output_path, sheet_name='Sheet1', index=False)
        print("**********************************")
        print(f"第{index}轮处理完成")
        print("**********************************")


def reanswer_20_compressed(input_path: str, output_path: str):
    df = pd.read_excel(input_path, sheet_name='Sheet1', engine='openpyxl')
    df["answer_20_compressed"] = df["answer_20_compressed"].astype(object)

    with open("../prompts/调用qwen_32b回答.txt", 'r', encoding='utf-8') as f:
        system_prompt = f.read()

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
        df.to_excel(output_path, sheet_name='Sheet1', index=False)
        print("**********************************")
        print(f"第{index}轮处理完成")
        print("**********************************")


# ─────────────────────────────────────────────
# 第三步：清除旧分并重新打分（全量覆盖）
# ─────────────────────────────────────────────

def rescore(input_path: str, output_path: str):
    df = pd.read_excel(input_path, sheet_name='Sheet1', engine='openpyxl')

    old_prefixes = ["score_5+15", "score_20_compressed"]
    old_dims = ["_上下文利用准确性", "_回答完整性", "_回答准确性", "_reason"]
    old_cols = [p + d for p in old_prefixes for d in old_dims]
    df = df.drop(columns=[c for c in old_cols if c in df.columns])
    df.to_excel(output_path, sheet_name='Sheet1', index=False)
    print("旧打分列已清除，开始重新打分")

    judge_single(output_path, output_path, answer_col="answer_5+15",          score_col_prefix="score_5+15")
    judge_single(output_path, output_path, answer_col="answer_20_compressed",  score_col_prefix="score_20_compressed")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # print("====== 第一步：重新压缩 user_compressed / assistant_compressed ======")
    # recompress_all(XLSX, XLSX)

    print("====== 第二步：重新生成 answer_5+15 ======")
    reanswer_5_and_15(XLSX, XLSX)

    print("====== 第二步：重新生成 answer_20_compressed ======")
    reanswer_20_compressed(XLSX, XLSX)

    print("====== 第三步：清除旧分并重新打分 ======")
    rescore(XLSX, XLSX)

    print("**********************************")
    print("全部完成")
    print("**********************************")

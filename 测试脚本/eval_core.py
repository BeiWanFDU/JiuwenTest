"""
多轮对话上下文策略评测 —— 核心函数库

包含以下能力：
  - 上下文构建（5+15 / 5+15截断 / 20轮全压缩）
  - 对话压缩（生成 user_compressed / assistant_compressed）
  - 答案生成（jiuwen / no_context / 5+15 / 5+15_3000 / 20_compressed）
  - 评分（单答案独立打分）

所有答案生成与评分函数均支持 force 参数：
  force=True （默认）：全量覆盖，重新生成所有行
  force=False        ：断点续跑，跳过已有内容
"""

import json
import os

import pandas as pd
import tiktoken

import 模型调用脚本_外部模型

# prompt 文件所在目录（相对于本文件）
_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompts")


def _load_prompt(filename: str) -> str:
    """从 prompts 目录加载 system prompt 文本。"""
    with open(os.path.join(_PROMPTS_DIR, filename), "r", encoding="utf-8") as f:
        return f.read()


def _ensure_object_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """确保指定列为 object 类型，避免向 float64 列写入字符串时报错。"""
    if col in df.columns:
        df[col] = df[col].astype(object)
    return df


def _should_skip(df: pd.DataFrame, index, col: str, force: bool) -> bool:
    """判断当前行是否应跳过：force=True 时永不跳过；force=False 时已有内容则跳过。"""
    if force:
        return False
    return col in df.columns and pd.notna(df.at[index, col]) and df.at[index, col] != ""


# ─────────────────────────────────────────────
# 上下文构建
# ─────────────────────────────────────────────

def build_5_15_context(df: pd.DataFrame, current_index) -> tuple:
    """
    按 Excel 行顺序（跨对话）取当前行之前最多 20 轮构建 5+15 上下文：
      - 最近 5 轮：使用 assistant 原文
      - 第 6～20 轮：使用 assistant_compressed 压缩摘要
      - user 始终使用原文

    返回 (context: list[dict], token_count: int)
    """
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


def build_5_15_context_3000(df: pd.DataFrame, current_index) -> tuple:
    """
    与 build_5_15_context 逻辑相同，但从最近一轮往前累加 token，
    超过 3000 则停止，保留最近的完整轮次（对话级截断，不拆分单轮）。

    返回 (context: list[dict], token_count: int)
    """
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


def build_20_compressed_context(df: pd.DataFrame, current_index) -> tuple:
    """
    按 Excel 行顺序（跨对话）取当前行之前最多 20 轮，
    user 使用原文，assistant 全部使用 assistant_compressed 压缩摘要。

    返回 (context: list[dict], token_count: int)
    """
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
# 对话压缩
# ─────────────────────────────────────────────

def compress_context(input_path: str, output_path: str, force: bool = True):
    """
    调用压缩模型（qwen3.5-27b）将每轮 user/assistant 原文压缩为摘要，
    结果写入 user_compressed / assistant_compressed 列。

    prompt 文件：prompts/压缩.txt
    force=True：全量重新压缩；force=False：跳过已有内容。
    """
    df = pd.read_excel(input_path, sheet_name="Sheet1", engine="openpyxl")
    system_prompt = _load_prompt("压缩.txt")

    for index, row in df.iterrows():
        if _should_skip(df, index, "assistant_compressed", force):
            print(f"第{index}行已有压缩结果，跳过")
            continue

        user_prompt = json.dumps([
            {"role": "user",      "content": df.at[index, "user"]},
            {"role": "assistant", "content": df.at[index, "assistant"]}
        ], ensure_ascii=False)

        response = 模型调用脚本_外部模型.invoke_model(system_prompt, user_prompt, "qwen3.5-27b")

        try:
            response = json.loads(response)
        except (json.JSONDecodeError, Exception) as e:
            print(f"压缩解析失败，跳过index {index}：{e}，原始response：{response}")
            continue

        abstract = response.get("abstract", [])
        df.at[index, "user_compressed"]      = next((x["content"] for x in abstract if x["role"] == "user"), "")
        df.at[index, "assistant_compressed"] = next((x["content"] for x in abstract if x["role"] == "assistant"), "")

        print(f"[{index}] user_compressed: {df.at[index, 'user_compressed']}")
        print(f"[{index}] assistant_compressed: {df.at[index, 'assistant_compressed']}")
        df.to_excel(output_path, sheet_name="Sheet1", index=False)
        print(f"第{index}行压缩完成")


def build_min_set_context(df: pd.DataFrame, current_index) -> tuple:
    """
    构建 min_set + 最近一轮 混合上下文：
      - 始终包含紧邻前一行（原文，user + assistant），跨 session 也取
      - 若 min_set 非空，额外加入标注轮次（user 原文 + assistant_compressed）
      - 去重：若紧邻前一行的 context_id 在 min_set 中，以原文为准不重复添加
      - 最终按 Excel 行号升序排列

    返回 (context: list[dict], token_count: int)
    """
    prev_indices = [idx for idx in df.index if idx < current_index]
    if not prev_indices:
        return [], 0

    last_idx = prev_indices[-1]

    # 解析 min_set（逗号分隔的 context_id 列表）
    min_set_ids = set()
    min_set_val = df.at[current_index, "min_set"]
    if pd.notna(min_set_val) and str(min_set_val).strip():
        for cid in str(min_set_val).split(","):
            min_set_ids.add(cid.strip())

    # 用 dict 以行号为 key 收集，便于去重
    context_rows = {}

    # 最近一轮：始终用原文
    context_rows[last_idx] = {
        "user":      str(df.at[last_idx, "user"]),
        "assistant": str(df.at[last_idx, "assistant"]),
    }

    # min_set 标注轮次：user 原文 + assistant_compressed
    # 若已被最近一轮占用（原文优先），跳过
    if min_set_ids:
        for idx in prev_indices:
            cid = str(df.at[idx, "context_id"])
            if cid in min_set_ids and idx not in context_rows:
                context_rows[idx] = {
                    "user":      str(df.at[idx, "user"]),
                    "assistant": str(df.at[idx, "assistant_compressed"]),
                }

    # 按行号升序，保持时间顺序
    context = [context_rows[i] for i in sorted(context_rows.keys())]
    token_count = len(tiktoken.encoding_for_model("gpt-4").encode(str(context)))
    return context, token_count


# ─────────────────────────────────────────────
# 答案生成
# ─────────────────────────────────────────────

def answer_by_jiuwen(
    input_path: str,
    output_path: str,
    context_col: str = "context_content_jiuwen",
    answer_col: str  = "answer_jiuwen",
    token_col: str   = "tokens_jiuwen",
    force: bool      = True,
):
    """
    策略：九问检索上下文。
    从 context_col 列读取九问服务返回的上下文，调用 qwen3-32b 生成答案。
    同步统计 context_col 的 token 数写入 token_col。

    force=True：全量覆盖；force=False：断点续跑。
    """
    df = pd.read_excel(input_path, sheet_name="Sheet1", engine="openpyxl")
    df = _ensure_object_col(df, answer_col)
    system_prompt = _load_prompt("调用qwen_32b回答.txt")

    for index, row in df.iterrows():
        if _should_skip(df, index, answer_col, force):
            print(f"第{index}行已有结果，跳过")
            continue

        user_prompt = f"\n上下文：{df.at[index, context_col]}\n用户当前query：{df.at[index, 'user']}\n"
        response = 模型调用脚本_外部模型.invoke_model(system_prompt, user_prompt, "qwen3-32b")

        df.at[index, answer_col] = response
        df.at[index, token_col]  = len(tiktoken.encoding_for_model("gpt-4").encode(
            str(df.at[index, context_col])
        ))
        print(f"[{index}] user: {df.at[index, 'user']}")
        print(f"[{index}] {answer_col}: {response}")
        df.to_excel(output_path, sheet_name="Sheet1", index=False)
        print(f"第{index}行处理完成")


def answer_by_no_context(
    input_path: str,
    output_path: str,
    answer_col: str = "answer_no_context",
    force: bool     = True,
):
    """
    策略：无上下文基线。
    传入空上下文，模型完全依赖自身知识作答，用于建立得分下界。

    force=True：全量覆盖；force=False：断点续跑。
    """
    df = pd.read_excel(input_path, sheet_name="Sheet1", engine="openpyxl")
    df = _ensure_object_col(df, answer_col)
    system_prompt = _load_prompt("调用qwen_32b回答.txt")

    for index, row in df.iterrows():
        if _should_skip(df, index, answer_col, force):
            print(f"第{index}行已有结果，跳过")
            continue

        user_prompt = f"\n上下文：[]\n用户当前query：{df.at[index, 'user']}\n"
        response = 模型调用脚本_外部模型.invoke_model(system_prompt, user_prompt, "qwen3-32b")

        df.at[index, answer_col] = response
        print(f"[{index}] user: {df.at[index, 'user']}")
        print(f"[{index}] {answer_col}: {response}")
        df.to_excel(output_path, sheet_name="Sheet1", index=False)
        print(f"第{index}行处理完成")


def answer_by_5_15(
    input_path: str,
    output_path: str,
    answer_col: str = "answer_5+15",
    token_col: str  = "tokens_5+15",
    force: bool     = True,
):
    """
    策略：5+15 滑动窗口。
    近5轮使用 assistant 原文，6-20轮使用 assistant_compressed 压缩摘要，
    跨对话按 Excel 行顺序取上下文。同步统计 token 数写入 token_col。

    force=True：全量覆盖；force=False：断点续跑。
    """
    df = pd.read_excel(input_path, sheet_name="Sheet1", engine="openpyxl")
    df = _ensure_object_col(df, answer_col)
    system_prompt = _load_prompt("调用qwen_32b回答.txt")

    for index, row in df.iterrows():
        if _should_skip(df, index, answer_col, force):
            print(f"第{index}行已有结果，跳过")
            continue

        context, token_count = build_5_15_context(df, index)
        user_prompt = f"\n上下文：{str(context)}\n用户当前query：{df.at[index, 'user']}\n"
        response = 模型调用脚本_外部模型.invoke_model(system_prompt, user_prompt, "qwen3-32b")

        df.at[index, answer_col] = response
        df.at[index, token_col]  = token_count
        print(f"[{index}] user: {df.at[index, 'user']}")
        print(f"[{index}] context轮数: {len(context)} | tokens: {token_count}")
        print(f"[{index}] {answer_col}: {response}")
        df.to_excel(output_path, sheet_name="Sheet1", index=False)
        print(f"第{index}行处理完成")


def answer_by_5_15_3000(
    input_path: str,
    output_path: str,
    answer_col: str = "answer_5+15_3000",
    token_col: str  = "tokens_5+15_3000",
    force: bool     = True,
):
    """
    策略：5+15 滑动窗口（3000 token 截断）。
    构建逻辑与 5+15 相同，但从最近一轮往前累加 token，
    超过 3000 则截断，保留最近的完整轮次（对话级截断）。

    force=True：全量覆盖；force=False：断点续跑。
    """
    df = pd.read_excel(input_path, sheet_name="Sheet1", engine="openpyxl")
    df = _ensure_object_col(df, answer_col)
    system_prompt = _load_prompt("调用qwen_32b回答.txt")

    for index, row in df.iterrows():
        if _should_skip(df, index, answer_col, force):
            print(f"第{index}行已有结果，跳过")
            continue

        context, token_count = build_5_15_context_3000(df, index)
        user_prompt = f"\n上下文：{str(context)}\n用户当前query：{df.at[index, 'user']}\n"
        response = 模型调用脚本_外部模型.invoke_model(system_prompt, user_prompt, "qwen3-32b")

        df.at[index, answer_col] = response
        df.at[index, token_col]  = token_count
        print(f"[{index}] user: {df.at[index, 'user']}")
        print(f"[{index}] context轮数: {len(context)} | tokens: {token_count}")
        print(f"[{index}] {answer_col}: {response}")
        df.to_excel(output_path, sheet_name="Sheet1", index=False)
        print(f"第{index}行处理完成")


def answer_by_20_compressed(
    input_path: str,
    output_path: str,
    answer_col: str = "answer_20_compressed",
    token_col: str  = "tokens_20_compressed",
    force: bool     = True,
):
    """
    策略：20 轮全压缩。
    取前最多 20 轮，user 使用原文，assistant 全部使用 assistant_compressed 压缩摘要。
    用于验证近轮保留原文的必要性（与 5+15 对比）。

    force=True：全量覆盖；force=False：断点续跑。
    """
    df = pd.read_excel(input_path, sheet_name="Sheet1", engine="openpyxl")
    df = _ensure_object_col(df, answer_col)
    system_prompt = _load_prompt("调用qwen_32b回答.txt")

    for index, row in df.iterrows():
        if _should_skip(df, index, answer_col, force):
            print(f"第{index}行已有结果，跳过")
            continue

        context, token_count = build_20_compressed_context(df, index)
        user_prompt = f"\n上下文：{str(context)}\n用户当前query：{df.at[index, 'user']}\n"
        response = 模型调用脚本_外部模型.invoke_model(system_prompt, user_prompt, "qwen3-32b")

        df.at[index, answer_col] = response
        df.at[index, token_col]  = token_count
        print(f"[{index}] user: {df.at[index, 'user']}")
        print(f"[{index}] context轮数: {len(context)} | tokens: {token_count}")
        print(f"[{index}] {answer_col}: {response}")
        df.to_excel(output_path, sheet_name="Sheet1", index=False)
        print(f"第{index}行处理完成")


# ─────────────────────────────────────────────
# 评分
# ─────────────────────────────────────────────

def judge_single(
    input_path: str,
    output_path: str,
    answer_col: str,
    score_col_prefix: str,
    force: bool = True,
):
    """
    单答案独立评分（LLM-as-judge，裁判模型 qwen-max）。

    以 guide（答题要求）为核心评判依据，assistant 为参考答案（非唯一标准），
    对 answer_col 列的被测答案按三个维度独立打分（1-5分）：
      - 上下文利用准确性
      - 回答完整性
      - 回答准确性

    结果写入以 score_col_prefix 为前缀的四列：
      {prefix}_上下文利用准确性 / _回答完整性 / _回答准确性 / _reason

    force=True：全量覆盖；force=False：断点续跑。
    """
    df = pd.read_excel(input_path, sheet_name="Sheet1", engine="openpyxl")
    system_prompt = _load_prompt("单答案打分.txt")
    check_col = f"{score_col_prefix}_回答准确性"

    for index, row in df.iterrows():
        if _should_skip(df, index, check_col, force):
            print(f"第{index}行已有评分，跳过")
            continue

        user_prompt = (
            f"\n答题要求：{df.at[index, 'guide']}"
            f"\n用户问题：{df.at[index, 'user']}"
            f"\n参考答案：{df.at[index, 'assistant']}"
            f"\n被测答案：{df.at[index, answer_col]}\n"
        )
        response = 模型调用脚本_外部模型.invoke_model(system_prompt, user_prompt, "qwen-max")

        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            print(f"JSON解析失败，跳过index {index}，原始response：{response}")
            continue
        except Exception as e:
            print(f"解析失败，跳过index {index}，错误：{e}，原始response：{response}")
            continue

        df.at[index, f"{score_col_prefix}_上下文利用准确性"] = response["上下文利用准确性"]
        df.at[index, f"{score_col_prefix}_回答完整性"]       = response["回答完整性"]
        df.at[index, f"{score_col_prefix}_回答准确性"]       = response["回答准确性"]
        df.at[index, f"{score_col_prefix}_reason"]           = response["reason"]

        print(f"[{index}] user: {df.at[index, 'user']}")
        print(f"[{index}] {score_col_prefix}: {response}")
        df.to_excel(output_path, sheet_name="Sheet1", index=False)
        print(f"第{index}行评分完成")


def answer_by_min_set(
    input_path: str,
    output_path: str,
    answer_col: str = "answer_1_and_select",
    token_col: str  = "tokens_1_and_select",
    force: bool     = True,
):
    """
    策略：min_set 标注上下文 + 最近一轮原文。

    上下文构建规则（见 build_min_set_context）：
      - 紧邻前一行始终以原文加入（user + assistant），跨 session 也取
      - min_set 列标注的轮次以 user 原文 + assistant_compressed 加入
      - 若最近一轮 context_id 在 min_set 中，原文优先不重复
      - 按行号升序排列

    force=True：全量覆盖；force=False：断点续跑。
    """
    df = pd.read_excel(input_path, sheet_name="Sheet1", engine="openpyxl")
    df = _ensure_object_col(df, answer_col)
    system_prompt = _load_prompt("调用qwen_32b回答.txt")

    for index, row in df.iterrows():
        if _should_skip(df, index, answer_col, force):
            print(f"第{index}行已有结果，跳过")
            continue

        context, token_count = build_min_set_context(df, index)
        user_prompt = f"\n上下文：{str(context)}\n用户当前query：{df.at[index, 'user']}\n"
        response = 模型调用脚本_外部模型.invoke_model(system_prompt, user_prompt, "qwen3-32b")

        df.at[index, answer_col] = response
        df.at[index, token_col]  = token_count
        print(f"[{index}] user: {df.at[index, 'user']}")
        print(f"[{index}] context轮数: {len(context)} | tokens: {token_count}")
        print(f"[{index}] {answer_col}: {response}")
        df.to_excel(output_path, sheet_name="Sheet1", index=False)
        print(f"第{index}行处理完成")
import pandas as pd
import tiktoken

import 模型调用脚本_外部模型


def answer_by_qwen32b_jiuwen_10(input_path: str, output_path: str):
    df = pd.read_excel(input_path, sheet_name='Sheet1', engine='openpyxl')

    with open("../prompts/调用qwen_32b回答.txt", 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    for index, row in df.iterrows():
        if pd.notna(df.at[index, "answer_jiuwen_10轮"]) and df.at[index, "answer_jiuwen_10轮"] != "":
            print(f"第{index}轮已有结果，跳过")
            continue

        user_prompt = f"""
上下文：{df.at[index, "context_content_jiuwen_10轮"]}
用户当前query：{df.at[index, "user"]}
        """

        response = 模型调用脚本_外部模型.invoke_model(system_prompt, user_prompt, "qwen3-32b")

        df.at[index, "answer_jiuwen_10轮"] = response
        df.at[index, "tokens_jiuwen_10轮"] = len(tiktoken.encoding_for_model("gpt-4").encode(str(df.at[index, "context_content_jiuwen_10轮"])))
        print(f"user: {df.at[index, 'user']}")
        print(f"answer_jiuwen_10轮: {response}")
        df.to_excel(output_path, sheet_name='Sheet1', index=False)
        print("**********************************")
        print(f"第{index}轮处理完成")
        print("**********************************")


def answer_by_qwen32b_jiuwen_20(input_path: str, output_path: str):
    df = pd.read_excel(input_path, sheet_name='Sheet1', engine='openpyxl')

    with open("../prompts/调用qwen_32b回答.txt", 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    for index, row in df.iterrows():
        if pd.notna(df.at[index, "answer_jiuwen_20轮"]) and df.at[index, "answer_jiuwen_20轮"] != "":
            print(f"第{index}轮已有结果，跳过")
            continue

        user_prompt = f"""
上下文：{df.at[index, "context_content_jiuwen_20轮"]}
用户当前query：{df.at[index, "user"]}
        """

        response = 模型调用脚本_外部模型.invoke_model(system_prompt, user_prompt, "qwen3-32b")

        df.at[index, "answer_jiuwen_20轮"] = response
        df.at[index, "tokens_jiuwen_20轮"] = len(tiktoken.encoding_for_model("gpt-4").encode(str(df.at[index, "context_content_jiuwen_20轮"])))
        print(f"user: {df.at[index, 'user']}")
        print(f"answer_jiuwen_20轮: {response}")
        df.to_excel(output_path, sheet_name='Sheet1', index=False)
        print("**********************************")
        print(f"第{index}轮处理完成")
        print("**********************************")


def build_5_15_context(df, current_index):
    # 取当前行在excel中位置之前的所有行（按行顺序，不区分对话）
    prev_indices = [idx for idx in df.index if idx < current_index]

    # 按5+15策略拼接上下文，distance=1表示最近一轮，distance=n表示最早一轮
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
    # 取当前行在excel中位置之前的所有行（按行顺序，不区分对话）
    prev_indices = [idx for idx in df.index if idx < current_index]

    # 先按5+15策略构建候选列表，distance=1表示最近一轮，distance=n表示最早一轮
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

    # 从最近一轮往前累加token，超过3000则截断，保留最近的轮次
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


def answer_by_qwen32b_5_and_15(input_path: str, output_path: str):
    df = pd.read_excel(input_path, sheet_name='Sheet1', engine='openpyxl')

    with open("../prompts/调用qwen_32b回答.txt", 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    for index, row in df.iterrows():
        if pd.notna(df.at[index, "answer_5+15"]) and df.at[index, "answer_5+15"] != "":
            print(f"第{index}轮已有结果，跳过")
            continue

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


def answer_by_qwen32b_5_and_15_3000(input_path: str, output_path: str):
    df = pd.read_excel(input_path, sheet_name='Sheet1', engine='openpyxl')

    with open("../prompts/调用qwen_32b回答.txt", 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    for index, row in df.iterrows():
        if pd.notna(df.at[index, "answer_5+15_3000"]) and df.at[index, "answer_5+15_3000"] != "":
            print(f"第{index}轮已有结果，跳过")
            continue

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
        df.to_excel(output_path, sheet_name='Sheet1', index=False)
        print("**********************************")
        print(f"第{index}轮处理完成")
        print("**********************************")


def build_20_compressed_context(df, current_index):
    # 取当前行在excel中位置之前的所有行（按行顺序，不区分对话），最多20轮
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


def answer_by_qwen32b_20_compressed(input_path: str, output_path: str):
    df = pd.read_excel(input_path, sheet_name='Sheet1', engine='openpyxl')
    if "answer_20_compressed" in df.columns:
        df["answer_20_compressed"] = df["answer_20_compressed"].astype(object)

    with open("../prompts/调用qwen_32b回答.txt", 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    for index, row in df.iterrows():
        if pd.notna(df.at[index, "answer_20_compressed"]) and df.at[index, "answer_20_compressed"] != "":
            print(f"第{index}轮已有结果，跳过")
            continue

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


def answer_no_context(input_path: str, output_path: str):
    df = pd.read_excel(input_path, sheet_name='Sheet1', engine='openpyxl')

    with open("../prompts/调用qwen_32b回答.txt", 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    for index, row in df.iterrows():
        if pd.notna(df.at[index, "answer_no_context"]) and df.at[index, "answer_no_context"] != "":
            print(f"第{index}轮已有结果，跳过")
            continue

        user_prompt = f"""
上下文：[]
用户当前query：{df.at[index, "user"]}
        """

        response = 模型调用脚本_外部模型.invoke_model(system_prompt, user_prompt, "qwen3-32b")

        df.at[index, "answer_no_context"] = response
        print(f"user: {df.at[index, 'user']}")
        print(f"answer_no_context: {response}")
        df.to_excel(output_path, sheet_name='Sheet1', index=False)
        print("**********************************")
        print(f"第{index}轮处理完成")
        print("**********************************")


if __name__ == "__main__":
    XLSX = "../测试用例/0311模型生成测试用例.xlsx"

    # 第一阶段：生成各策略回答
    # answer_by_qwen32b_jiuwen_10(XLSX, XLSX)
    # answer_by_qwen32b_jiuwen_20(XLSX, XLSX)
    # answer_by_qwen32b_5_and_15(XLSX, XLSX)
    # answer_no_context(XLSX, XLSX)
    # answer_by_qwen32b_5_and_15_3000(XLSX, XLSX)
    answer_by_qwen32b_20_compressed(XLSX, XLSX)

    # 第二阶段：清除旧打分列，重新按新评分标准打分
    print("**********************************")
    print("回答生成完毕，开始清除旧打分列并重新评分")
    print("**********************************")

    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from 单答案评分 import judge_single

    # 重新打分
    # judge_single(XLSX, XLSX, answer_col="answer_jiuwen_10轮",    score_col_prefix="score_jiuwen_10轮")
    # judge_single(XLSX, XLSX, answer_col="answer_jiuwen_20轮",    score_col_prefix="score_jiuwen_20轮")
    # judge_single(XLSX, XLSX, answer_col="answer_no_context",     score_col_prefix="score_no_context")
    # judge_single(XLSX, XLSX, answer_col="answer_5+15",           score_col_prefix="score_5+15")
    # judge_single(XLSX, XLSX, answer_col="answer_5+15_3000",      score_col_prefix="score_5+15_3000")
    judge_single(XLSX, XLSX, answer_col="answer_20_compressed",  score_col_prefix="score_20_compressed")

    print("**********************************")
    print("全部完成")
    print("**********************************")
import pandas as pd
import tiktoken

import 模型调用脚本_外部模型


def answer_by_qwen32b_jiuwen(input_path: str, output_path: str):
    df = pd.read_excel(input_path, sheet_name='Sheet1', engine='openpyxl')

    with open("../prompts/调用qwen_32b回答.txt", 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    for index, row in df.iterrows():
        user_prompt = f"""
上下文：{df.at[index, "context_content_jiuwen"]}
用户当前query：{df.at[index, "user"]}
        """

        response = 模型调用脚本_外部模型.invoke_model(system_prompt, user_prompt, "qwen3-32b")

        df.at[index, "answer_jiuwen"] = response
        content_length = len(tiktoken.encoding_for_model("gpt-4").encode(df.at[index, "context_content_jiuwen"]))
        df.at[index, "tokens_jiuwen"] = content_length
        print(f"user: {df.at[index, 'user']}")
        print(f"answer_jiuwen: {response}")
        df.to_excel(output_path, sheet_name='Sheet1', index=False)
        print("**********************************")
        print(f"第{index}轮处理完成")
        print("**********************************")

def build_5_15_context(df, current_index):
    current_context_id = str(df.at[current_index, "context_id"])
    # 解析对话id和轮次，例如 "3_5" -> conv_id="3", turn_index=5
    parts = current_context_id.rsplit("_", 1)
    conv_id = parts[0]
    turn_index = int(parts[1])

    # 找到同一对话中排在当前行之前的所有轮次
    prev_rows = []
    for idx in df.index:
        cid = str(df.at[idx, "context_id"])
        c_parts = cid.rsplit("_", 1)
        if c_parts[0] == conv_id and int(c_parts[1]) < turn_index:
            prev_rows.append((int(c_parts[1]), idx))
    prev_rows.sort(key=lambda x: x[0])

    # 按5+15策略拼接上下文，distance=1表示最近一轮，distance=n表示最早一轮
    context = []
    n = len(prev_rows)
    for i, (_, row_idx) in enumerate(prev_rows):
        distance = n - i
        if distance > 20:
            continue
        user_content = str(df.at[row_idx, "user"])
        if distance <= 5:
            assistant_content = str(df.at[row_idx, "assistant"])
        else:
            assistant_content = str(df.at[row_idx, "assistant_compressed"])
        context.append({"user": user_content, "assistant": assistant_content})

    return context


def answer_by_qwen32b_5_and_15(input_path: str, output_path: str):
    df = pd.read_excel(input_path, sheet_name='Sheet1', engine='openpyxl')

    with open("../prompts/调用qwen_32b回答.txt", 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    for index, row in df.iterrows():
        context = build_5_15_context(df, index)

        user_prompt = f"""
上下文：{str(context)}
用户当前query：{df.at[index, "user"]}
        """

        response = 模型调用脚本_外部模型.invoke_model(system_prompt, user_prompt, "qwen3-32b")

        df.at[index, "answer_5+15"] = response

        print(f"user: {df.at[index, 'user']}")
        print(f"context轮数: {len(context)}")
        print(f"answer_5+15: {response}")
        df.to_excel(output_path, sheet_name='Sheet1', index=False)
        print("**********************************")
        print(f"第{index}轮处理完成")
        print("**********************************")


def answer_no_context(input_path: str, output_path: str):
    df = pd.read_excel(input_path, sheet_name='Sheet1', engine='openpyxl')

    with open("../prompts/调用qwen_32b回答.txt", 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    for index, row in df.iterrows():
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
    # answer_by_qwen32b_jiuwen("../测试用例/0311模型生成测试用例_jiuwen_select.xlsx","../测试用例/0311模型生成测试用例_jiuwen_select.xlsx")

    # answer_by_qwen32b_5_and_15("../测试用例/0311模型生成测试用例_jiuwen_select.xlsx","../测试用例/0311模型生成测试用例_jiuwen_select.xlsx")

    answer_no_context("../测试用例/0311模型生成测试用例_jiuwen_select.xlsx","../测试用例/0311模型生成测试用例_jiuwen_select.xlsx")
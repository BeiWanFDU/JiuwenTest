import json

import pandas as pd

import 模型调用脚本_外部模型


def judge_single(input_path: str, output_path: str, answer_col: str, score_col_prefix: str):
    df = pd.read_excel(input_path, sheet_name='Sheet1', engine='openpyxl')

    with open("../prompts/单答案打分.txt", 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    for index, row in df.iterrows():
        col = f"{score_col_prefix}_回答准确性"
        if col in df.columns and pd.notna(df.at[index, col]) and df.at[index, col] != "":
            print(f"第{index}轮已有评分，跳过")
            continue

        user_prompt = f"""
答题要求：{df.at[index, "guide"]}
用户问题：{df.at[index, "user"]}
参考答案：{df.at[index, "assistant"]}
被测答案：{df.at[index, answer_col]}
        """

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
        df.at[index, f"{score_col_prefix}_回答完整性"] = response["回答完整性"]
        df.at[index, f"{score_col_prefix}_回答准确性"] = response["回答准确性"]
        df.at[index, f"{score_col_prefix}_reason"] = response["reason"]

        print(f"user: {df.at[index, 'user']}")
        print(f"{score_col_prefix}: {response}")
        df.to_excel(output_path, sheet_name='Sheet1', index=False)
        print("**********************************")
        print(f"第{index}轮处理完成")
        print("**********************************")


if __name__ == "__main__":
    # judge_single("../测试用例/0311模型生成测试用例.xlsx",
    #              "../测试用例/0311模型生成测试用例.xlsx",
    #              answer_col="answer_jiuwen_10轮",
    #              score_col_prefix="score_jiuwen_10轮")

    # judge_single("../测试用例/0311模型生成测试用例.xlsx",
    #              "../测试用例/0311模型生成测试用例.xlsx",
    #              answer_col="answer_jiuwen_20轮",
    #              score_col_prefix="score_jiuwen_20轮")

    # judge_single("../测试用例/0311模型生成测试用例.xlsx",
    #              "../测试用例/0311模型生成测试用例.xlsx",
    #              answer_col="answer_5+15",
    #              score_col_prefix="score_5+15")

    # judge_single("../测试用例/0311模型生成测试用例.xlsx",
    #              "../测试用例/0311模型生成测试用例.xlsx",
    #              answer_col="answer_5+15_3000",
    #              score_col_prefix="score_5+15_3000")

    # judge_single("../测试用例/0311模型生成测试用例.xlsx",
    #              "../测试用例/0311模型生成测试用例.xlsx",
    #              answer_col="answer_no_context",
    #              score_col_prefix="score_no_context")
    pass
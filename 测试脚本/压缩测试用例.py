import json

import pandas as pd

import 模型调用脚本_外部模型


def compress_context(input_path: str, output_path: str):
    df = pd.read_excel(input_path, sheet_name='Sheet1', engine='openpyxl')

    with open("../prompts/压缩.txt", 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    for index, row in df.iterrows():
        user_prompt = json.dumps([
            {"role": "user", "content": df.at[index, "user"]},
            {"role": "assistant", "content": df.at[index, "answer"]}
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


if __name__ == "__main__":
    compress_context("../测试用例/0311模型生成测试用例.xlsx",
                     "../测试用例/0311模型生成测试用例.xlsx")
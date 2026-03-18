import pandas as pd

import qwen32b回答测试


def count_tokens_5_15(input_path: str, output_path: str):
    df = pd.read_excel(input_path, sheet_name='Sheet1', engine='openpyxl')

    for index, row in df.iterrows():
        context, token_count = qwen32b回答测试.build_5_15_context(df, index)

        df.at[index, "tokens_5+15"] = token_count

        print(f"index {index} | context轮数: {len(context)} | tokens: {token_count}")
        df.to_excel(output_path, sheet_name='Sheet1', index=False)
        print("**********************************")
        print(f"第{index}轮处理完成")
        print("**********************************")


if __name__ == "__main__":
    count_tokens_5_15("../测试用例/0311模型生成测试用例_jiuwen_select.xlsx",
                      "../测试用例/0311模型生成测试用例_jiuwen_select.xlsx")
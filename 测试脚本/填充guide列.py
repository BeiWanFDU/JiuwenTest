import json

import pandas as pd


def fill_guide(json_path: str, input_path: str, output_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 构建 context_id -> guide 的映射，context_id格式与xlsx一致："{session_id}_{turn_id}"
    guide_map = {}
    for session in data:
        for item in session["session"]:
            context_id = f"{item['session_id']}_{item['turn_id']}"
            guide_map[context_id] = item.get("guide", "")

    df = pd.read_excel(input_path, sheet_name='Sheet1', engine='openpyxl')
    df["guide"] = df["context_id"].astype(str).map(guide_map)

    df.to_excel(output_path, sheet_name='Sheet1', index=False)
    print(f"完成，共填充 {df['guide'].notna().sum()} 行")


if __name__ == "__main__":
    fill_guide("../测试用例/0311模型生成测试用例.json",
               "../测试用例/0311模型生成测试用例.xlsx",
               "../测试用例/0311模型生成测试用例.xlsx")

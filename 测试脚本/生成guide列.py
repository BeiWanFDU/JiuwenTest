"""
为 0307AISF精选测试用例.xlsx 生成 guide 列。
按 session 重建完整对话历史，调用模型生成每一轮的 guide。
支持断点续跑（已有 guide 的行跳过）。
"""

import json

import pandas as pd

import 模型调用脚本_外部模型

XLSX = "../测试用例/0307AISF精选测试用例.xlsx"


def build_history(df, session_id: str, current_turn: int) -> list:
    """重建当前轮次之前的完整对话历史（同 session 内按轮次顺序）"""
    history = []
    for _, row in df.iterrows():
        cid = str(row["context_id"])
        parts = cid.rsplit("_", 1)
        if len(parts) != 2:
            continue
        sid, turn = parts[0], int(parts[1])
        if sid == session_id and turn < current_turn:
            history.append({
                "turn": turn,
                "user": str(row["user"]),
                "assistant": str(row["answer"])
            })
    history.sort(key=lambda x: x["turn"])
    return history


def generate_guide(input_path: str, output_path: str):
    df = pd.read_excel(input_path, sheet_name='Sheet1', engine='openpyxl')

    if "guide" not in df.columns:
        df["guide"] = ""
    df["guide"] = df["guide"].astype(object)

    with open("../prompts/生成guide.txt", 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    for index, row in df.iterrows():
        # 断点续跑：已有 guide 则跳过
        if pd.notna(df.at[index, "guide"]) and str(df.at[index, "guide"]).strip() != "":
            print(f"第{index}行已有guide，跳过")
            continue

        cid = str(row["context_id"])
        parts = cid.rsplit("_", 1)
        if len(parts) != 2:
            print(f"第{index}行 context_id 格式异常，跳过：{cid}")
            continue
        session_id, current_turn = parts[0], int(parts[1])

        history = build_history(df, session_id, current_turn)

        user_prompt = f"""历史对话：{json.dumps(history, ensure_ascii=False, indent=2)}
当前提问：{row['user']}"""

        response = 模型调用脚本_外部模型.invoke_model(system_prompt, user_prompt, "qwen-max")

        df.at[index, "guide"] = response.strip()

        print(f"context_id: {cid}")
        print(f"user: {row['user']}")
        print(f"guide: {response.strip()}")
        df.to_excel(output_path, sheet_name='Sheet1', index=False)
        print("**********************************")
        print(f"第{index}行处理完成")
        print("**********************************")


if __name__ == "__main__":
    generate_guide(XLSX, XLSX)
    print("guide 列生成完成")
import random

import pandas as pd
import requests

def process_context(input_path:str,output_path:str):
    df = pd.read_excel(input_path, sheet_name='Sheet1',engine='openpyxl')
    for index, row in df.iterrows():
        # 请求本轮query所需要的上下文id，context_id_selected_by_jiuwen
        context_selected_by_jiuwen = query_context(df.at[index,"user"])
        context_id_selected_by_jiuwen = [item["context_id"] for item in context_selected_by_jiuwen ["context"]["qa_list"]]

        # 将本轮query所需要的上下文拼接成json数组并赋值给变量context_selected_by_jiuwen，其来源为context_selected_by_jiuwen的qa_list字段，形式如下
        context_content_jiuwen = [
            {
                "user": item["query"]["content"],
                "assistant": item["answer"]["content"]
            }
            for item in context_selected_by_jiuwen["context"]["qa_list"]
        ]


        # 将本轮对话添加到上下文服务，并获取本轮对话所对应的上下文id，context_id_jiuwen
        context_id_jiuwen = add_context(df.at[index,"user"],df.at[index,"answer"])["context_id"][0]



        print(f"九问的上下文id：{context_id_jiuwen}")
        print(f"九问的选择上下文id：{context_id_selected_by_jiuwen}")
        print(f"九问输出的上下文内容{context_content_jiuwen}")
        print("user_query:")
        print(df.at[index,"user"])
        print("assistant:")
        print(df.at[index,"answer"])

        # 将context_id_jiuwen，context_id_selected_by_jiuwen添加到excel表中
        df.at[index,"context_id_jiuwen"] = context_id_jiuwen
        df.at[index,"context_id_selected_by_jiuwen"] = ",".join(context_id_selected_by_jiuwen)
        df.at[index,"context_content_jiuwen"] = str(context_content_jiuwen)

        df.to_excel(output_path, sheet_name='Sheet1', index=False)
        print("**********************************")
        print(f"第{index}轮处理完成")
        print("**********************************")


def add_context(
    query: str,
    answer: str,
    user_id: str = "zyk_005",
    sub_user_id: str = "sub_user_678",
    session_id: str = "session_abc123",
    agent_name: str = "math_tutor_bot",
    label: str = "math_qa",
    aging_policy: dict = None,
    base_url: str = "http://9.15.87.116:19000"
) -> dict:
    # 用于模拟测试
    # return {
    #     "context_id": [str(random.randint(1, 10000))]
    # }

    if aging_policy is None:
        aging_policy = {"max_turns": 20, "expire_time": 3600}

    url = f"{base_url}/api/v1/context/add"
    payload = {
        "user_id": user_id,
        "sub_user_id": sub_user_id,
        "session_id": session_id,
        "agent_name": agent_name,
        "label": label,
        "context": [
            {"query": query, "answer": answer}
        ],
        "aging_policy": aging_policy
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


def query_context(
    query: str,
    session_id: str = "session_abc123",
    user_id: str = "zyk_005",
    current_agent: str = "math_tutor_bot",
    query_type: str = "all",
    limit_qa_rounds: int = 20,
    max_qa_tokens: int = 1000,
    base_url: str = "http://9.15.87.116:19000"
) -> dict:
    # 用于模拟测试
    # return {
    #     "context":
    #         {
    #             "qa_list": [
    #                 {
    #                     "context_id": "1",
    #                     "label": "xx",
    #                     "owner_agent": "",
    #                     "query": {
    #                         "role": "user",
    #                         "content": "xx"
    #                     },
    #                     "answer": {
    #                         "role": "assistant",
    #                         "content": "xx"
    #                     },
    #                     "re_write_query": "xx",
    #                     "custom_summary": "",
    #                     "date": "xxx"
    #                 },
    #                 {
    #                     "context_id": "2",
    #                     "label": "xx",
    #                     "owner_agent": "",
    #                     "query": {
    #                         "role": "user",
    #                         "content": "xx"
    #                     },
    #                     "answer": {
    #                         "role": "assistant",
    #                         "content": "xx"
    #                     },
    #                     "re_write_query": "xx",
    #                     "custom_summary": "",
    #                     "date": "xxx"
    #                 }
    #             ],
    #             "abstract_qa": [
    #                 {
    #                     "context_id": "",
    #                     "query": {
    #                         "role": "user",
    #                         "content": "xx"
    #                     },
    #                     "answer": {
    #                         "role": "assistant",
    #                         "content": "xx"
    #                     }
    #                 }
    #             ]
    #         },
    #     "context_msg": [
    #         {
    #             "context_id": "",
    #             "query": {
    #                 "role": "user",
    #                 "content": "xx"
    #             },
    #             "answer": {
    #                 "role": "assistant",
    #                 "content": "xx"
    #             }
    #         }
    #     ],
    #     "related_qa": [""]
    # }

    url = f"{base_url}/api/v1/context/query"
    payload = {
        "session_id": session_id,
        "user_id": user_id,
        "current_agent": current_agent,
        "query": query,
        "query_type": query_type,
        "limit_qa_rounds": limit_qa_rounds,
        "max_qa_tokens": max_qa_tokens
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    # process_context("0311模型生成测试用例_jiuwen.xlsx","0311模型生成测试用例_jiuwen_select.xlsx")
    df = pd.read_excel("0311模型生成测试用例_jiuwen_select.xlsx")
    df.to_csv("0311模型生成测试用例_jiuwen_select.csv",index=False)

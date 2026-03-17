import json
import random

import pandas as pd
import requests
import tiktoken

import 模型调用脚本_外部模型


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


def judge_by_llm(input_path:str,output_path:str):
    df = pd.read_excel(input_path, sheet_name='Sheet1',engine='openpyxl')
    for index, row in df.iterrows():
        system_prompt = """
# 任务
你需要分析用户当前query，判断其中是否存在需要依赖上下文才能完成的信息缺口，并评估给定的上下文是否补全了这些信息缺口。

# 核心概念区分
## 信息缺口（本任务关注的对象）
指query中隐含的、在当前语句中未明确提及、需要从【对话上下文历史】中获取才能理解query本身的要素。
常见形式：
- 代词指代："他"、"这个"、"那里"、"刚才那部"等
- 省略信息："也一样"、"继续"、"再说一遍"、"那个方案"等
- 隐含引用："按照之前的方案"、"跟刚才说的一样"、"上面提到的那个"
- 上下文依赖："然后呢？"、"为什么？"、"具体怎么操作？"

## 信息需求（不属于信息缺口）
指用户希望模型从外部知识或实时数据中查询/获取的新信息，这类信息在query中已经完整描述了查询意图，不依赖上下文即可理解。
常见形式：
- 查询天气："查成都武侯区未来三天的天气"
- 查询实时数据："宁德时代今天的股票涨跌幅"
- 推荐/搜索："推荐武侯祠附近的三甲医院"
- 预订类："查从北京飞上海的机票"

【判断关键】：问自己"这句话我能看懂用户想要什么吗？" 如果能看懂（即使答案未知），就不是信息缺口。

# 评分标准（1-5分）
## 5分 - 完全满足
情况A（无信息缺口）：query本身完整自洽，不依赖任何上下文即可理解和回答
情况B（有信息缺口且已补全）：query存在信息缺口，但上下文提供了所有必要信息

## 4分 - 基本满足
query存在信息缺口，上下文提供了大部分必要信息，但可能缺少少量非关键细节

## 3分 - 部分满足
query存在信息缺口，上下文只提供了部分信息，仍有重要信息缺失

## 2分 - 较少满足
query存在信息缺口，上下文仅提供了极少量相关信息，无法支撑完整回答

## 1分 - 完全不满足
query存在关键信息缺口，但上下文完全缺失相关必要信息

# 分析步骤
第一步：识别信息缺口 - 分析query中是否有需要从对话历史中才能理解的要素（注意区分"信息缺口"与"信息需求"）
第二步：检查上下文 - 如果存在信息缺口，查看上下文是否包含填补缺口所需的信息
第三步：综合评估 - 基于缺口填补情况给出评分

# 输入格式
上下文：{上下文内容}
用户当前query：{用户问题}

# 输出格式
请严格按照以下JSON格式返回结果，不要包含```json代码块标记或其他额外内容：
{
    "score": "评分（1-5之间的整数）",
    "reason": "判断理由（需说明：①query是否存在信息缺口；②如果存在，上下文是否补全了这些缺口；③评分依据）"
}

# 注意事项
- 关键原则：query不依赖上下文时，即使上下文为空、或query中包含大量待查询信息，也应评5分
- 信息需求≠信息缺口：用户要求查询天气、股票、机票、推荐地点等，是对外部信息的需求，query本身已完整表达意图，不构成信息缺口
- 仅基于提供的上下文进行判断，不考虑外部知识
- 理由要具体，指出存在哪些信息缺口，以及上下文是否提供了对应信息

# 示例
## 示例1：无信息缺口（query完整自洽）
输入：
上下文：[]
用户query：我下周五要去杭州出差，帮我查一下下周五杭州的天气，查一下本周四晚广州到杭州的高铁票，再推荐一下杭州西湖景区附近的商务酒店，另外查一下西湖景区近期的商务会展活动
输出：
{
    "score": "5",
    "reason": "query不存在信息缺口：所有时间、地点、需求都在query中明确给出，查天气/查高铁/推荐酒店/查活动均属于信息需求而非信息缺口，不依赖任何上下文即可完整理解query意图，因此评5分"
}

## 示例2：无信息缺口（含大量信息需求但query完整）
输入：
上下文：[旅游相关的其他城市对话历史]
用户query：查询成都武侯区未来三天的天气状况、成都武侯区距离武侯祠最近的三甲医院地址及联系电话、宁德时代今天的股票涨跌幅，还有从成都飞往北京的经济舱机票价格
输出：
{
    "score": "5",
    "reason": "query不存在信息缺口：查询成都天气、查三甲医院、查股票涨跌幅、查机票价格均为信息需求，query中已完整描述了所有查询意图（城市、时间范围、具体需求均明确），不需要借助上下文来理解query本身，因此评5分"
}

## 示例3：有信息缺口且上下文补全
输入：
上下文：[{"role": "user", "content": "我想去杭州出差"}, {"role": "assistant", "content": "请问您什么时候去？"}]
用户query：下周五
输出：
{
    "score": "5",
    "reason": "query存在信息缺口（仅说'下周五'，省略了去杭州出差这一核心意图），但上下文中明确提到了'去杭州出差'，信息缺口被完整补全，因此评5分"
}

## 示例4：有信息缺口且上下文补全（含指代）
输入：
上下文：[{"role": "user", "content": "推荐一部最近的好电影"}, {"role": "assistant", "content": "推荐《某某》，豆瓣9.2分，3月8日上映"}]
用户query：刚才说的那部电影，能告诉我上映时间吗？
输出：
{
    "score": "5",
    "reason": "query存在信息缺口（'刚才说的那部电影'需要从上下文获取片名），上下文中助手明确推荐了《某某》，信息缺口被完整补全，因此评5分"
}

## 示例5：有信息缺口但上下文未补全
输入：
上下文：[{"role": "user", "content": "我想去旅游"}]
用户query：那里天气怎么样？
输出：
{
    "score": "1",
    "reason": "query存在关键信息缺口（'那里'指代不明），但上下文中只有'想去旅游'，未明确具体目的地，无法补全指代信息，因此评1分"
}
        """
        user_prompt = f"""
        上下文：{df.at[index,"context_content_jiuwen"]}
        用户当前query：{df.at[index,"user"]}
        """
        response = 模型调用脚本_外部模型.invoke_model(system_prompt,user_prompt,"qwen-max")
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            print(f"JSON解析失败，跳过index {index}，原始response：{response}")
            continue
        except Exception as e:
            print(f"解析失败，跳过index {index}，错误：{e}，原始response：{response}")
            continue
        content_length = len(tiktoken.encoding_for_model("gpt-4").encode(df.at[index,"context_content_jiuwen"]))
        df.at[index,"tokens_jiuwen"] = content_length
        df.at[index,"score_jiuwen"] = response["score"]
        df.at[index,"reason_jiuwen"] = response["reason"]
        print(f"content_length；{content_length}")
        print(response)
        df.to_excel(output_path, sheet_name='Sheet1', index=False, )


def add_context(
    query: str,
    answer: str,
    user_id: str = "zyk_002",
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
        aging_policy = {"max_turns": 10, "expire_time": 3600}

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
    session_id: str = "zyk_002",
    user_id: str = "user_12345",
    current_agent: str = "math_tutor_bot",
    query_type: str = "all",
    limit_qa_rounds: int = 10,
    max_qa_tokens: int = 1000,
    base_url: str = "http://9.15.87.116:19000"
) -> dict:

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
    # process_context("AISF_0307_19_jiuwen.xlsx","AISF_0307_19_jiuwen_select.xlsx")
    judge_by_llm("AISF_0307_19_jiuwen_select.xlsx","AISF_0307_19_jiuwen_select_judge_by_llm.xlsx")
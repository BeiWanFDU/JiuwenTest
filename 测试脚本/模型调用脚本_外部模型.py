# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import base64
import os
from openai import OpenAI

# --- 配置信息 ---
# 注意：虽然使用的是OpenAI的Client，但BASE_URL和API_KEY是阿里云千问的配置。

# 阿里云千问兼容模式的Base URL
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 您的API Key
API_KEY = "sk-127ffd0777364de68ad5c5c1a68f2141"

# 使用的模型：qwen3-vl-flash 是一个多模态模型
# MODEL_NAME = "qwen3-vl-flash"
# MODEL_NAME = "qwen-plus"
MODEL_NAME = "qwen3-32b"
# MODEL_NAME = "qwen3-8b"



def create_ali_openai_client(api_key: str, base_url: str) -> OpenAI:
    """
    创建OpenAI Client实例，用于访问阿里云千问的兼容模式API。
    """
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        print("OpenAI Client 创建成功，目标URL:", base_url)
        return client
    except Exception as e:
        print(f"创建Client时发生错误: {e}")
        raise

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def send_text_content(client: OpenAI, system_prompt: str, user_prompt: str,model_name) -> str:
    """
    功能一：仅发送文字内容给LLM。
    """
    # print(f"\n--- 正在发送纯文本请求 ---")
    # print(f"Prompt: {prompt}")

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,  # 纯文本内容
                },
                {
                    "role": "user",
                    "content": user_prompt,  # 纯文本内容
                }
            ],
            extra_body={"enable_thinking": False},
            temperature = 0
            # max_tokens=100  # 可以根据需要设置最大回复长度
        )

        # 打印LLM的回复
        # print("\n--- LLM 文本回复 ---")
        # if response.choices:
        #     print(response.choices[0].message.content)
        #     print(response)
        # else:
        #     print("未收到有效的回复。")
        print(response)
        return response.choices[0].message.content

    except Exception as e:
        print(f"发送纯文本请求失败: {e}")
        return None

def send_image_url_and_text(client: OpenAI, image_url: str, user_prompt: str, system_prompt:str):
    """
    功能二：发送URL图片和文字内容给多模态LLM。
    """
    print(f"\n--- 正在发送多模态请求 (图片URL + 文本) ---")
    # print(f"Image URL: {image_url}")
    print(f"Prompt: {user_prompt}")

    # 构造多模态请求的内容列表
    messages_content = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt
                },
                {
                    "type": "image_url",
                    # "image_url": {"url": image_url}
                    "image_url": {"url": f"data:image/jpeg;base64,{image_url}"}
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages_content
            # max_tokens=500
        )

        # 打印LLM的回复
        print("\n--- LLM 多模态回复 ---")
        if response.choices:
            print(response.choices[0].message.content)
        else:
            print("未收到有效的回复。")

        return response

    except Exception as e:
        print(f"发送多模态请求失败: {e}")
        # 如果是常见的图片URL问题，可以提示用户检查URL是否可公开访问
        if "invalid url" in str(e).lower():
            print("TIP: 请确保提供的图片URL是公开可访问的。")
        return None

def invoke_model( system_prompt: str, user_prompt: str,model_name:str = "qwen-max")-> str:
    print("")
    print("")
    print("")
    print("")
    print("")
    print(user_prompt)
    ali_client = create_ali_openai_client(API_KEY, BASE_URL)
    return send_text_content(ali_client, system_prompt, user_prompt,model_name)

# --- 主执行区 ---
if __name__ == "__main__":
    # 1. 初始化 Client
    ali_client = create_ali_openai_client(API_KEY, BASE_URL)

    # 2. 演示功能一: 发送纯文本
    # text_prompt = "请简要介绍一下大型语言模型（LLM）的关键优势。"
    # send_text_content(ali_client, text_prompt)

    # 3. 演示功能二: 发送图片URL和文本
    # 注意：请替换成一个真实且公开可访问的图片URL
    # example_image_url = "https://pic1.zhimg.com/v2-5d719cec37bd545bce8ca3a6ac64c05c_1440w.jpg"
    # user_prompt = "我会给你一张照片。用20句话内讲明白里面的关键信息。"
    #
    # user_prompt = "我会给你一张关于我的狗的照片，其名字叫毛毛。"
    # example_image_url = encode_image("images/maomao.jpeg")

    # user_prompt = "这是永康的电话，你记录一下"
    # example_image_url = encode_image("images/手机号码.jpeg")

    user_prompt = "我叫加加，我在华为上海练秋湖工作"
    # example_image_url = encode_image("images/穿搭.jpeg")

    with open("../../记忆萃取流程/提取成独立信息点/提取成独立信息点_画像记忆.txt", 'r', encoding='utf-8') as file:
        system_prompt = file.read()
    # send_image_url_and_text(ali_client, example_image_url, user_prompt, system_prompt)
    send_text_content(ali_client, system_prompt, user_prompt)

    # Case1：
    # 上传一张狗的照片（颜色、品种）
    # 用户：这是我的宠物毛毛
    # 用户：毛毛是什么颜色？
    #
    # Case2：
    # 上传一张手写的照片（王总的电话：18700002222）
    # 用户：帮我查一下王总的电话
    #
    # Case3：
    # 上传一张全身照片，问穿搭如何
    # 用户：我上周的穿搭如何

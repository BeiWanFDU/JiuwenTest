import pandas as pd
df = pd.read_csv("0311模型生成测试用例_jiuwen_真20轮.csv")
df.to_excel("00311模型生成测试用例_jiuwen_真20轮.xlsx",index=False)
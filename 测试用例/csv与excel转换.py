import pandas as pd
df = pd.read_csv("0307AISF精选测试用例.csv")
df.to_excel("0307AISF精选测试用例.xlsx",index=False)
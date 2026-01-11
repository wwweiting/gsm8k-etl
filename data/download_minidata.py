import pandas as pd

# 1. 讀取完整檔案
df = pd.read_parquet("gsm8k_dataset/main/train-00000-of-00001.parquet")

# 2. 只取前 100 筆 (0 到 99)
mini_df = df.iloc[:100]

# 3. 檢查一下
print(f"原本資料筆數: {len(df)}")
print(f"測試資料筆數: {len(mini_df)}")
print(mini_df.head())

# 把這 100 筆存成一個新的 CSV 或 JSON，方便你用 Excel 或文字編輯器查看
mini_df.to_csv("mini_train.csv", index=False)
mini_df.to_json("mini_train.json", orient="records", indent=4, force_ascii=False)

print("已儲存 mini_train.csv 和 mini_train.json！")
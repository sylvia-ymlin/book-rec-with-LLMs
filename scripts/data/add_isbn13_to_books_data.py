import pandas as pd

# 读取主表和 books_data_with_isbn.csv
main = pd.read_csv("data/books_with_emotions.csv", usecols=["title", "isbn13"])
data = pd.read_csv("data/books_data_with_isbn.csv")

# 标准化标题
main["title"] = main["title"].astype(str).str.strip().str.lower()
data["Title"] = data["Title"].astype(str).str.strip().str.lower()

# 合并，左连接
merged = data.merge(main, left_on="Title", right_on="title", how="left")

# 保存新文件
merged.to_csv("data/books_data_with_isbn13.csv", index=False)
print("已生成 data/books_data_with_isbn13.csv，包含 isbn13 字段。")

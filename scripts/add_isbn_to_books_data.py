import pandas as pd

# 读取 books_data.csv
books_data = pd.read_csv("data/books_data.csv")

# 读取 Books_rating.csv，只取 Title 和 Id 字段
ratings = pd.read_csv("data/Books_rating.csv", usecols=["Title", "Id"])

# 去重，避免多对一
ratings = ratings.drop_duplicates(subset=["Title"])

# 合并，左连接，保留 books_data.csv 所有行
merged = books_data.merge(ratings, on="Title", how="left")

# 重命名 Id 为 isbn
merged = merged.rename(columns={"Id": "isbn"})

# 保存新文件
merged.to_csv("data/books_data_with_isbn.csv", index=False)

print("已生成 data/books_data_with_isbn.csv，包含 isbn 字段。")

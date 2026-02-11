import pandas as pd
import csv

# 读取原始数据，遇到格式错误行自动跳过，保证流程不中断
books_data = pd.read_csv(
    "data/books_data.csv",
    engine="python",
    quotechar='"',
    escapechar='\\',
    on_bad_lines='skip'  # pandas >=1.3
)
ratings = pd.read_csv("data/Books_rating.csv", engine="python", quotechar='"', escapechar='\\', on_bad_lines='skip')

# 只保留有用字段
books_cols = [
    "Title", "description", "authors", "image", "publisher", "publishedDate", "categories"
]
books_data = books_data[books_cols]

# 只保留 Title, Id, review/score 字段用于合并
ratings_cols = ["Title", "Id", "review/score"]
ratings = ratings[ratings_cols]

# 去重
ratings = ratings.drop_duplicates(subset=["Title"])

# 合并，左连接，保留 books_data 所有行
merged = books_data.merge(ratings, on="Title", how="left")

# 重命名字段
merged = merged.rename(columns={
    "Id": "isbn10",
    "Title": "title",
    "authors": "authors",
    "description": "description",
    "image": "image",
    "publisher": "publisher",
    "publishedDate": "publishedDate",
    "categories": "categories",
    "review/score": "average_rating"
})

# 生成 isbn13（如有更复杂规则可补充，这里仅占位）
merged["isbn13"] = None  # 可后续补充isbn13生成逻辑

# 保存新表，强制所有字段加引号，防止description等字段被截断
merged.to_csv("data/books_basic_info.csv", index=False, quoting=csv.QUOTE_ALL, quotechar='"', escapechar='\\')
print("已生成 data/books_basic_info.csv，包含基础书籍信息字段。")

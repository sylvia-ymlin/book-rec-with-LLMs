#!/usr/bin/env python3
"""
数据划分脚本 - 为推荐系统准备训练/验证/测试集

划分策略: 时序划分 (Leave-Last-Out)
- 每个用户的最后一次评分 → test
- 每个用户的倒数第二次评分 → val  
- 其余评分 → train

只保留评分 >= 3 次的用户 (有足够历史)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

print('='*60)
print('推荐系统数据划分')
print('='*60)

start_time = time.time()

# 路径配置
DATA_PATH = Path('data/raw/Books_rating.csv')
OUTPUT_DIR = Path('data/rec')
OUTPUT_DIR.mkdir(exist_ok=True)

# ==================== 1. 加载数据 ====================
print('\n[1/5] 加载原始评论数据...')
# 原始列: Id (ISBN), User_id (用户), review/score, review/time, review/text
df = pd.read_csv(DATA_PATH, usecols=['Id', 'User_id', 'review/score', 'review/time', 'review/text'])
df.columns = ['isbn', 'user_id', 'rating', 'timestamp', 'review']

print(f'  原始记录数: {len(df):,}')
print(f'  用户数: {df["user_id"].nunique():,}')
print(f'  书籍数: {df["isbn"].nunique():,}')

# ==================== 2. 数据清洗 ====================
print('\n[2/5] 数据清洗...')

# 去除重复评分 (同一用户对同一本书)
df = df.drop_duplicates(subset=['user_id', 'isbn'], keep='last')
print(f'  去重后: {len(df):,}')

# 去除缺失值
df = df.dropna(subset=['rating', 'timestamp'])
print(f'  去除缺失后: {len(df):,}')

# 过滤低质量评分 (可选: 只保留 rating > 0)
df = df[df['rating'] > 0]
print(f'  过滤低质量后: {len(df):,}')

# ==================== 3. 用户筛选 ====================
print('\n[3/5] 筛选活跃用户...')

# 统计每个用户的评分数
user_counts = df.groupby('user_id').size()
print(f'  评分分布:')
print(f'    1次: {(user_counts == 1).sum():,}')
print(f'    2次: {(user_counts == 2).sum():,}')
print(f'    3-5次: {((user_counts >= 3) & (user_counts <= 5)).sum():,}')
print(f'    5-10次: {((user_counts > 5) & (user_counts <= 10)).sum():,}')
print(f'    10+次: {(user_counts > 10).sum():,}')

# 只保留评分 >= 3 次的用户 (需要 1 train + 1 val + 1 test)
active_users = user_counts[user_counts >= 3].index
df = df[df['user_id'].isin(active_users)]
print(f'  活跃用户 (>=3次): {len(active_users):,}')
print(f'  筛选后记录数: {len(df):,}')

# ==================== 4. 时序划分 ====================
print('\n[4/5] 时序划分 (Leave-Last-Out)...')

# 按用户和时间排序
df = df.sort_values(['user_id', 'timestamp'])

train_list = []
val_list = []
test_list = []

for user_id, group in tqdm(df.groupby('user_id'), desc='  划分用户'):
    # 按时间排序
    group = group.sort_values('timestamp')
    n = len(group)
    
    # 最后一条 → test
    test_list.append(group.iloc[-1])
    
    # 倒数第二条 → val
    val_list.append(group.iloc[-2])
    
    # 其余 → train
    train_list.extend(group.iloc[:-2].to_dict('records'))

# 转换为 DataFrame
train_df = pd.DataFrame(train_list)
val_df = pd.DataFrame(val_list)
test_df = pd.DataFrame(test_list)

print(f'  训练集: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)')
print(f'  验证集: {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)')
print(f'  测试集: {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)')

# ==================== 5. 保存数据 ====================
print('\n[5/5] 保存数据...')

train_df.to_csv(OUTPUT_DIR / 'train.csv', index=False)
val_df.to_csv(OUTPUT_DIR / 'val.csv', index=False)
test_df.to_csv(OUTPUT_DIR / 'test.csv', index=False)

# 保存用户列表 (用于后续评估)
active_users_df = pd.DataFrame({'user_id': active_users})
active_users_df.to_csv(OUTPUT_DIR / 'active_users.csv', index=False)

# 保存统计信息
stats = {
    'total_records': len(df),
    'train_records': len(train_df),
    'val_records': len(val_df),
    'test_records': len(test_df),
    'active_users': len(active_users),
    'books': df['isbn'].nunique(),
}

with open(OUTPUT_DIR / 'stats.txt', 'w') as f:
    for k, v in stats.items():
        f.write(f'{k}: {v:,}\n')

elapsed = time.time() - start_time

print('\n' + '='*60)
print('✅ 数据划分完成!')
print('='*60)
print(f'输出目录: {OUTPUT_DIR}')
print(f'  - train.csv: {len(train_df):,} 条')
print(f'  - val.csv: {len(val_df):,} 条')
print(f'  - test.csv: {len(test_df):,} 条')
print(f'  - active_users.csv: {len(active_users):,} 用户')
print(f'执行时间: {elapsed:.1f}秒')
print('='*60)

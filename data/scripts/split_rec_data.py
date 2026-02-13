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
import logging

logger = logging.getLogger(__name__)

DATA_PATH = Path("data/raw/Books_rating.csv")
OUTPUT_DIR = Path("data/rec")


def run(
    data_path: Path = DATA_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Split train/val/test with Leave-Last-Out. Callable from Pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    logger.info("Loading raw ratings...")
    df = pd.read_csv(data_path, usecols=['Id', 'User_id', 'review/score', 'review/time', 'review/text'])
    df.columns = ['isbn', 'user_id', 'rating', 'timestamp', 'review']
    logger.info(f"  Records: {len(df):,}, Users: {df['user_id'].nunique():,}, Items: {df['isbn'].nunique():,}")

    logger.info("Cleaning data...")
    df = df.drop_duplicates(subset=['user_id', 'isbn'], keep='last')
    df = df.dropna(subset=['rating', 'timestamp'])
    df = df[df['rating'] > 0]

    logger.info("Filtering active users (>=3 interactions)...")
    user_counts = df.groupby('user_id').size()
    active_users = user_counts[user_counts >= 3].index
    df = df[df['user_id'].isin(active_users)]
    logger.info(f"  Active users: {len(active_users):,}, Records: {len(df):,}")

    logger.info("Splitting train/val/test (Leave-Last-Out)...")
    df = df.sort_values(['user_id', 'timestamp'])

    train_list = []
    val_list = []
    test_list = []

    for user_id, group in tqdm(df.groupby('user_id'), desc="  Splitting"):
        group = group.sort_values('timestamp')
        test_list.append(group.iloc[-1])
        val_list.append(group.iloc[-2])
        train_list.extend(group.iloc[:-2].to_dict('records'))

    train_df = pd.DataFrame(train_list)
    val_df = pd.DataFrame(val_list)
    test_df = pd.DataFrame(test_list)

    logger.info(f"  Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    train_df.to_csv(output_dir / 'train.csv', index=False)
    val_df.to_csv(output_dir / 'val.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)
    pd.DataFrame({'user_id': active_users}).to_csv(output_dir / 'active_users.csv', index=False)

    with open(output_dir / 'stats.txt', 'w') as f:
        for k, v in [('total_records', len(df)), ('train_records', len(train_df)),
                     ('val_records', len(val_df)), ('test_records', len(test_df)),
                     ('active_users', len(active_users)), ('books', df['isbn'].nunique())]:
            f.write(f'{k}: {v:,}\n')

    logger.info("Split complete in %.1fs", time.time() - start_time)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run()

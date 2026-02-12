# Hugging Face Spaces 部署指南 (Deployment Guide)

本指南介绍如何将项目部署到 Hugging Face Spaces，并配置数据持久化存储。

## 1. 准备工作

确保本地代码已提交到 Git。

```bash
# 1. 在 Hugging Face 创建一个新的 Space
#    - SDK: Docker
#    - Hardware: CPU Basic (Free) or GPU (Paid)

# 2. Add Remote
git remote add space https://huggingface.co/spaces/USERNAME/SPACE_NAME
```

## 2. 配置持久化存储 (Data Persistence)

Hugging Face Spaces 默认重启后会重置文件系统。为了保存用户数据（评分、收藏），需要挂载持久化存储。

### 步骤 A: 申请 Persistent Storage (推荐)
1.  进入 Space -> **Settings**.
2.  找到 **Persistent Storage**.
3.  选择 **Small (Free on standard CPU spaces, or Paid tiered)**.
4.  挂载路径填写 `/data/user`.

### 步骤 B: 配置环境变量
1.  进入 Space -> **Settings** -> **Variables and secrets**.
2.  添加变量：
    *   `USER_DATA_PATH`: `/data/user`  (对应挂载路径)

此配置会让应用将 `user_profiles.json` 保存到挂载的持久盘中，重启 Space 数据不会丢失。

## 3. 推送代码

```bash
git push space main
```

## 4. 常见问题 (FAQ)

### Q: 用户数据可以实时更新推荐吗？
**答**: 
*   **可以实现实时过滤和重排序**: 用户点了“喜欢”或“评分”后，系统会立即记录到 `user_profiles.json`。推荐接口 (`RecommendationService.get_recommendations`) 每次请求都会实时读取最新的用户历史，用于：
    1.  **去重**: 过滤掉已读/已收藏的书。
    2.  **特征生成**: 比如 `sim_max` (与最近交互物品的相似度) 会根据即时历史变化，从而影响排序分数。
*   **但模型不会自动重训**: 模型的权重是固定的。要让模型学到新的协同过滤模式 (Collaborative Filtering Patterns)，需要定期(如每周)在后台运行训练脚本 `scripts/model/train_ranker.py` 并更新模型文件。这通常通过 CI/CD 或 Scheduled Jobs 完成，而不是在 Space 在线服务中进行。

### Q: `chroma_db` 需要持久化吗？
**答**: 
*   如果你只提供只读搜索（不让用户上传新书），则不需要。当前的 ChromaDB 数据可以直接打包在 Docker 镜像中。
*   如果需要支持上传新书，则需要将 `DATA_DIR` 也指向持久化路径。

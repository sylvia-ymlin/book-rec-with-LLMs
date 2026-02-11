# 纸间留白 · 私人书斋（前端）

React + Vite 前端，使用 Tailwind CDN 与 lucide-react 图标，连接现有 FastAPI 后端。

## 开发

1. 安装依赖：

```bash
cd web
npm install
```

2. 本地开发：

```bash
npm run dev
```

默认端口 `5173`。后端需运行在 `http://localhost:6006`。

如需修改后端地址，创建 `.env` 并设置：

```
VITE_API_URL=http://localhost:6006
```

## 后端接口
- `POST /recommend` { query, category, tone } → 书籍列表
- `POST /favorites/add` { isbn, user_id } → 收藏计数
- `GET /user/{user_id}/persona` → 用户画像
- `POST /marketing/highlights` { isbn, user_id } → 个性化卖点

## 注意
- 已在后端添加 CORS 允许 `http://localhost:5173`。
- 初版将中文分类/心境映射到后端 `category/tone`，可根据数据进一步细化。

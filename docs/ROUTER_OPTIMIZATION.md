# 路由器優化指南

## 現狀

- **預設**: 當 `data/model/intent_classifier.pkl` 不存在時，使用規則路由（關鍵詞 + 正則）
- **規則路由**: 依賴 `config/router.json` 的 `detail_keywords`、`natural_language_keywords`，易受同義詞、表述變化影響

## 輕量級分類器（已支援）

`IntentClassifier` 已支援三種後端，可替換規則路由：

| 後端 | 延遲 | 準確度 | 依賴 |
|------|------|--------|------|
| TF-IDF + LogisticRegression | ~1–2ms | 中等 | sklearn（預設） |
| FastText | ~1ms | 較高 | `pip install fasttext` |
| DistilBERT（zero-shot） | ~50–100ms | 最高 | `transformers` |

## 啟用步驟

```bash
# 1. 訓練 TF-IDF 分類器（預設，無額外依賴）
python scripts/model/train_intent_router.py

# 2. 或使用 FastText（更快、更魯棒）
python scripts/model/train_intent_router.py --backend fasttext

# 3. 或使用 DistilBERT（最高準確度，需較多算力）
python scripts/model/train_intent_router.py --backend distilbert
```

輸出會保存到 `data/model/intent_classifier.pkl`（或 `.bin` for FastText）。Router 會在啟動時自動載入。

## 擴充訓練資料

預設使用合成 seed data。若要提升泛化，可新增真實標註：

```bash
# 建立 data/intent_labels.csv，格式：query,intent
# intent 選項：small_to_big, fast, deep
python scripts/model/train_intent_router.py --data data/intent_labels.csv
```

## 驗證

訓練後重啟服務，觀察日誌應出現 `Intent classifier loaded from ...`，而非 `Router (rules): ...`。

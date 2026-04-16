"""
运营 KPI 端点 (v2.1) — 供产品经理和运营团队使用。

端点
----
GET /ops/kpi          业务 KPI 快照（LLM evaluate 质量指标 + 缓存状态）
GET /ops/cache/stats  LLM evaluate 缓存详情
GET /ops/cache/clear  清空 LLM evaluate 缓存（管理员）

所有端点均不依赖外部服务，直接读取进程内状态，响应 < 5ms。
"""
from fastapi import APIRouter, Header, HTTPException
from typing import Optional
import os

from src.support.agentic.llm_evaluate_cache import llm_evaluate_cache
from src.infra.utils import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/ops", tags=["Operations"])

# 简单 ops token 保护（非生产级 auth；完整 auth 属 v2.3 范畴）
_OPS_TOKEN = os.getenv("OPS_API_TOKEN", "")


def _require_token(token: Optional[str]):
    if _OPS_TOKEN and token != f"Bearer {_OPS_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid ops token")


@router.get(
    "/kpi",
    summary="业务 KPI 快照",
    description="""返回 LLM evaluate 质量指标快照，供产品经理 / 运营监控使用。

字段说明：
- `cache_live_entries`：当前 TTL 内有效缓存条目数
- `cache_hit_rate_approx`：近似缓存命中率（live / total；精确数据看 Prometheus）
- `llm_evaluate_enabled`：当前 USE_LLM_EVALUATE 配置值
- `quality_threshold`：当前语义质量阈值
- `provider`：当前 LLM provider
""",
    responses={200: {"description": "KPI snapshot"}, 401: {"description": "Unauthorized"}},
)
async def get_kpi(authorization: Optional[str] = Header(default=None)):
    _require_token(authorization)
    from src.infra.config import (
        USE_LLM_EVALUATE,
        LLM_QUALITY_THRESHOLD,
        LLM_PROVIDER,
        LLM_EVALUATE_TIMEOUT,
    )

    cache_stats = llm_evaluate_cache.stats()
    total = cache_stats["total_entries"]
    live = cache_stats["live_entries"]
    hit_rate_approx = round(live / total, 3) if total > 0 else 0.0

    return {
        "version": "v2.1",
        "llm_evaluate_enabled": USE_LLM_EVALUATE,
        "quality_threshold": LLM_QUALITY_THRESHOLD,
        "provider": LLM_PROVIDER,
        "llm_evaluate_timeout_s": LLM_EVALUATE_TIMEOUT,
        "cache": {
            **cache_stats,
            "hit_rate_approx": hit_rate_approx,
        },
        "sla_targets": {
            "semantic_fix_rate_min": 0.60,
            "false_trigger_rate_max": 0.20,
            "p95_latency_ms_max": 6000,
            "degradation_rate_max": 0.20,
        },
    }


@router.get(
    "/cache/stats",
    summary="LLM evaluate 缓存详情",
    description="返回 LLMEvaluateCache 当前状态：条目数、TTL、容量。",
    responses={200: {"description": "Cache stats"}},
)
async def cache_stats(authorization: Optional[str] = Header(default=None)):
    _require_token(authorization)
    return llm_evaluate_cache.stats()


@router.delete(
    "/cache/clear",
    summary="清空 LLM evaluate 缓存",
    description="强制清空进程内 LLM evaluate 缓存（下次请求将重新调用 LLM）。需要 OPS_API_TOKEN。",
    responses={
        200: {"description": "Cache cleared"},
        401: {"description": "Unauthorized"},
    },
)
async def cache_clear(authorization: Optional[str] = Header(default=None)):
    _require_token(authorization)
    before = llm_evaluate_cache.stats()["total_entries"]
    llm_evaluate_cache.clear()
    logger.info("OPS: LLM evaluate cache cleared (%d entries removed)", before)
    return {"status": "cleared", "entries_removed": before}

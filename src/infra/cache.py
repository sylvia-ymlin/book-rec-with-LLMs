import redis
import json
from typing import Optional, Any
from src.infra.config import REDIS_URL, CACHE_TTL
from src.infra.utils import setup_logger

logger = setup_logger(__name__)


class CacheManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize Redis connection."""
        try:
            self.client = redis.from_url(REDIS_URL, decode_responses=True)
            # Test connection
            self.client.ping()
            self.enabled = True
            logger.info(f"Redis cache initialized successfully: {REDIS_URL}")
        except redis.ConnectionError as e:
            self.enabled = False
            logger.warning(f"Redis cache initialization failed: {e}. Caching disabled.")
        except Exception as e:
            self.enabled = False
            logger.warning(f"Unexpected Redis error: {e}. Caching disabled.")

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache."""
        if not self.enabled:
            return None
        try:
            val = self.client.get(key)
            if val:
                logger.debug(f"Cache HIT for key: {key}")
                return json.loads(val)
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
        return None

    def set(self, key: str, value: Any, ttl: int = CACHE_TTL) -> bool:
        """Set value in cache with TTL."""
        if not self.enabled:
            return False
        try:
            self.client.setex(key, ttl, json.dumps(value))
            return True
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False

    def generate_key(self, prefix: str, **kwargs) -> str:
        """Generate a consistent cache key from arguments."""
        sorted_kwargs = dict(sorted(kwargs.items()))
        key_part = "_".join([f"{k}:{v}" for k, v in sorted_kwargs.items()])
        return f"{prefix}:{key_part}"


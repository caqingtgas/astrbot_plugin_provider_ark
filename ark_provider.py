# /opt/AstrBot/data/plugins/astrbot_plugin_provider_ark/ark_provider.py
import aiohttp
from typing import List, Dict, Tuple, Optional
from types import SimpleNamespace

from astrbot.api import logger  # 规范的日志导入
from astrbot.core.provider.provider import Provider
from astrbot.core.provider.register import register_provider_adapter

# 兼容新旧路径
try:
    from astrbot.core.provider.entities import LLMResponse, ProviderType
except Exception:  # pragma: no cover
    from astrbot.core.provider.entites import LLMResponse, ProviderType  # type: ignore


@register_provider_adapter(
    "ark_context",
    "Volcengine Ark (Context API)",
    provider_type=ProviderType.CHAT_COMPLETION,
)
class ArkContextProvider(Provider):
    """
    非流式 Provider（极简稳定）：
    - 首轮 /context/create 写入 system prompt，后续复用 context_id
    - 返回 LLMResponse(role='assistant', completion_text=...)，由 AstrBot 发送
    - raw_completion.usage 回填（token 统计插件读取）
    """

    def __init__(self, provider_config: dict, provider_settings: dict, default_persona=None):
        super().__init__(provider_config, provider_settings, default_persona)
        self._keys: List[str] = [(k or "").strip() for k in provider_config.get("key", []) if (k or "").strip()]
        self._key_idx: int = 0
        self._base: str = (provider_config.get("api_base") or "https://ark.cn-beijing.volces.com/api/v3").strip().rstrip("/")
        self._model_default: str = (provider_config.get("model") or provider_config.get("model_config", {}).get("model") or "").strip()
        self._ttl: int = int(provider_config.get("ttl", 86400))
        self._session: Optional[aiohttp.ClientSession] = None
        self._ctx_map: Dict[str, str] = {}  # AstrBot session_id -> Ark context_id

        masked = (self._keys[0][:6] + "..." + self._keys[0][-4:]) if self._keys else "EMPTY"
        logger.info(f"[ArkProvider] init base={self._base}, model={self._model_default}, key={masked}")

    # ===== Provider 基础能力 =====
    def get_models(self) -> List[str]:
        models = self.provider_config.get("models")
        return models if isinstance(models, list) and models else ([self._model_default] if self._model_default else [])

    def get_current_key(self) -> str:
        return self._keys[self._key_idx] if self._keys else ""

    def set_key(self, key: str):
        k = (key or "").strip()
        if k in self._keys:
            self._key_idx = self._keys.index(k)

    # ===== Ark HTTP =====
    async def _sess(self) -> aiohttp.ClientSession:
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))
        return self._session

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.get_current_key().strip()}", "Content-Type": "application/json"}

    async def _context_create(self, model: str, system_prompt: str) -> str:
        url = f"{self._base}/context/create"
        payload = {
            "model": model,
            "mode": "session",
            "ttl": self._ttl,
            "messages": [{"role": "system", "content": (system_prompt or "").strip()}],
        }
        async with (await self._sess()).post(url, json=payload, headers=self._headers()) as resp:
            data = await (resp.json() if resp.content_type == "application/json" else resp.text())
            if isinstance(data, dict) and "id" in data and resp.status < 400:
                ctx_id = data["id"]
                logger.info(f"[ArkProvider] context created: {ctx_id}")
                return ctx_id
            raise RuntimeError(f"Ark context_create failed: {data}")

    async def _context_chat(self, model: str, ctx_id: str, user_text: str) -> Tuple[str, dict]:
        url = f"{self._base}/context/chat/completions"
        payload = {
            "model": model,
            "context_id": ctx_id,
            "messages": [{"role": "user", "content": str(user_text)}],
            "stream": False,
        }
        async with (await self._sess()).post(url, json=payload, headers=self._headers()) as resp:
            data = await (resp.json() if resp.content_type == "application/json" else resp.text())
            if isinstance(data, dict) and resp.status < 400:
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {}) or {}
                logger.info(f"[ArkProvider] chat ok ctx={ctx_id} len={len(content)}")
                return content, usage
            raise RuntimeError(f"Ark context_chat failed: {data}")

    # ===== AstrBot Provider 入口 =====
    async def text_chat(
        self,
        prompt: str,
        session_id: str = None,
        image_urls: List[str] = None,
        func_tool=None,
        contexts: List[dict] = None,
        system_prompt: str = None,
        tool_calls_result=None,
        model: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        model_name = (model or self.get_model() or self._model_default).strip()
        if not model_name:
            raise RuntimeError("ArkProvider: model not configured (use ep-... endpoint id)")
        if not self.get_current_key():
            raise RuntimeError("ArkProvider: API Key not configured")

        # /reset 配合：AstrBot 清空历史后（contexts 为空），丢弃旧 ctx 强制重建
        skey = (session_id or "global").strip()
        if not contexts and skey in self._ctx_map:
            dropped = self._ctx_map.pop(skey, None)
            logger.info(f"[ArkProvider] reset detected -> drop ctx={dropped} skey={skey}")

        # 命中/创建 Ark 上下文
        ctx_id = self._ctx_map.get(skey)
        if not ctx_id:
            ctx_id = await self._context_create(model_name, system_prompt or "")
            self._ctx_map[skey] = ctx_id
            logger.info(f"[ArkProvider] create ctx={ctx_id} skey={skey}")

        # 请求 Ark
        content, usage = await self._context_chat(model_name, ctx_id, prompt)
        text = str(content)

        # —— 统计 & 记录缓存命中数（prompt_tokens_details.cached_tokens）——
        cached_tokens = 0
        try:
            cached_tokens = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
        except Exception:
            pass

        # 记录到实例属性，便于后续指令查询
        if not hasattr(self, "_last_usage"):
            self._last_usage: Dict[str, dict] = {}
        self._last_usage[skey] = usage
        self._last_recent: Tuple[str, dict] = (skey, usage)

        # 控制台输出命中情况
        logger.info(
            "[ArkProvider] usage: prompt=%s completion=%s total=%s cached_tokens=%s",
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            usage.get("total_tokens", 0),
            cached_tokens,
        )

        # 最稳返回：completion_text 路径
        resp = LLMResponse(role="assistant", completion_text=text)

        # usage 回填（使用 SimpleNamespace 提升可读性）
        raw = SimpleNamespace()
        raw.usage = SimpleNamespace()
        raw.usage.prompt_tokens = int(usage.get("prompt_tokens", 0))
        raw.usage.completion_tokens = int(usage.get("completion_tokens", 0))
        raw.usage.total_tokens = int(
            usage.get("total_tokens", raw.usage.prompt_tokens + raw.usage.completion_tokens)
        )
        resp.raw_completion = raw

        # 透传更多细节，便于外部读取（含缓存命中数）
        try:
            resp.extra = {
                "ark_usage": usage,
                "ark_cached_tokens": cached_tokens,
                "ark_context_id": ctx_id,
            }
        except AttributeError as e:
            logger.warning(f"[ArkProvider] 设置响应 extra 属性时出错: {e}")
        except Exception as e:
            logger.warning(f"[ArkProvider] 设置响应 extra 属性时发生未知错误: {e}")

        # 兼容字段
        for attr in ("text", "answer", "plain_text", "content", "message"):
            try:
                setattr(resp, attr, text)
            except AttributeError as e:
                logger.warning(f"[ArkProvider] 设置响应 {attr} 属性时出错: {e}")
            except Exception as e:
                logger.warning(f"[ArkProvider] 设置响应 {attr} 属性时发生未知错误: {e}")

        return resp

    async def close(self):
        """显式资源释放；可在插件生命周期里调用"""
        try:
            if self._session and not self._session.closed:
                await self._session.close()
        except Exception as e:
            logger.warning(f"[ArkProvider] 关闭 HTTP 会话时出错: {e}")

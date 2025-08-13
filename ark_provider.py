# /opt/AstrBot/data/plugins/astrbot_plugin_provider_ark/ark_provider.py
import asyncio
import aiohttp
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from types import SimpleNamespace

from astrbot.api import logger
from astrbot.api.star import StarTools
from astrbot.core.provider.provider import Provider
from astrbot.core.provider.register import register_provider_adapter
from astrbot.core.provider.entities import LLMResponse, ProviderType


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

        # ---- 基础配置 ----
        self._keys: List[str] = [(k or "").strip() for k in provider_config.get("key", []) if (k or "").strip()]
        self._key_idx: int = 0
        self._base: str = (provider_config.get("api_base") or "https://ark.cn-beijing.volces.com/api/v3").strip().rstrip("/")
        self._model_default: str = (provider_config.get("model") or provider_config.get("model_config", {}).get("model") or "").strip()
        self._ttl: int = int(provider_config.get("ttl", 86400))
        self._timeout: int = int(provider_config.get("timeout", 60))  # 可配置

        # ---- 状态 ----
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_usage: Dict[str, dict] = {}
        self._ctx_map: Dict[str, str] = {}  # AstrBot session_id -> Ark context_id

        # ---- 持久化（默认开启） ----
        self._persist_ctx: bool = bool(provider_config.get("persist_ctx", True))
        data_dir = StarTools.get_data_dir("astrbot_plugin_provider_ark")
        self._ctx_file: Path = Path(data_dir) / "ctx_map.json"
        self._ctx_lock = asyncio.Lock()
        self._ctx_loaded: bool = False  # 懒加载标记（不在 __init__ 里读盘）

        masked = (self._keys[0][:6] + "..." + self._keys[0][-4:]) if self._keys else "EMPTY"
        logger.info(
            "[ArkProvider] init base=%s, model=%s, key=%s, persist_ctx=%s, ctx_path=%s, timeout=%s",
            self._base, self._model_default, masked, self._persist_ctx, self._ctx_file, self._timeout
        )

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
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._timeout))
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
                logger.info("[ArkProvider] context created: %s", ctx_id)
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
                logger.info("[ArkProvider] chat ok ctx=%s len=%d", ctx_id, len(content))
                return content, usage
            raise RuntimeError(f"Ark context_chat failed: {data}")

    # ===== 私有辅助：懒加载 ctx_map =====
    async def _ensure_ctx_loaded(self):
        if self._ctx_loaded or not self._persist_ctx:
            self._ctx_loaded = True
            return
        p = self._ctx_file

        def _read_sync() -> Dict[str, str]:
            if not p.exists():
                return {}
            data = json.loads(p.read_text(encoding="utf-8"))
            m = data.get("map", {})
            return {str(k): str(v) for k, v in m.items()} if isinstance(m, dict) else {}

        try:
            self._ctx_map = await asyncio.to_thread(_read_sync)
            if self._ctx_map:
                logger.info("[ArkProvider] loaded %d ctx bindings from %s", len(self._ctx_map), p)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("[ArkProvider] failed to load ctx_map from %s: %s", p, e)
        finally:
            self._ctx_loaded = True

    # ===== 私有辅助：保存 ctx_map（互斥 + 原子写 + 异步线程）=====
    async def _save_ctx_map(self):
        if not self._persist_ctx:
            return
        p = self._ctx_file

        def _write_sync():
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps({"map": self._ctx_map}, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(p)

        async with self._ctx_lock:
            try:
                await asyncio.to_thread(_write_sync)
            except (OSError, PermissionError) as e:
                logger.warning("[ArkProvider] failed to save ctx_map to %s: %s", p, e)

    # ===== 私有辅助：获取或创建 context =====
    async def _get_or_create_context(
        self, skey: str, model_name: str, system_prompt: str, contexts: Optional[List[dict]]
    ) -> str:
        # /reset 配合：AstrBot 清空历史后（contexts 为空），丢弃旧 ctx 强制重建
        if not contexts and skey in self._ctx_map:
            dropped = self._ctx_map.pop(skey, None)
            logger.info("[ArkProvider] reset detected -> drop ctx=%s skey=%s", dropped, skey)
            await self._save_ctx_map()

        ctx_id = self._ctx_map.get(skey)
        if not ctx_id:
            ctx_id = await self._context_create(model_name, system_prompt or "")
            self._ctx_map[skey] = ctx_id
            await self._save_ctx_map()
            logger.info("[ArkProvider] create ctx=%s skey=%s", ctx_id, skey)
        return ctx_id

    # ===== 私有辅助：聊天并按需重试 =====
    async def _perform_chat_with_retry(
        self, model_name: str, ctx_id: str, prompt: str, system_prompt: str, skey: str
    ) -> Tuple[str, dict, str]:
        try:
            content, usage = await self._context_chat(model_name, ctx_id, prompt)
            return content, usage, ctx_id
        except RuntimeError as e:
            msg = str(e).lower()
            if "context" in msg and any(k in msg for k in ("invalid", "expire", "not found", "does not exist")):
                logger.warning("[ArkProvider] ctx invalid -> recreate & retry. old=%s skey=%s", ctx_id, skey)
                new_ctx = await self._context_create(model_name, system_prompt or "")
                self._ctx_map[skey] = new_ctx
                await self._save_ctx_map()
                content, usage = await self._context_chat(model_name, new_ctx, prompt)
                return content, usage, new_ctx
            raise

    # ===== 私有辅助：构建 LLMResponse =====
    def _build_llm_response(self, text: str, usage: dict, ctx_id: str) -> LLMResponse:
        resp = LLMResponse(role="assistant", completion_text=str(text))

        # usage 回填（更健壮的 total_tokens）
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = usage.get("total_tokens")
        try:
            total = int(total_tokens) if total_tokens is not None else prompt_tokens + completion_tokens
        except (ValueError, TypeError):
            logger.warning("[ArkProvider] 无法将 total_tokens ('%s') 转为整数，回退为计算值", total_tokens)
            total = prompt_tokens + completion_tokens

        raw = SimpleNamespace()
        raw.usage = SimpleNamespace()
        raw.usage.prompt_tokens = prompt_tokens
        raw.usage.completion_tokens = completion_tokens
        raw.usage.total_tokens = total
        resp.raw_completion = raw

        # extra（含缓存命中）
        cached_tokens = 0
        ptd = usage.get("prompt_tokens_details")
        if isinstance(ptd, dict):
            try:
                cached_tokens = int(ptd.get("cached_tokens", 0))
            except (TypeError, ValueError) as e:
                logger.warning("[ArkProvider] cached_tokens parse error: %s", e)
        try:
            resp.extra = {"ark_usage": usage, "ark_cached_tokens": cached_tokens, "ark_context_id": ctx_id}
        except AttributeError as e:
            logger.warning("[ArkProvider] 设置响应 extra 属性时出错: %s", e)

        return resp

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
        await self._ensure_ctx_loaded()

        model_name = (model or self.get_model() or self._model_default).strip()
        if not model_name:
            raise RuntimeError("ArkProvider: model not configured (use ep-... endpoint id)")
        if not self.get_current_key():
            raise RuntimeError("ArkProvider: API Key not configured")

        skey = (session_id or "global").strip()
        ctx_id = await self._get_or_create_context(skey, model_name, system_prompt or "", contexts)

        content, usage, ctx_id = await self._perform_chat_with_retry(
            model_name, ctx_id, prompt, system_prompt or "", skey
        )
        text = str(content)

        # 记录最近 usage
        self._last_usage[skey] = usage
        self._last_recent: Tuple[str, dict] = (skey, usage)

        # usage 概览日志（含缓存命中）
        cached_tokens = 0
        ptd = usage.get("prompt_tokens_details")
        if isinstance(ptd, dict):
            try:
                cached_tokens = int(ptd.get("cached_tokens", 0))
            except (TypeError, ValueError) as e:
                logger.warning("[ArkProvider] cached_tokens parse error: %s", e)
        logger.info(
            "[ArkProvider] usage: prompt=%s completion=%s total=%s cached_tokens=%s",
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            usage.get("total_tokens", 0),
            cached_tokens,
        )

        return self._build_llm_response(text, usage, ctx_id)

    async def close(self):
        try:
            if self._session and not self._session.closed:
                await self._session.close()
        except Exception as e:
            logger.warning("[ArkProvider] 关闭 HTTP 会话时出错: %s", e)

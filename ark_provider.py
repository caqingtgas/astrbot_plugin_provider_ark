import asyncio
import aiohttp
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from types import SimpleNamespace
import uuid

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
    - ✅ 支持工具调用（tools / tool_calls）并驱动 AstrBot 的 ToolLoopAgent
    """

    def __init__(self, provider_config: dict, provider_settings: dict, default_persona=None):
        """
        初始化 Provider。

        Args:
            provider_config: AstrBot 中为该 Provider 配置的字典（key、api_base、ttl、timeout、persist_ctx 等）。
            provider_settings: 由框架注入的运行时设置。
            default_persona: 默认人格（未使用）。
        """
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
        self._ctx_map: Dict[str, str] = {}  # AstrBot session_id -> Ark context_id

        # ---- 持久化（默认开启） ----
        self._persist_ctx: bool = bool(provider_config.get("persist_ctx", True))
        data_dir = StarTools.get_data_dir("astrbot_plugin_provider_ark")
        self._ctx_file: Path = Path(data_dir) / "ctx_map.json"
        self._ctx_lock = asyncio.Lock()          # I/O 与懒加载互斥
        self._ctx_loaded: bool = False           # 懒加载标记（不在 __init__ 里读盘）

        # ---- 逐键互斥锁（为 skey 的 context 创建提供互斥，解决并发竞态）----
        self._ctx_locks: Dict[str, asyncio.Lock] = {}
        self._ctx_locks_lock = asyncio.Lock()    # 保护 _ctx_locks 字典自身

        masked = (self._keys[0][:6] + "..." + self._keys[0][-4:]) if self._keys else "EMPTY"
        logger.info(
            "[ArkProvider] init base=%s, model=%s, key=%s, persist_ctx=%s, ctx_path=%s, timeout=%s",
            self._base, self._model_default, masked, self._persist_ctx, self._ctx_file, self._timeout
        )

    # ===== Provider 基础能力 =====
    def get_models(self) -> List[str]:
        """返回可用模型列表（来自 provider_config.models 或默认单模型）。"""
        models = self.provider_config.get("models")
        return models if isinstance(models, list) and models else ([self._model_default] if self._model_default else [])

    def get_current_key(self) -> str:
        """获取当前使用的 API Key（支持多 Key 轮换选择）。"""
        return self._keys[self._key_idx] if self._keys else ""

    def set_key(self, key: str):
        """设置当前使用的 API Key。"""
        k = (key or "").strip()
        if k in self._keys:
            self._key_idx = self._keys.index(k)

    # ===== Ark HTTP =====
    async def _sess(self) -> aiohttp.ClientSession:
        """获取（或创建）共享的 aiohttp.ClientSession，会使用可配置的总超时。"""
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._timeout))
        return self._session

    def _headers(self) -> dict:
        """构造 Ark API 所需请求头。"""
        return {"Authorization": f"Bearer {self.get_current_key().strip()}", "Content-Type": "application/json"}

    async def _context_create(self, model: str, system_prompt: str) -> str:
        """创建 Ark 对话上下文（Context）。"""
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

    # ===== 工具：入参转换 =====
    def _to_tools(self, func_tool) -> Optional[List[dict]]:
        """
        将 AstrBot 的 FuncTool/工具描述 转为 OpenAI/Ark 兼容的 tools 列表：
        [{"type":"function","function":{"name":..., "description":..., "parameters":{...}}}, ...]
        """
        if not func_tool:
            return None
        tools: List[dict] = []
        try:
            seq = func_tool if isinstance(func_tool, (list, tuple)) else [func_tool]
            for t in seq:
                name = getattr(t, "name", None) or getattr(t, "func_name", None)
                params = getattr(t, "parameters", None) or {}
                desc = getattr(t, "description", "") or getattr(t, "desc", "")
                if not name:
                    continue
                tools.append({
                    "type": "function",
                    "function": {
                        "name": str(name),
                        "description": str(desc) if desc else "",
                        "parameters": params if isinstance(params, dict) else {}
                    }
                })
        except Exception as e:
            logger.warning("[ArkProvider] 转换 tools 失败: %s", e)
        return tools or None

    def _tool_results_to_messages(self, tool_calls_result) -> List[dict]:
        """
        将 AstrBot 回传的工具执行结果转为本轮追加消息（role='tool'）。
        预期元素示例：
            {'id': 'call_xxx', 'name': 'web_search', 'result': '...'}
            或 {'tool_call_id': 'call_xxx', 'name': 'web_search', 'content': '...'}
        """
        if not tool_calls_result:
            return []
        results = tool_calls_result if isinstance(tool_calls_result, (list, tuple)) else [tool_calls_result]
        msgs: List[dict] = []
        for r in results:
            try:
                tool_call_id = r.get("id") or r.get("tool_call_id")
                name = r.get("name")
                content = r.get("result") if "result" in r else r.get("content", "")
                if tool_call_id and name is not None:
                    msgs.append({
                        "role": "tool",
                        "tool_call_id": str(tool_call_id),
                        "name": str(name),
                        "content": str(content or "")
                    })
            except Exception as e:
                logger.warning("[ArkProvider] 解析 tool_calls_result 失败: %s, data=%r", e, r)
        return msgs

    @staticmethod
    def _normalize_tool_calls(assistant_message: dict) -> Tuple[List[dict], Optional[dict]]:
        """
        把模型返回的工具调用统一为 tool_calls（新样式），并兼容旧式 function_call。
        返回 (tool_calls, function_call_or_none)
        """
        tool_calls = assistant_message.get("tool_calls") or []
        function_call = assistant_message.get("function_call")  # 旧版单函数
        if tool_calls:
            return tool_calls, None
        if function_call and isinstance(function_call, dict):
            # 包装为单条 tool_calls，以便上游统一处理
            call = {
                "id": "call_" + uuid.uuid4().hex[:8],
                "type": "function",
                "function": {
                    "name": function_call.get("name") or "",
                    "arguments": function_call.get("arguments") or ""
                }
            }
            return [call], function_call
        return [], None

    # ===== Chat：支持 tools / tool_choice =====
    async def _context_chat(
        self,
        model: str,
        ctx_id: str,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        tool_choice: str | dict = "auto",
    ) -> Tuple[dict, dict]:
        """
        使用既有 context 进行一次非流式对话请求；支持工具调用。
        返回 (assistant_message, usage)
        """
        url = f"{self._base}/context/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "context_id": ctx_id,
            "messages": messages,  # 本轮增量消息（含 user / tool）
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice  # "auto" | {"type":"function","function":{"name":"xxx"}}

        async with (await self._sess()).post(url, json=payload, headers=self._headers()) as resp:
            data = await (resp.json() if resp.content_type == "application/json" else resp.text())
            if isinstance(data, dict) and resp.status < 400:
                msg = ((data.get("choices") or [{}])[0] or {}).get("message") or {}
                usage = data.get("usage", {}) or {}
                has_tc = bool((msg or {}).get("tool_calls"))
                logger.info("[ArkProvider] chat ok ctx=%s has_tool_calls=%s", ctx_id, has_tc)
                return msg, usage
            raise RuntimeError(f"Ark context_chat failed: {data}")

    # ===== 懒加载：读取 ctx_map（加锁 + 二次检查）=====
    async def _ensure_ctx_loaded(self):
        """
        懒加载上下文映射表：在首次使用前从磁盘加载到内存。
        加锁并二次检查，避免并发重复加载；I/O 在线程池中执行，避免阻塞事件循环。
        """
        if self._ctx_loaded or not self._persist_ctx:
            return
        async with self._ctx_lock:
            if self._ctx_loaded:
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

    # ===== 保存 ctx_map（互斥 + 原子写 + 异步线程）=====
    async def _save_ctx_map(self):
        """
        将内存中的上下文映射表保存到磁盘。
        使用互斥锁与原子替换，I/O 在线程池执行。
        """
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

    # ===== 逐键互斥：获取某个 skey 的锁 =====
    async def _get_key_lock(self, skey: str) -> asyncio.Lock:
        """返回该 skey 对应的互斥锁；必要时在受保护字典内创建。"""
        async with self._ctx_locks_lock:
            lock = self._ctx_locks.get(skey)
            if lock is None:
                lock = asyncio.Lock()
                self._ctx_locks[skey] = lock
            return lock

    # ===== 获取或创建 context（逐键加锁，避免并发竞态）=====
    async def _get_or_create_context(
        self, skey: str, model_name: str, system_prompt: str, contexts: Optional[List[dict]]
    ) -> str:
        """
        获取（或在必要时创建）给定 skey 的 Ark context。
        将“检查是否存在 + 创建 + 写入映射”作为一个临界区，在 per-skey 互斥锁下执行，避免并发重复创建。
        """
        key_lock = await self._get_key_lock(skey)
        async with key_lock:
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

    # ===== 聊天并按需重试 =====
    async def _perform_chat_with_retry(
        self,
        model_name: str,
        ctx_id: str,
        messages: List[dict],
        tools: Optional[List[dict]],
        tool_choice: str | dict,
        system_prompt: str,
        skey: str,
    ) -> Tuple[dict, dict, str]:
        """
        调用 Ark 接口进行对话；若因 context 失效/过期导致失败，则自动重建一次并重试。
        返回 (assistant_message, usage, ctx_id)
        """
        try:
            msg, usage = await self._context_chat(model_name, ctx_id, messages, tools=tools, tool_choice=tool_choice)
            return msg, usage, ctx_id
        except RuntimeError as e:
            msg_text = str(e).lower()
            if "context" in msg_text and any(k in msg_text for k in ("invalid", "expire", "not found", "does not exist")):
                logger.warning("[ArkProvider] ctx invalid -> recreate & retry. old=%s skey=%s", ctx_id, skey)
                new_ctx = await self._context_create(model_name, system_prompt or "")
                self._ctx_map[skey] = new_ctx
                await self._save_ctx_map()
                msg, usage = await self._context_chat(model_name, new_ctx, messages, tools=tools, tool_choice=tool_choice)
                return msg, usage, new_ctx
            raise

    # ===== 解析 cached_tokens =====
    @staticmethod
    def _parse_cached_tokens(usage: dict) -> int:
        """从 usage.prompt_tokens_details.cached_tokens 解析缓存命中数；异常时返回 0。"""
        ptd = usage.get("prompt_tokens_details")
        if isinstance(ptd, dict):
            try:
                return int(ptd.get("cached_tokens", 0))
            except (TypeError, ValueError):
                logger.warning("[ArkProvider] cached_tokens parse error: %r", ptd.get("cached_tokens", None))
        return 0

    # ===== 安全设置属性（避免异常作流程控制）=====
    @staticmethod
    def _safe_set_attr(obj: Any, name: str, value: Any) -> bool:
        """在可写时设置属性；失败返回 False。"""
        if getattr(obj, "__dict__", None) is not None:
            try:
                setattr(obj, name, value)
                return True
            except Exception:
                return False
        slots = getattr(type(obj), "__slots__", ())
        if isinstance(slots, (list, tuple, set)) and name in slots:
            try:
                setattr(obj, name, value)
                return True
            except Exception:
                return False
        return False

    # ===== 构建 LLMResponse =====
    def _build_llm_response(self, text: str, usage: dict, ctx_id: str) -> LLMResponse:
        """
        将模型输出与 usage 组装为 LLMResponse；对 total_tokens 做健壮处理，并在 extra 中包含缓存命中数与 context_id。
        """
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
        cached_tokens = self._parse_cached_tokens(usage)
        extra_payload = {"ark_usage": usage, "ark_cached_tokens": cached_tokens, "ark_context_id": ctx_id}

        if not self._safe_set_attr(resp, "extra", extra_payload):
            rc = getattr(resp, "raw_completion", None)
            if rc is not None:
                self._safe_set_attr(rc, "extra", extra_payload)

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
        """
        AstrBot 调用的主入口：按需懒加载上下文映射，获取/创建 context，调用 Ark API 并返回 LLMResponse。
        这里完成工具调用（function-calling）的透传与解析。
        """
        await self._ensure_ctx_loaded()

        model_name = (model or self.get_model() or self._model_default).strip()
        if not model_name:
            raise RuntimeError("ArkProvider: model not configured (use ep-... endpoint id)")
        if not self.get_current_key():
            raise RuntimeError("ArkProvider: API Key not configured")

        skey = (session_id or "global").strip()
        ctx_id = await self._get_or_create_context(skey, model_name, system_prompt or "", contexts)

        # 1) 组装这轮 messages：用户消息 + （可选）工具执行结果（role='tool'）
        round_messages: List[dict] = [{"role": "user", "content": str(prompt)}]
        round_messages += self._tool_results_to_messages(tool_calls_result)

        # 2) 如果传入了工具描述，构造 tools 并打日志
        tools = self._to_tools(func_tool)
        if tools:
            logger.info("[ArkProvider] tools enabled: %d 个", len(tools))

        # 3) 发起请求（支持 tools），拿到 assistant_message
        assistant_message, usage, ctx_id = await self._perform_chat_with_retry(
            model_name=model_name,
            ctx_id=ctx_id,
            messages=round_messages,
            tools=tools,
            tool_choice="auto",
            system_prompt=system_prompt or "",
            skey=skey,
        )

        # 4) 解析文本与 tool_calls / function_call
        text = str(assistant_message.get("content") or "")
        tool_calls, function_call = self._normalize_tool_calls(assistant_message)

        # 5) 组装响应（沿用 usage 处理），并塞入工具调用信息，方便 ToolLoopAgent 识别
        resp = self._build_llm_response(text, usage, ctx_id)

        # 在可写时，将工具调用结构透传到标准位置
        if tool_calls:
            self._safe_set_attr(resp, "tool_calls", tool_calls)
        if function_call:
            self._safe_set_attr(resp, "function_call", function_call)

        # 同时放入 extra 兜底
        extra_now = getattr(resp, "extra", {}) or {}
        extra_now.update({
            "ark_assistant_message": assistant_message,  # 完整 assistant 消息（含 tool_calls）
            "ark_tool_calls": tool_calls,
            "ark_function_call": function_call,
        })
        if not self._safe_set_attr(resp, "extra", extra_now):
            rc = getattr(resp, "raw_completion", None)
            if rc is not None:
                self._safe_set_attr(rc, "extra", extra_now)

        # 6) usage 概览日志（含 tool_calls 数量）
        cached_tokens = (getattr(resp, "extra", {}) or {}).get("ark_cached_tokens", 0)
        logger.info(
            "[ArkProvider] usage: prompt=%s completion=%s total=%s cached_tokens=%s tool_calls=%d",
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            usage.get("total_tokens", 0),
            cached_tokens,
            len(tool_calls),
        )

        return resp

    async def close(self):
        """关闭 HTTP 会话（由上层生命周期在合适时机调用）。"""
        try:
            if self._session and not self._session.closed:
                await self._session.close()
        except Exception as e:
            logger.warning("[ArkProvider] 关闭 HTTP 会话时出错: %s", e)

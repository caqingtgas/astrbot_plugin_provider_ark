import asyncio
import aiohttp
import json
from typing import List, Dict, Tuple, Optional, Any
from types import SimpleNamespace

from astrbot.api import logger
from astrbot.api.star import StarTools
from astrbot.core.provider.provider import Provider
from astrbot.core.provider.register import register_provider_adapter
from astrbot.core.provider.entities import LLMResponse, ProviderType

ARK_BASE_DEFAULT = "https://ark.cn-beijing.volces.com/api/v3"

@register_provider_adapter(
    "ark_context",  # ← 与配置一致，确保能被加载
    "Volcengine Ark (OpenAI-Compatible Chat Completions)",
    provider_type=ProviderType.CHAT_COMPLETION,
)
class ArkContextProvider(Provider):
    """
    Ark Chat Completions（OpenAI 兼容）Provider：
    - 使用 /chat/completions（非 context 版），可传入 tools 与 role="tool" 消息
    - func_tool -> tools；tool_calls_result -> role="tool" messages
    - 把返回的 tool_calls 写回 LLMResponse，驱动 AstrBot 的 ToolLoopAgent
    """

    def __init__(self, provider_config: dict, provider_settings: dict, default_persona=None):
        super().__init__(provider_config, provider_settings, default_persona)
        self._keys: List[str] = [(k or "").strip() for k in provider_config.get("key", []) if (k or "").strip()]
        self._key_idx: int = 0
        self._base: str = (provider_config.get("api_base") or ARK_BASE_DEFAULT).strip().rstrip("/")
        self._model_default: str = (provider_config.get("model") or provider_config.get("model_config", {}).get("model") or "").strip()
        self._timeout: int = int(provider_config.get("timeout", 60))
        self._session: Optional[aiohttp.ClientSession] = None

        # 仅用于需要的本地数据目录（不做 context 映射）
        StarTools.get_data_dir("astrbot_plugin_provider_ark")  # 确保目录存在即可

        masked = (self._keys[0][:6] + "..." + self._keys[0][-4:]) if self._keys else "EMPTY"
        logger.info("[ArkProvider] init base=%s, model=%s, key=%s, timeout=%s",
                    self._base, self._model_default, masked, self._timeout)

    # -------- 基础能力 --------
    def get_models(self) -> List[str]:
        models = self.provider_config.get("models")
        return models if isinstance(models, list) and models else ([self._model_default] if self._model_default else [])

    def get_current_key(self) -> str:
        return self._keys[self._key_idx] if self._keys else ""

    def set_key(self, key: str):
        k = (key or "").strip()
        if k in self._keys:
            self._key_idx = self._keys.index(k)

    # -------- HTTP --------
    async def _sess(self) -> aiohttp.ClientSession:
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._timeout))
        return self._session

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.get_current_key().strip()}", "Content-Type": "application/json"}

    # -------- 工具与消息构造 --------
    @staticmethod
    def _tools_from_manager(func_tool: Any) -> List[dict]:
        """
        从 AstrBot 的 func_tool 管理器提取 OpenAI 兼容 tools：
        优先使用管理器的导出方法；失败则兜底从内部工具列表提取（不抛错）。
        """
        tools: List[dict] = []
        if not func_tool:
            return tools

        # 优先：常见导出方法
        for attr in ("to_openai_tools", "to_tools", "export_openai_tools"):
            if hasattr(func_tool, attr):
                try:
                    res = getattr(func_tool, attr)()
                    if isinstance(res, list) and res:
                        return res
                except Exception as e:
                    logger.debug("[ArkProvider] func_tool.%s() failed: %s", attr, e)

        # 兜底：读取内部工具集合
        try:
            iterable = None
            if hasattr(func_tool, "tools"):
                iterable = getattr(func_tool, "tools")
            elif hasattr(func_tool, "get_tools"):
                iterable = getattr(func_tool, "get_tools")()
            if iterable:
                for t in iterable:
                    name = getattr(t, "name", None)
                    params = getattr(t, "parameters", None)
                    desc = getattr(t, "description", "") or ""
                    active = getattr(t, "active", True)
                    if name and isinstance(params, dict) and active:
                        tools.append({"type": "function", "function": {"name": name, "description": desc, "parameters": params}})
        except Exception as e:
            logger.debug("[ArkProvider] fallback build tools failed: %s", e)
        return tools

    @staticmethod
    def _append_tool_results(messages: List[dict], tool_calls_result: Any) -> None:
        """
        将 AstrBot 的工具执行结果转为 role="tool" 消息，追加到 messages。
        兼容：对象（有 to_openai_messages 方法）/ 字典 / 列表。
        """
        if not tool_calls_result:
            return
        # 1) 优先走标准导出
        items = tool_calls_result if isinstance(tool_calls_result, (list, tuple)) else [tool_calls_result]
        appended = 0
        for r in items:
            if hasattr(r, "to_openai_messages"):
                try:
                    msgs = r.to_openai_messages()
                    if isinstance(msgs, list):
                        messages.extend(msgs); appended += len(msgs)
                    continue
                except Exception:
                    pass
            # 2) 容错：按常见字段拼
            try:
                rc = r if isinstance(r, dict) else getattr(r, "__dict__", {}) or {}
                tool_call_id = rc.get("tool_call_id") or rc.get("id") or rc.get("call_id")
                name = rc.get("name")
                content = rc.get("result", None) or rc.get("content", None) or rc.get("output", "")
                msg = {"role": "tool", "content": str(content or "")}
                if tool_call_id: msg["tool_call_id"] = str(tool_call_id)
                if name: msg["name"] = str(name)
                messages.append(msg); appended += 1
            except Exception:
                continue
        if appended:
            logger.info("[ArkProvider] appended tool result messages: %d", appended)

    @staticmethod
    def _normalize_contexts(contexts: Optional[List[dict]]) -> List[dict]:
        """
        将 AstrBot 传入的 contexts 原样透传（已是 OpenAI 兼容结构）。
        """
        if not contexts: return []
        out = []
        for m in contexts:
            if isinstance(m, dict) and "role" in m:
                out.append(m)
        return out

    @staticmethod
    def _mk_user_message(prompt: str, image_urls: Optional[List[str]]) -> dict:
        if image_urls:
            parts = [{"type": "text", "text": str(prompt)}] if (prompt or "").strip() else []
            for url in image_urls:
                if url: parts.append({"type": "image_url", "image_url": {"url": str(url)}})
            return {"role": "user", "content": parts}
        return {"role": "user", "content": str(prompt)}

    def _build_messages(
        self,
        prompt: str,
        contexts: Optional[List[dict]],
        image_urls: Optional[List[str]],
        system_prompt: Optional[str],
        tool_calls_result: Any,
    ) -> List[dict]:
        """
        组装 messages：
        - 若存在 tool_calls_result：这是“工具回传后的续问”，只追加 role="tool" 结果，不再新建本轮 user 消息；
        - 否则：按常规追加本轮 user 消息（可含图片）。
        """
        messages: List[dict] = []
        if (system_prompt or "").strip():
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.extend(self._normalize_contexts(contexts))
        if tool_calls_result:
            self._append_tool_results(messages, tool_calls_result)
        else:
            messages.append(self._mk_user_message(prompt, image_urls))
        return messages

    # -------- Ark Chat Completions --------
    async def _chat_completions(self, model: str, messages: List[dict], tools: List[dict]) -> Tuple[dict, dict]:
        url = f"{self._base}/chat/completions"
        payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        async with (await self._sess()).post(url, json=payload, headers=self._headers()) as resp:
            data = await (resp.json() if resp.content_type == "application/json" else resp.text())
            if not isinstance(data, dict) or resp.status >= 400:
                raise RuntimeError(f"Ark chat_completions failed: {data}")
            chs = data.get("choices") or []
            if not chs:
                raise RuntimeError(f"Ark chat_completions empty choices: {data}")
            msg = (chs[0].get("message") or {})
            usage = data.get("usage") or {}
            return msg, usage

    @staticmethod
    def _extract_tool_calls(msg: dict) -> Tuple[List[dict], Optional[dict]]:
        """
        兼容：message.tool_calls（新）与 message.function_call（旧）。
        返回 (tool_calls, function_call_legacy)
        """
        tool_calls = msg.get("tool_calls") or []
        fn_legacy = msg.get("function_call")
        norm: List[dict] = []
        for c in tool_calls:
            fn = (c or {}).get("function") or {}
            name, arguments = fn.get("name"), fn.get("arguments")
            if name and arguments is not None:
                norm.append({
                    "id": c.get("id") or "",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                })
        return norm, fn_legacy

    # -------- usage / extra 填充 --------
    @staticmethod
    def _parse_cached_tokens(usage: dict) -> int:
        ptd = usage.get("prompt_tokens_details")
        if isinstance(ptd, dict):
            try: return int(ptd.get("cached_tokens", 0))
            except Exception: return 0
        return 0

    def _fill_usage_and_extra(self, resp: LLMResponse, usage: dict):
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = usage.get("total_tokens")
        try:
            total = int(total_tokens) if total_tokens is not None else prompt_tokens + completion_tokens
        except Exception:
            total = prompt_tokens + completion_tokens
        raw = SimpleNamespace()
        raw.usage = SimpleNamespace()
        raw.usage.prompt_tokens = prompt_tokens
        raw.usage.completion_tokens = completion_tokens
        raw.usage.total_tokens = total
        resp.raw_completion = raw

        cached = self._parse_cached_tokens(usage)
        extra = getattr(resp, "extra", {}) or {}
        extra.update({"ark_usage": usage, "ark_cached_tokens": cached})
        try:
            resp.extra = extra
        except Exception:
            # 若 LLMResponse 限制属性，则忽略（不影响 ToolLoop）
            pass

    # -------- AstrBot Provider 入口 --------
    async def text_chat(
        self,
        prompt: str,
        session_id: str = None,
        image_urls: List[str] = None,
        func_tool=None,
        contexts: List[dict] = None,
        system_prompt: str = None,
        tool_calls_result: Any = None,
        model: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        把 contexts / func_tool / tool_calls_result 映射为 OpenAI 兼容的 messages / tools，
        调用 Ark Chat Completions，返回带 tool_calls 的 LLMResponse。
        """
        model_name = (model or self.get_model() or self._model_default).strip()
        if not model_name:
            raise RuntimeError("ArkProvider: model not configured")
        if not self.get_current_key():
            raise RuntimeError("ArkProvider: API Key not configured")

        messages = self._build_messages(
            prompt=prompt,
            contexts=contexts,
            image_urls=image_urls,
            system_prompt=system_prompt,
            tool_calls_result=tool_calls_result,
        )

        tools = self._tools_from_manager(func_tool)
        logger.info("[ArkProvider] tools enabled=%s tools_num=%d",
                    bool(tools), len(tools))

        assistant_message, usage = await self._chat_completions(model_name, messages, tools)
        tool_calls, fn_legacy = self._extract_tool_calls(assistant_message)

        text = assistant_message.get("content") or ""
        resp = LLMResponse(role="assistant", completion_text=str(text))

        # 回填 usage / extra
        if usage: self._fill_usage_and_extra(resp, usage)

        # 写回工具调用结果（标准字段 + extra 兜底）
        if tool_calls:
            try:
                resp.tool_calls = tool_calls  # AstrBot ToolLoopAgent 会读取
            except Exception:
                extra = getattr(resp, "extra", {}) or {}
                extra["__tool_calls__"] = tool_calls
                try: resp.extra = extra
                except Exception: pass

        if fn_legacy and not tool_calls:
            try:
                resp.function_call = fn_legacy
            except Exception:
                pass

        logger.info("[ArkProvider] chat ok has_tool_calls=%s tool_calls_n=%d",
                    bool(tool_calls or fn_legacy), len(tool_calls))
        logger.info("[ArkProvider] usage: prompt=%s completion=%s total=%s cached_tokens=%s tool_calls=%s",
                    (usage or {}).get("prompt_tokens", 0),
                    (usage or {}).get("completion_tokens", 0),
                    (usage or {}).get("total_tokens", 0),
                    (getattr(resp, "extra", {}) or {}).get("ark_cached_tokens", 0),
                    len(tool_calls))

        return resp

    async def close(self):
        try:
            if self._session and not self._session.closed:
                await self._session.close()
        except Exception as e:
            logger.warning("[ArkProvider] close session error: %s", e)

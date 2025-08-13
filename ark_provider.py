# /opt/AstrBot/data/plugins/astrbot_plugin_provider_ark/ark_provider.py
import asyncio
import aiohttp
import json
from typing import List, Dict, Tuple, Optional, Any
from types import SimpleNamespace

from astrbot.api import logger
from astrbot.api.star import StarTools
from astrbot.core.provider.provider import Provider
from astrbot.core.provider.register import register_provider_adapter
from astrbot.core.provider.entities import LLMResponse, ProviderType, ToolCallsResult


ARK_BASE_DEFAULT = "https://ark.cn-beijing.volces.com/api/v3"


@register_provider_adapter(
    "ark_openai_compat",
    "Volcengine Ark (OpenAI-Compatible Chat Completions)",
    provider_type=ProviderType.CHAT_COMPLETION,
)
class ArkOpenAICompatProvider(Provider):
    """
    方舟 Chat Completions（OpenAI 兼容）Provider：
    - 统一使用 /chat/completions（非 context 版），以便传入 tools 与 role="tool" 消息。
    - 严格对接 AstrBot 的工具系统：func_tool -> tools；tool_calls_result -> role="tool"。
    - 返回时把 message.tool_calls 写回 LLMResponse，交给 ToolLoopAgent 继续工具环。
    参考：AstrBot Provider 协议(Function-calling 注释)，以及方舟 ChatCompletions 文档。 
    """

    def __init__(self, provider_config: dict, provider_settings: dict, default_persona=None):
        super().__init__(provider_config, provider_settings, default_persona)

        # ---- 基础配置 ----
        self._keys: List[str] = [(k or "").strip() for k in provider_config.get("key", []) if (k or "").strip()]
        self._key_idx: int = 0
        self._base: str = (provider_config.get("api_base") or ARK_BASE_DEFAULT).strip().rstrip("/")
        self._model_default: str = (provider_config.get("model") or provider_config.get("model_config", {}).get("model") or "").strip()
        self._timeout: int = int(provider_config.get("timeout", 60))

        # ---- HTTP 会话 ----
        self._session: Optional[aiohttp.ClientSession] = None

        masked = (self._keys[0][:6] + "..." + self._keys[0][-4:]) if self._keys else "EMPTY"
        logger.info(
            "[ArkProvider] init base=%s, model=%s, key=%s, timeout=%s",
            self._base, self._model_default, masked, self._timeout
        )

    # ===== Provider 能力 =====
    def get_models(self) -> List[str]:
        models = self.provider_config.get("models")
        return models if isinstance(models, list) and models else ([self._model_default] if self._model_default else [])

    def get_current_key(self) -> str:
        return self._keys[self._key_idx] if self._keys else ""

    def set_key(self, key: str):
        k = (key or "").strip()
        if k in self._keys:
            self._key_idx = self._keys.index(k)

    # ===== HTTP 基础 =====
    async def _sess(self) -> aiohttp.ClientSession:
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._timeout))
        return self._session

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.get_current_key().strip()}", "Content-Type": "application/json"}

    # ===== OpenAI 兼容消息构造 =====
    @staticmethod
    def _normalize_messages(
        contexts: Optional[List[dict]],
        prompt: str,
        system_prompt: Optional[str],
        tool_calls_result: Optional[ToolCallsResult | List[ToolCallsResult]],
    ) -> List[dict]:
        """
        根据 AstrBot 传入的上下文 + 当前 user prompt + 工具返回，拼装 OpenAI 兼容 messages。
        注意：AstrBot 的 contexts 尚未包含“当前这一轮的 user 消息”，因此这里追加。
        """
        messages: List[dict] = []

        # 历史上下文（已是 OpenAI 结构：role/content），保持原样
        if contexts:
            for m in contexts:
                role = m.get("role")
                if role in ("system", "user", "assistant", "tool"):
                    messages.append(m)

        # 本轮 system prompt（如果外部传入）
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})

        # 工具返回值（如果有），需要放在 user 前或后？
        # AstrBot 的 ToolLoopAgent 会把“上一步工具结果”通过 tool_calls_result 回传给下一次 LLM 请求。
        # 在 OpenAI 语义里，这是 assistant 产生 tool_calls 之后，工具以 role="tool" 回给模型。
        # 因而它应当出现在“这一次 user 消息”之前，作为模型续推的证据。
        if tool_calls_result:
            if isinstance(tool_calls_result, list):
                for tcr in tool_calls_result:
                    try:
                        messages.extend(tcr.to_openai_messages())
                    except Exception as e:
                        logger.warning("[ArkProvider] tool_calls_result item to_openai_messages failed: %s", e)
            else:
                try:
                    messages.extend(tool_calls_result.to_openai_messages())
                except Exception as e:
                    logger.warning("[ArkProvider] tool_calls_result to_openai_messages failed: %s", e)

        # 本轮 user
        messages.append({"role": "user", "content": str(prompt)})

        return messages

    @staticmethod
    def _tools_from_manager(func_tool: Any) -> List[dict]:
        """
        从 AstrBot 的 func_tool 管理器提取 OpenAI 兼容 tools：
        优先使用管理器的导出方法；失败则兜底从内部工具列表提取（容错，不抛错）。
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
                        tools = res
                        return tools
                except Exception as e:
                    logger.debug("[ArkProvider] func_tool.%s() failed: %s", attr, e)

        # 兜底：尝试读取内部注册的本地/MCP 工具
        try:
            iterable = None
            if hasattr(func_tool, "tools"):
                iterable = getattr(func_tool, "tools")
            elif hasattr(func_tool, "get_tools"):
                iterable = getattr(func_tool, "get_tools")()

            if iterable:
                for t in iterable:
                    # 兼容 dataclass FuncTool(name, parameters, description, active, origin=...)
                    name = getattr(t, "name", None)
                    params = getattr(t, "parameters", None)
                    desc = getattr(t, "description", "")
                    active = getattr(t, "active", True)
                    if name and isinstance(params, dict) and active:
                        tools.append({
                            "type": "function",
                            "function": {"name": name, "description": desc or "", "parameters": params}
                        })
        except Exception as e:
            logger.debug("[ArkProvider] fallback build tools failed: %s", e)

        return tools

    @staticmethod
    def _parse_usage(data: dict) -> Tuple[int, int, int, int]:
        usage = data.get("usage", {}) or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        total = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
        cached = 0
        try:
            cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
        except Exception:
            pass
        return prompt_tokens, completion_tokens, total, cached

    @staticmethod
    def _extract_tool_calls(msg: dict) -> Tuple[List[dict], Optional[dict]]:
        """
        兼容两种返回：message.tool_calls（新）与 message.function_call（旧）。
        """
        tool_calls = msg.get("tool_calls") or []
        function_call = msg.get("function_call")  # 旧字段
        # 统一校验
        norm_calls: List[dict] = []
        for c in tool_calls:
            fn = (c or {}).get("function") or {}
            name, arguments = fn.get("name"), fn.get("arguments")
            if name and arguments is not None:
                norm_calls.append({
                    "id": c.get("id") or "",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                })
        return norm_calls, function_call

    async def _chat_completions(
        self, model: str, messages: List[dict], tools: List[dict]
    ) -> Tuple[dict, List[dict], Optional[dict]]:
        """
        调用 Ark Chat Completions，返回 (message, tool_calls, function_call_legacy)。
        """
        url = f"{self._base}/chat/completions"  # 方舟 ChatCompletions 端点（OpenAI 兼容）
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"  # 允许模型按需触发工具

        async with (await self._sess()).post(url, json=payload, headers=self._headers()) as resp:
            data = await (resp.json() if resp.content_type == "application/json" else resp.text())
            if not isinstance(data, dict) or resp.status >= 400:
                raise RuntimeError(f"Ark chat_completions failed: {data}")

            chs = (data.get("choices") or [])
            if not chs:
                raise RuntimeError(f"Ark chat_completions empty choices: {data}")
            msg = (chs[0].get("message") or {})
            tool_calls, fn_legacy = self._extract_tool_calls(msg)
            return msg, tool_calls, fn_legacy

    # ===== 主入口 =====
    async def text_chat(
        self,
        prompt: str,
        session_id: str = None,
        image_urls: List[str] = None,
        func_tool=None,
        contexts: List[dict] = None,
        system_prompt: str = None,
        tool_calls_result: ToolCallsResult | List[ToolCallsResult] = None,
        model: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        AstrBot 调用的主入口：拼装 OpenAI 兼容消息，注入 tools，调用方舟 ChatCompletions。
        """
        model_name = (model or self.get_model() or self._model_default).strip()
        if not model_name:
            # 方舟文档示例也使用模型名直填（如 doubao-pro-32k-240615），或使用 ep-...。参见官方 ChatCompletions 文档。
            raise RuntimeError("ArkProvider: model not configured")
        if not self.get_current_key():
            raise RuntimeError("ArkProvider: API Key not configured")

        # 1) 组装 messages（历史 + tool 返回 + system + 本轮 user）
        messages = self._normalize_messages(contexts, prompt, system_prompt, tool_calls_result)

        # 2) 注入 tools
        tools = self._tools_from_manager(func_tool)
        tool_names = [t.get("function", {}).get("name") for t in tools if isinstance(t, dict)]
        logger.info("[ArkProvider] tools enabled=%s tools_num=%d tools=%s",
                    bool(tools), len(tools), tool_names[:5])

        # 3) 调用 Ark ChatCompletions
        msg, tool_calls, fn_legacy = await self._chat_completions(model_name, messages, tools)

        content = msg.get("content") or ""
        pt, ct, tt, cached = 0, 0, 0, 0  # usage 统计若需要，可二次请求 data，但这里先在 _build_llm_response 填充
        logger.info("[ArkProvider] chat ok has_tool_calls=%s tool_calls_n=%d legacy_fn=%s",
                    bool(tool_calls or fn_legacy), len(tool_calls), bool(fn_legacy))

        # 4) 构建 LLMResponse（含 tool_calls 回传）
        resp = self._build_llm_response(content, usage=None)  # usage 稍后补充
        # ——关键：把 tool_calls 写回，交给 ToolLoopAgent 继续跑工具环
        try:
            resp.tool_calls = tool_calls  # 标准字段（与 OpenAI 结构一致）
        except Exception:
            # 若 LLMResponse 限制了属性，写到 extra 做兜底（仍建议升级 AstrBot 至较新版本）
            if getattr(resp, "__dict__", None) is not None:
                extra = getattr(resp, "extra", {}) or {}
                extra["__tool_calls__"] = tool_calls
                resp.extra = extra

        # 兼容旧式 function_call
        if fn_legacy and not tool_calls:
            try:
                resp.function_call = fn_legacy
            except Exception:
                pass

        # 5) usage（如果需要展示/统计，可选：读取 data.usage）
        #   有些后端会返回 usage，也可能不返回。为健壮，这里只在存在时填充。
        #   Ark 文档示例展示了 usage 以及 prompt_tokens_details.cached_tokens。见引文。
        #   由于本函数只返回 message，此处无法直接拿到；你如果需要，可在 _chat_completions 返回 data 一并解析。
        #   这里保留接口：从 kwargs["__raw_usage"] 读取（留给你后续优化）。
        usage = kwargs.get("__raw_usage")
        if usage and isinstance(usage, dict):
            self._fill_usage(resp, usage)

        logger.info("[ArkProvider] usage: cached_tokens=%s tool_calls=%s",
                    getattr(resp, "extra", {}).get("ark_cached_tokens", 0) if getattr(resp, "extra", None) else 0,
                    len(tool_calls))

        return resp

    # ===== 组装 LLMResponse / usage 回填 =====
    def _build_llm_response(self, text: str, usage: Optional[dict]) -> LLMResponse:
        resp = LLMResponse(role="assistant", completion_text=str(text or ""))

        # 回填 usage（尽可能健壮）
        if usage and isinstance(usage, dict):
            self._fill_usage(resp, usage)

        return resp

    def _fill_usage(self, resp: LLMResponse, usage: dict):
        # 兼容 ark usage 结构
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

        # 记录 cached_tokens（若提供）
        cached = 0
        try:
            cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
        except Exception:
            pass

        # 写到 extra
        if getattr(resp, "__dict__", None) is not None:
            extra = getattr(resp, "extra", {}) or {}
            extra.update({"ark_usage": usage, "ark_cached_tokens": cached})
            resp.extra = extra

    async def close(self):
        try:
            if self._session and not self._session.closed:
                await self._session.close()
        except Exception as e:
            logger.warning("[ArkProvider] close session error: %s", e)

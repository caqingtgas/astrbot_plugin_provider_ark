# /opt/AstrBot/data/plugins/astrbot_plugin_provider_ark/ark_provider.py
# 说明：
# - 该实现基于 Ark Chat Completions（OpenAI 兼容），支持工具调用（Tool Calling）
# - 不再使用 /context/* API；会直接使用 AstrBot 提供的 contexts 作为“对话历史”
# - 与 AstrBot Provider 约定对齐：func_tool -> tools，tool_calls_result -> role="tool" 消息

import asyncio
import aiohttp
import json
from typing import List, Dict, Tuple, Optional, Any
from types import SimpleNamespace
from pathlib import Path

from astrbot.api import logger
from astrbot.api.star import StarTools
from astrbot.core.provider.provider import Provider
from astrbot.core.provider.register import register_provider_adapter
from astrbot.core.provider.entities import LLMResponse, ProviderType


# 兼容旧配置名：沿用 "ark_context"（无需改 config），但内部已切换到 Chat Completions
@register_provider_adapter(
    "ark_context",
    "Volcengine Ark (Chat Completions + Tools)",
    provider_type=ProviderType.CHAT_COMPLETION,
)
class ArkChatProvider(Provider):
    """
    基于 Ark Chat Completions 的 AstrBot Provider（支持 Tool Calling）。
    - 接口：POST {api_base}/chat/completions
    - 工具：func_tool -> tools，tool_calls_result -> role="tool" messages
    - 多模态：image_urls -> content list [{type:"text"},{type:"image_url"}]
    """

    def __init__(self, provider_config: dict, provider_settings: dict, default_persona=None):
        super().__init__(provider_config, provider_settings, default_persona)

        # ---- 基础配置 ----
        self._keys: List[str] = [(k or "").strip() for k in provider_config.get("key", []) if (k or "").strip()]
        self._key_idx: int = 0
        self._base: str = (provider_config.get("api_base") or "https://ark.cn-beijing.volces.com/api/v3").strip().rstrip("/")
        # 模型可在 provider_config["model"] 或 provider_config["model_config"]["model"] 中设置
        self._model_default: str = (provider_config.get("model") or provider_config.get("model_config", {}).get("model") or "").strip()
        self._timeout: int = int(provider_config.get("timeout", 60))

        # ---- HTTP 会话 ----
        self._session: Optional[aiohttp.ClientSession] = None

        # ---- （可选）本地落盘：仅用于日志与后续可能的持久化需要，实际不做 context 映射 ----
        data_dir = StarTools.get_data_dir("astrbot_plugin_provider_ark")
        self._data_dir: Path = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        masked = (self._keys[0][:6] + "..." + self._keys[0][-4:]) if self._keys else "EMPTY"
        logger.info(
            "[ArkProvider] init base=%s, model=%s, key=%s, timeout=%s",
            self._base, self._model_default, masked, self._timeout
        )

    # ===== Provider 能力 =====
    def get_models(self) -> List[str]:
        """
        返回可用模型列表（来自 provider_config.models 或默认单模型）。
        """
        models = self.provider_config.get("models")
        return models if isinstance(models, list) and models else ([self._model_default] if self._model_default else [])

    def get_current_key(self) -> str:
        """获取当前使用的 API Key（支持多 Key 轮换）。"""
        return self._keys[self._key_idx] if self._keys else ""

    def set_key(self, key: str):
        """设置当前使用的 API Key。"""
        k = (key or "").strip()
        if k in self._keys:
            self._key_idx = self._keys.index(k)

    # ===== Ark HTTP =====
    async def _sess(self) -> aiohttp.ClientSession:
        """获取（或创建）共享的 HTTP 会话。"""
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._timeout))
        return self._session

    def _headers(self) -> dict:
        """构造请求头。"""
        return {
            "Authorization": f"Bearer {self.get_current_key().strip()}",
            "Content-Type": "application/json",
        }

    # ======== 构造 Chat Completions Payload ========
    @staticmethod
    def _convert_tools(func_tool: Any) -> List[dict]:
        """
        将 AstrBot 的 func_tool（可能是单个或列表，且对象/字典皆可）转换为 OpenAI 兼容的 tools。
        """
        if not func_tool:
            return []
        tools = []
        items = func_tool if isinstance(func_tool, (list, tuple)) else [func_tool]
        for t in items:
            # 兼容对象/字典两种取值方式
            name = getattr(t, "name", None) if not isinstance(t, dict) else t.get("name")
            desc = getattr(t, "description", "") if not isinstance(t, dict) else t.get("description", "")
            params = getattr(t, "parameters", {}) if not isinstance(t, dict) else t.get("parameters", {})
            if not name:
                continue
            tools.append({
                "type": "function",
                "function": {
                    "name": str(name)[:64],  # 名称长度保险
                    "description": str(desc) if desc is not None else "",
                    "parameters": params if isinstance(params, (dict, list)) else {},
                }
            })
        return tools

    @staticmethod
    def _mk_user_message(prompt: str, image_urls: Optional[List[str]]) -> dict:
        """
        构造当前轮的用户消息。若包含图片，则采用多段 content。
        """
        if image_urls:
            parts = [{"type": "text", "text": str(prompt)}] if (prompt or "").strip() else []
            for url in image_urls:
                if not url:
                    continue
                parts.append({"type": "image_url", "image_url": {"url": str(url)}})
            return {"role": "user", "content": parts}
        else:
            return {"role": "user", "content": str(prompt)}

    @staticmethod
    def _append_tool_results(messages: List[dict], tool_calls_result: Any) -> None:
        """
        将 AstrBot 的工具执行结果转为 role="tool" 的消息，追加到 messages。
        兼容单个/列表、对象/字典。
        """
        if not tool_calls_result:
            return
        items = tool_calls_result if isinstance(tool_calls_result, (list, tuple)) else [tool_calls_result]
        for r in items:
            # 兼容对象/字典两种形态
            tool_call_id = getattr(r, "tool_call_id", None) if not isinstance(r, dict) else r.get("tool_call_id") or r.get("id")
            name = getattr(r, "name", None) if not isinstance(r, dict) else r.get("name")
            # result 字段可能名为 result / content / output ...
            content = None
            if isinstance(r, dict):
                content = r.get("result", None) or r.get("content", None) or r.get("output", None)
            else:
                content = getattr(r, "result", None) or getattr(r, "content", None) or getattr(r, "output", None)
            msg = {"role": "tool", "content": str(content) if content is not None else ""}
            if tool_call_id:
                msg["tool_call_id"] = str(tool_call_id)
            if name:
                msg["name"] = str(name)
            messages.append(msg)

    @staticmethod
    def _normalize_contexts(contexts: Optional[List[dict]]) -> List[dict]:
        """
        将 AstrBot 传入的 contexts 直接透传为 OpenAI 兼容 messages（AstrBot 已按该格式维护历史）。
        若为空，返回 []。
        """
        if not contexts:
            return []
        # 为健壮性，确保每项至少包含 role 与 content 字段（若没有则跳过）
        out = []
        for m in contexts:
            if isinstance(m, dict) and "role" in m:
                out.append(m)
        return out

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
        - 若存在 tool_calls_result：说明这是“工具调用后的下一跳”，不再追加新的 user 消息，仅把工具结果追加到历史末尾
        - 否则：追加本轮 user 消息（可含图片）
        """
        messages: List[dict] = []
        if (system_prompt or "").strip():
            messages.append({"role": "system", "content": str(system_prompt)})

        # 历史消息（AstrBot 已是 OpenAI 兼容结构：包含 user/assistant/tool/tool_calls 等）
        messages.extend(self._normalize_contexts(contexts))

        if tool_calls_result:
            # 工具调用后的“回传结果”轮：只追加 tool 结果，不新建用户问句
            self._append_tool_results(messages, tool_calls_result)
        else:
            # 正常一轮问答：把当前用户问句放到末尾
            messages.append(self._mk_user_message(prompt, image_urls))

        return messages

    async def _chat_completions(
        self,
        model: str,
        messages: List[dict],
        func_tool: Any = None,
    ) -> Tuple[dict, dict]:
        """
        调用 Ark Chat Completions，返回 (message, usage)：
        - message: choices[0].message
        - usage: response.usage
        """
        url = f"{self._base}/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        tools = self._convert_tools(func_tool)
        if tools:
            payload["tools"] = tools
            # 明确开启自动工具调用（大多数 OpenAI 兼容实现默认即为 auto；这里显式设置更稳）
            payload["tool_choice"] = "auto"

        async with (await self._sess()).post(url, json=payload, headers=self._headers()) as resp:
            data = await (resp.json() if resp.content_type == "application/json" else resp.text())
            if not isinstance(data, dict) or resp.status >= 400:
                raise RuntimeError(f"Ark chat_completions failed: {data}")
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"Ark chat_completions empty choices: {data}")
            msg = choices[0].get("message") or {}
            usage = data.get("usage") or {}
            return msg, usage

    # ===== 解析 usage.cached_tokens =====
    @staticmethod
    def _parse_cached_tokens(usage: dict) -> int:
        """
        从 usage.prompt_tokens_details.cached_tokens 解析缓存命中数；异常时返回 0。
        """
        ptd = usage.get("prompt_tokens_details")
        if isinstance(ptd, dict):
            try:
                return int(ptd.get("cached_tokens", 0))
            except (TypeError, ValueError):
                logger.warning("[ArkProvider] cached_tokens parse error: %r", ptd.get("cached_tokens", None))
        return 0

    # ===== 组装 LLMResponse =====
    def _build_llm_response(self, message: dict, usage: dict) -> LLMResponse:
        """
        将 Ark/OpenAI 的 message+usage 转为 AstrBot 的 LLMResponse：
        - 提取 content
        - 提取 tool_calls（若有）
        - 回填 usage 到 raw_completion
        """
        text = message.get("content") or ""
        resp = LLMResponse(role="assistant", completion_text=str(text))

        # 解析 tool_calls
        tool_calls = message.get("tool_calls") or []
        parsed_tool_calls = []
        for tc in tool_calls:
            try:
                fn = (tc or {}).get("function") or {}
                name = fn.get("name")
                args_raw = fn.get("arguments")
                # Ark/OpenAI 返回的 arguments 正常为 str；若是 dict，也兼容
                if isinstance(args_raw, (dict, list)):
                    args = args_raw
                else:
                    try:
                        args = json.loads(args_raw) if args_raw else {}
                    except json.JSONDecodeError:
                        args = {"_raw": args_raw}
                parsed_tool_calls.append({
                    "id": tc.get("id"),
                    "name": name,
                    "arguments": args,
                })
            except Exception as e:
                logger.warning("[ArkProvider] parse tool_call error: %s | raw=%r", e, tc)

        if parsed_tool_calls:
            # AstrBot ToolLoopAgent 会读取该字段以决定是否执行工具
            try:
                resp.tool_calls = parsed_tool_calls  # type: ignore[attr-defined]
            except Exception:
                # 如果 LLMResponse 未显式声明该属性，降级放进 extra
                pass

        # usage 回填
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = usage.get("total_tokens")
        try:
            total = int(total_tokens) if total_tokens is not None else prompt_tokens + completion_tokens
        except (ValueError, TypeError):
            total = prompt_tokens + completion_tokens

        raw = SimpleNamespace()
        raw.usage = SimpleNamespace()
        raw.usage.prompt_tokens = prompt_tokens
        raw.usage.completion_tokens = completion_tokens
        raw.usage.total_tokens = total
        resp.raw_completion = raw

        # extra：包含 ark_context（无）、cached_tokens、原始 tool_calls
        extra_payload = {
            "ark_usage": usage,
            "ark_cached_tokens": self._parse_cached_tokens(usage),
            "ark_tool_calls": tool_calls,
        }

        # 尽量不以异常作流程控制；判定可写性后再赋值
        can_set = getattr(resp, "__dict__", None) is not None or (
            isinstance(getattr(type(resp), "__slots__", ()), (list, tuple, set)) and "extra" in getattr(type(resp), "__slots__", ())
        )
        if can_set:
            resp.extra = extra_payload  # type: ignore[attr-defined]
        else:
            rc = getattr(resp, "raw_completion", None)
            if rc is not None and getattr(rc, "__dict__", None) is not None:
                rc.extra = extra_payload  # type: ignore[attr-defined]

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
        AstrBot 调用的主入口：将 contexts/func_tool/tool_calls_result 映射为 OpenAI 兼容消息与 tools，
        通过 Ark Chat Completions 获取回复/工具调用请求。
        """
        model_name = (model or self.get_model() or self._model_default).strip()
        if not model_name:
            raise RuntimeError("ArkProvider: model not configured (use endpoint id: ep-...)")
        if not self.get_current_key():
            raise RuntimeError("ArkProvider: API Key not configured")

        messages = self._build_messages(
            prompt=prompt,
            contexts=contexts,
            image_urls=image_urls,
            system_prompt=system_prompt,
            tool_calls_result=tool_calls_result,
        )

        message, usage = await self._chat_completions(
            model=model_name,
            messages=messages,
            func_tool=func_tool,
        )

        resp = self._build_llm_response(message, usage)

        # 便于确认是否触发工具
        has_tool_calls = bool((getattr(resp, "tool_calls", None) or []))
        logger.info(
            "[ArkProvider] chat ok has_tool_calls=%s", has_tool_calls
        )
        logger.info(
            "[ArkProvider] usage: prompt=%s completion=%s total=%s cached_tokens=%s tool_calls=%s",
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            usage.get("total_tokens", 0),
            (getattr(resp, "extra", {}) or {}).get("ark_cached_tokens", 0),
            len(getattr(resp, "tool_calls", []) or []),
        )
        return resp

    async def close(self):
        """关闭 HTTP 会话。"""
        try:
            if self._session and not self._session.closed:
                await self._session.close()
        except Exception as e:
            logger.warning("[ArkProvider] close session error: %s", e)

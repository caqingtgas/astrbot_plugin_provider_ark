# /opt/AstrBot/data/plugins/astrbot_plugin_provider_ark/main.py
from importlib import import_module

from astrbot.api.star import Star, Context  # 不再使用 @register
from astrbot.api import logger


class ArkProviderLoader(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        # 通过显式动态导入，触发 ark_provider 中 register_provider_adapter 的副作用注册
        mod = import_module(f"{__package__}.ark_provider")
        logger.info("[ArkProviderLoader] 已加载 Ark Provider 模块: %s", getattr(mod, "__name__", "unknown"))

    async def terminate(self):
        # 如后续需要在停用时清理资源，可在这里调用 provider 的 close()
        pass

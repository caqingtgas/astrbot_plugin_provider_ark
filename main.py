# /opt/AstrBot/data/plugins/astrbot_plugin_provider_ark/main.py
from astrbot.api.star import Star, Context  # 不再使用 @register
from astrbot.api import logger

# 确保 ark_provider.py 被导入，从而完成 Provider 的注册（有副作用）
from . import ark_provider  # noqa: F401

class ArkProviderLoader(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        logger.info("[ArkProviderLoader] 已加载 Ark Provider 模块")

    async def terminate(self):
        # 如后续需要在停用时清理资源，可在这里调用 provider 的 close()
        pass

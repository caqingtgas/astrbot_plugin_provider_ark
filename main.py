from astrbot.api.star import register, Star
from astrbot.api.star import Context
import logging

# 确保 ark_provider.py 会被加载并注册 Provider
from . import ark_provider

@register(
    name="astrbot_provider_ark_loader",
    desc="加载火山方舟 Ark Provider",
    author="you",
    version="0.1.0",
    repo=""
)
class ArkProviderLoader(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        logging.getLogger().info("[ArkProviderLoader] 已加载 Ark Provider 模块")

    async def terminate(self):
        pass

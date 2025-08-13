from astrbot.api.star import Star, Context
from astrbot.api import logger

# 导入 provider 模块以触发其内部的 register_provider_adapter 装饰器完成注册
from . import ark_provider  # noqa: F401


class ArkProviderLoader(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        logger.info("[ArkProviderLoader] 已加载 Ark Provider 模块")

    async def terminate(self):
        # 如后续需要在停用时清理资源，可在这里调用 provider 的 close()
        pass

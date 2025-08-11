# Ark Context Provider for AstrBot

**在 AstrBot 使用火山方舟(Volcengine Ark)上下文缓存API**

## 简介
ChatGPT5.0全新力作，之前用GPT4.5和Gemini2.5Pro都没能解决。
目前在AstrBot中没有火山方舟的模板预设，添加模型只能通过openai的接口，但用这种方法进行部署的话，无法使用上下文缓存，因为火山方舟专门把上下文缓存做成了单独的api，还分成俩，一个创建一个会话，我也不知道为啥这么设计。
总之，用这个插件就可以把火山方舟的上下文缓存API集成到AstrBot中。

- 上下文缓存实现：自动使用 Ark `/context/create` + `/context/chat` 实现会话
- 融合集成：将ark_context注册为了provider，但不能通过Webui面板直接添加，除此以外应该和任何自带配置都是相同的
- 消耗token统计：控制台中输出token消耗情况， `cached_tokens` 为命中缓存数

## 如何使用
1. 去https://console.volcengine.com/ark/region:ark+cn-beijing/endpoint创建对应模型的接入点，得到形如ep-xxxx-xxxx的ID
2. 在配置文件中添加供应商：
   ```json
    {
    "id": "任意",
    "enable": true,
    "type": "ark_context",
    "model_config": {
    "model": "形如ep-xxxx-xxxx的ID"
    },
    "model": "形如ep-xxxx-xxxx的ID",
    "key": [
    "你的APIkey"
    ],
    "api_base": "https://ark.cn-beijing.volces.com/api/v3",
    "ttl": 范围3600到604800，即1小时到7天
    }
   ```
3. 重启 AstrBot


# 智光实验室：光通信 AI 仿真平台（云端部署版）

本版本用于 Streamlit Community Cloud / Hugging Face Spaces / 服务器部署。学生和老师通过浏览器访问链接即可使用，无需本地安装 Python 或第三方库。

## 主要改动

- 默认使用“规则库模式”，不依赖本地 LM Studio，适合云端公开部署。
- 保留“在线 API 模式”，可通过 Streamlit Secrets 配置大模型。
- 保留“本地 LM Studio 模式”，仅用于本地增强版。
- IFTS 高保真模块保持可选导入；云端没有 IFTS 文件时不会影响教学模式运行。
- 删除外链图标依赖，避免国内网络加载失败。
- 代理设置改为环境变量控制，避免云端联网异常。

## Streamlit Cloud 部署

1. 将本文件夹上传到 GitHub 仓库。
2. 登录 Streamlit Community Cloud。
3. 选择仓库、分支和入口文件：`app.py`。
4. 部署完成后会得到 `https://xxxx.streamlit.app` 链接。

## 可选：配置在线大模型

在 Streamlit Cloud 的 Secrets 中加入：

```toml
OPENAI_API_KEY = "你的 API Key"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4o-mini"
```

也可以把 `OPENAI_BASE_URL` 改成其他兼容 OpenAI SDK 的服务地址。

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

如果需要本地 LM Studio，请在侧边栏选择“本地 LM Studio 模式”，并启动 LM Studio Server。

# Jeffmind Agent Framework

这是一个基于 Python 异步架构构建的高级 AI Agent 框架。它的底层设计哲学深受 TypeScript 高级架构（如 Pi）的启发，旨在解决大模型流式输出、工具调用（Tool Use）以及多路并发消费的复杂工程问题。

## 🎯 核心架构亮点

本框架严格遵循 **“机制与策略分离”** 的原则，分为以下三大核心层：

### 1. Provider 层 (AI 层)
坚守底线，只做最纯粹的翻译官。
- **统一接口**：将 OpenAI、DeepSeek、Ollama 等各家厂商的异构 API，强力抹平为标准的 `StreamEvent` 异步生成器流。
- **兼容性降级 (Compat)**：内置强大的消息转换器，解决部分模型不支持 `system` prompt、不支持原生 ToolCall（采用特殊 XML 格式）等兼容性问题。
- **孤儿 ToolCall 补偿**：自动处理模型输出格式错误时的工具结果孤儿问题。
- **鉴权链 (Auth Resolution)**：支持 `请求级覆盖 > 持久化凭证 > 环境变量` 的优雅降级解析机制。

### 2. Runtime 层 (运行时层)
解决 Python 原生 `AsyncGenerator` 单次消费的痛点。
- **StreamBroadcaster (EventBus)**：利用 `asyncio.Queue` 实现的并发事件总线。能够将底层的模型推理流无损、无阻塞地**广播**给前端 UI、审计日志、Trace 分析器等多个并发消费者，且自带完美的背压 (Backpressure) 控制。

### 3. Agent 层 (策略执行层)
- **解耦的 Tool 设计**：区分了 Provider 层的“纯 JSON Schema 工具”和 Agent 层的“包含真实 `fn` 的可执行工具”。
- **ReAct 引擎**：内置了一个健壮的 ReAct (Reasoning + Acting) 循环引擎。引擎自动捕获 `ToolCall` 事件，并发执行本地 Python 函数，并将结果组装回 Context，直到达成目标。

## 📁 目录结构

```text
agent/
├── agent/                  # Agent 核心逻辑
│   ├── react.py            # ReAct 循环引擎
│   └── eventbus.py         # 基于队列的 Pub/Sub 流分发器
├── provider/               # 大模型底层接入层
│   ├── api/                # HTTP 客户端与流解析 (如 OpenAI Completions)
│   ├── auth/               # 鉴权解析链路与凭证存储
│   ├── providers/          # 具体厂商接入 (Ollama, DeepSeek 等)
│   ├── models.py           # Stream 与 Complete 核心入口
│   └── types.py            # 全局统一类型定义 (Message, StreamEvent)
├── tests/                  # 完备的单元测试与集成测试 (75+ Test Cases)
├── example_react.py        # ReAct Agent 的终端交互演示
└── example_eventbus.py     # EventBus 多路并发消费演示
```

## 🚀 快速上手

### 1. 环境准备

本项目使用 [uv](https://github.com/astral-sh/uv) 进行极速包管理。

```bash
# 确保安装了 uv
# 安装依赖并同步环境
uv sync
```

### 2. 环境变量配置

复制环境变量示例文件：

```bash
cp .env.example .env
```
然后在 `.env` 中填入你的 API Key，例如 `DEEPSEEK_API_KEY`。如果不填，你可以完全依靠本地的 `Ollama` 运行。

### 3. 运行 ReAct Agent 演示

这个 Demo 展示了 Agent 如何根据你的问题，自主选择工具（查时间、算术计算、格式转换等）并执行。

```bash
uv run python example_react.py
```

### 4. 运行 EventBus 高并发分发演示

这个 Demo 展示了底层模型流是如何被独立分发给 UI 渲染、日志记录和后台审计分析三个并发协程的。

```bash
uv run python example_eventbus.py
```

## 🧪 运行测试

本项目极其重视底层确定性，包含了 75+ 个覆盖边缘情况的测试用例。

```bash
uv run pytest
```

## 📅 下一步计划 (Roadmap)

- [ ] **模型注册表持久化**: 将 `builtin_models` 剥离为 `models.json`，实现配置驱动的动态加载。
- [ ] **凭证持久化存储**: 实现 `FileCredentialStore` 或系统钥匙串集成。
- [ ] **Session & Memory**: 为 Agent 循环增加长短期记忆机制。
- [ ] **UI 集成**: 暴露标准的 WebSocket/SSE 接口供前端调用。

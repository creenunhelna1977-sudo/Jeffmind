"""
观察 Tool Call 全流程的 Event Stream 流转。

用假的 tool 定义（get_weather），真实调用 DeepSeek，
打印每一步 StreamEvent 的类型、内容和结构变化。
"""
import asyncio
import json
from dotenv import load_dotenv

from provider.providers.all import builtin_models
from provider.types import (
    Context,
    UserMessage,
    Tool,
    ToolResultMessage,
    TextContent,
)

from rich import print
from rich.panel import Panel
from rich.table import Table

load_dotenv()

# ---------------------------------------------------------------------------
# 假工具：获取天气
# ---------------------------------------------------------------------------
FAKE_WEATHER_TOOL = Tool(
    name="get_weather",
    description="Get the current weather in a given location. Returns temperature and conditions.",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
            }
        },
        "required": ["location"]
    }
)


def print_event_header(event_type: str, emoji: str, color: str):
    """打印统一的事件头"""
    print()
    print(f"[{color}]{emoji} ── [Event] {event_type} ──[/{color}]")


async def round_one(models, model):
    """
    第一轮：发送带 tool 定义的消息，观察模型如何"决定调用工具"。
    关注 toolcall_start → toolcall_delta → toolcall_end 事件序列。
    """
    context = Context(
        system_prompt="You are a helpful assistant. When asked about weather, use the get_weather tool.",
        messages=[UserMessage(content="What's the weather like in Tokyo today?")],
        tools=[FAKE_WEATHER_TOOL],
    )

    print()
    print(Panel.fit(
        f"[bold]Round 1[/bold] — 发送带 Tool 定义的消息\n"
        f"Tool: {FAKE_WEATHER_TOOL.name}\n"
        f"User: {context.messages[0].content}",
        title="📤 Request"
    ))

    assistant_message = None
    event_index = 0

    async for event in models.stream(model, context):
        event_index += 1

        # ── start ──
        if event.type == "start":
            print_event_header("start", "🚀", "bold cyan")
            print(f"  partial.response_id: {event.partial.response_id or '(未设置)'}")
            print(f"  partial.model: {event.partial.model}")
            print(f"  partial.content: {event.partial.content}")

        # ── text_delta ──
        elif event.type == "text_delta":
            print_event_header("text_delta", "📝", "green")
            print(f"  content_index: {event.content_index}")
            print(f"  delta: [green]{repr(event.delta)}[/green]")

        # ── toolcall_start ──
        elif event.type == "toolcall_start":
            print_event_header("toolcall_start", "🔧", "bold yellow")
            print(f"  content_index: {event.content_index}")
            print(f"  partial.content 中当前块数: {len(event.partial.content)}")

        # ── toolcall_delta ──
        elif event.type == "toolcall_delta":
            print_event_header("toolcall_delta", "📐", "yellow")
            print(f"  content_index: {event.content_index}")
            print(f"  delta: [yellow]{repr(event.delta)}[/yellow]")

        # ── toolcall_end ──
        elif event.type == "toolcall_end":
            print_event_header("toolcall_end", "✅", "bold green")
            print(f"  content_index: {event.content_index}")
            print(f"  tool_call.id: {event.tool_call.id}")
            print(f"  tool_call.name: {event.tool_call.name}")
            print(f"  tool_call.arguments: {json.dumps(event.tool_call.arguments, indent=2)}")

        # ── done ──
        elif event.type == "done":
            print_event_header("done", "🏁", "bold blue")
            print(f"  reason: {event.reason}")
            print(f"  message.stop_reason: {event.message.stop_reason}")
            print(f"  message.model: {event.message.model}")
            print(f"  message.response_id: {event.message.response_id}")
            print(f"  message.usage.input: {event.message.usage.input}")
            print(f"  message.usage.output: {event.message.usage.output}")
            print(f"  message.usage.total_tokens: {event.message.usage.total_tokens}")
            # 列举 message.content 中每种 block 的类型
            print(f"  message.content blocks ({len(event.message.content)}):")
            for i, block in enumerate(event.message.content):
                print(f"    [{i}] type={block.type}", end="")
                if hasattr(block, "text"):
                    print(f" text={repr(block.text[:80])}")
                elif hasattr(block, "name"):
                    print(f" name={block.name} id={block.id}")
                else:
                    print()

            assistant_message = event.message

        # ── error ──
        elif event.type == "error":
            print_event_header("error", "❌", "bold red")
            print(f"  reason: {event.reason}")
            print(f"  error.error_message: {event.error.error_message}")

    print()
    print(f"[dim]Round 1 共收到 {event_index} 个事件[/dim]")
    return context, assistant_message


async def round_two(models, model, context, assistant_message):
    """
    第二轮：把假工具的执行结果返回给模型，观察模型如何"基于工具结果做最终回复"。
    关注 text_delta 事件序列（模型用自然语言总结工具返回的数据）。
    """
    if not assistant_message or assistant_message.stop_reason != "toolUse":
        print()
        print("[bold red]模型没有调用工具，跳过 Round 2[/bold red]")
        return

    # 提取 tool calls
    from provider.types import ToolCall
    tool_calls = [b for b in assistant_message.content if isinstance(b, ToolCall)]
    if not tool_calls:
        print("[bold red]AssistantMessage 中没有 ToolCall block，跳过 Round 2[/bold red]")
        return

    # 把 assistant_message 追加到历史
    context.messages.append(assistant_message)

    for tc in tool_calls:
        fake_result = f"It's 25°C (77°F) in {tc.arguments.get('location', 'unknown')} right now, with clear skies and a light breeze."
        print()
        print(Panel.fit(
            f"Tool: {tc.name}\nArgs: {json.dumps(tc.arguments)}\n→ Fake Result: {fake_result}",
            title="🔧 模拟工具执行"
        ))
        context.messages.append(ToolResultMessage(
            tool_call_id=tc.id,
            tool_name=tc.name,
            content=[TextContent(text=fake_result)],
        ))

    print()
    print(Panel.fit("[bold]Round 2[/bold] — 把工具结果发回模型", title="📤 Request"))

    event_index = 0

    async for event in models.stream(model, context):
        event_index += 1

        if event.type == "start":
            print_event_header("start", "🚀", "bold cyan")

        elif event.type == "text_delta":
            print_event_header("text_delta", "📝", "green")
            print(f"  content_index: {event.content_index}")
            print(f"  delta: [green]{repr(event.delta)}[/green]")

        elif event.type == "toolcall_start":
            print_event_header("toolcall_start", "🔧", "bold yellow")
            print(f"  content_index: {event.content_index}")

        elif event.type == "toolcall_delta":
            print_event_header("toolcall_delta", "📐", "yellow")
            print(f"  delta: [yellow]{repr(event.delta)}[/yellow]")

        elif event.type == "toolcall_end":
            print_event_header("toolcall_end", "✅", "bold green")
            print(f"  tool_call.name: {event.tool_call.name}")
            print(f"  tool_call.arguments: {json.dumps(event.tool_call.arguments, indent=2)}")

        elif event.type == "done":
            print_event_header("done", "🏁", "bold blue")
            print(f"  reason: {event.reason}")
            print(f"  message.stop_reason: {event.message.stop_reason}")
            print(f"  message.usage.total_tokens: {event.message.usage.total_tokens}")
            print(f"  message.content blocks ({len(event.message.content)}):")
            for i, block in enumerate(event.message.content):
                print(f"    [{i}] type={block.type}", end="")
                if hasattr(block, "text"):
                    print(f" text={repr(block.text[:120])}")
                elif hasattr(block, "name"):
                    print(f" name={block.name} id={block.id}")
                else:
                    print()

        elif event.type == "error":
            print_event_header("error", "❌", "bold red")
            print(f"  error.error_message: {event.error.error_message}")

    print()
    print(f"[dim]Round 2 共收到 {event_index} 个事件[/dim]")


async def main():
    models = builtin_models()
    model = models.get_model("deepseek", "deepseek-v4-pro")
    if not model:
        print("[red]Model deepseek-v4-pro not found![/red]")
        return

    print()
    print(Panel.fit(
        f"Model: [bold]{model.name}[/bold]\n"
        f"Provider: {model.provider}  |  API: {model.api}\n"
        f"Base URL: {model.base_url}\n"
        f"Max Tokens: {model.max_tokens}  |  Context Window: {model.context_window}",
        title="Model Info"
    ))

    # Round 1: 模型看到 tool definition → 决定调用 tool
    context, assistant_message = await round_one(models, model)

    # Round 2: 把假工具结果发回去 → 模型用自然语言总结
    await round_two(models, model, context, assistant_message)

    print()
    print("[bold]✅ 全流程结束[/bold]")


if __name__ == "__main__":
    asyncio.run(main())

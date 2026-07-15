"""
example_eventbus.py

演示如何使用 StreamBroadcaster (EventBus) 将大模型的推理流分发给多个独立的并发消费者。
"""
import asyncio
import sys

from typing import AsyncGenerator

from dotenv import load_dotenv


# Force UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

from rich.console import Console
from rich.panel import Panel

from provider.providers.all import builtin_models
from provider.types import Context, UserMessage, StreamEvent
from agent.eventbus import StreamBroadcaster

load_dotenv()

console = Console(force_terminal=True)


async def consumer_ui(stream: asyncio.Queue | AsyncGenerator):
    """模拟前端 UI 消费者：只关心文本片段，并实时打字输出"""
    console.print("[blue][UI] 开始监听文字流...[/blue]")
    print_buffer = ""
    
    async for event in stream:
        if event.type == "text_delta":
            print_buffer += event.delta
            # 模拟在终端横向打字
            print(event.delta, end="", flush=True)
            
        elif event.type == "done":
            print()
            console.print("[blue][UI] 接收到 done 事件，停止渲染。[/blue]")
            
        # 故意忽略工具调用等其他事件，假装不知道
        await asyncio.sleep(0.01)  # 模拟一点渲染延迟


async def consumer_logger(stream: asyncio.Queue | AsyncGenerator):
    """模拟后端审计日志：只记录核心节点，写入假想的数据库"""
    console.print("[green][Logger] 开始记录审计日志...[/green]")
    
    async for event in stream:
        if event.type == "start":
            console.print("[green][Logger] 记录：生成任务开始...[/green]")
        elif event.type == "toolcall_start":
            console.print("[green][Logger] 记录：模型试图调用工具！[/green]")
        elif event.type == "done":
            usage = event.message.usage
            console.print(f"[green][Logger] 记录：任务结束。消耗 Tokens: in={usage.input}, out={usage.output}[/green]")
        elif event.type == "error":
            console.print(f"[bold red][Logger] 记录：任务报错中断！原因: {event.error.error_message}[/bold red]")

async def consumer_trace(stream: asyncio.Queue | AsyncGenerator):
    """模拟 Trace 消费者：全量监听，但处理得很慢，验证背压是否生效"""
    console.print("[magenta][Trace] 开始缓慢处理全量 Trace 数据...[/magenta]")
    event_count = 0
    
    async for event in stream:
        event_count += 1
        # 故意每条处理 50ms（非常慢），测试会不会阻塞 UI
        await asyncio.sleep(0.05)
        
    console.print(f"[magenta][Trace] 结束。全量接收完毕，共捕获 {event_count} 个事件！[/magenta]")


async def main():
    models = builtin_models()
    await models.refresh("ollama")
    model = models.get_models("ollama")[1]

    # model = models.get_model("deepseek","deepseek-v4-flash")
        
    console.print(Panel(f"使用模型: {model.name}", style="bold cyan"))
    
    context = Context(
        messages=[UserMessage(content="请写一首描述赛博朋克的简短现代诗。")]
    )
    
    # 1. 拿到最原始的、只能消费一次的 AsyncGenerator
    raw_stream = models.stream(model, context)
    
    # 2. 挂载到我们新写的 EventBus (StreamBroadcaster) 上
    bus = StreamBroadcaster[StreamEvent](raw_stream, max_queue_size=100)
    
    # 3. 牵出三根独立的水管 (AsyncGenerator)
    ui_stream = bus.subscribe()
    log_stream = bus.subscribe()
    trace_stream = bus.subscribe()
    
    # 4. 把水管交给各个独立并发的子系统
    # asyncio.gather 会让它们完全并发运行
    tasks = [
        asyncio.create_task(consumer_ui(ui_stream)),
        asyncio.create_task(consumer_logger(log_stream)),
        asyncio.create_task(consumer_trace(trace_stream)),
        
        # 最后，别忘了启动核心水泵引擎，开始从底层抽水
        bus.pump_in_background()
    ]
    
    # 等待所有消费者消费完毕
    await asyncio.gather(*tasks)
    
    console.print("\n[bold cyan]全部子系统处理完毕！[/bold cyan]")

if __name__ == "__main__":
    asyncio.run(main())

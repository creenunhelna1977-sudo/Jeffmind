"""
example_react.py

ReAct 演示脚本。

内置工具：
  - calculator    —— 计算数学表达式（Python eval 沙箱）
  - current_time  —— 返回当前时间
  - word_counter  —— 统计一段文字的词数 / 字符数
  - search_python_docs —— 模拟查文档（假工具，返回固定内容，演示多步推理）

运行方式：
  uv run python example_react.py
"""
import asyncio
import math
import re
import sys
import textwrap
from datetime import datetime
from dotenv import load_dotenv

# Force UTF-8 on Windows (avoids GBK UnicodeEncodeError with rich)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

# Use a single Console instance so we stay on the same stream
console = Console(force_terminal=True)

# Override built-in print with rich's console.print for consistent output
from functools import partial
print = partial(console.print)  # noqa: A001

from provider.providers.all import builtin_models
from provider.types import Context, UserMessage

from agent.react import ReActAgent, ReActTool

load_dotenv()
console = Console()


# ============================================================
# 工具实现
# ============================================================

async def calculator(expression: str) -> str:
    """安全计算数学表达式。只允许数字和基本运算。"""
    # 白名单：只允许数字、运算符、括号、小数点、空格、以及 math 函数名
    allowed = re.compile(r'^[\d\s\+\-\*\/\(\)\.\,\%\^a-zA-Z_]+$')
    if not allowed.match(expression):
        return f"Error: 表达式含非法字符 -> {expression!r}"
    try:
        # 暴露 math 模块里的函数，但不让 import 等指令进来
        safe_globals = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
        safe_globals['__builtins__'] = {}
        result = eval(expression, safe_globals)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


async def current_time(timezone: str = "local") -> str:
    """返回当前时间。"""
    now = datetime.now()
    return f"当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}（本地时间，参数 timezone={timezone!r} 仅供参考）"


async def word_counter(text: str) -> str:
    """统计文字中的词数、中文字数和总字符数。"""
    # 英文词数
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    # 中文字数（每个汉字算一词）
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_words = english_words + chinese_chars
    total_chars = len(text)
    return (
        f"分析结果：\n"
        f"  英文单词数：{english_words}\n"
        f"  中文字符数：{chinese_chars}\n"
        f"  合计词/字数：{total_words}\n"
        f"  总字符数（含空格标点）：{total_chars}"
    )


async def list_prime_numbers(limit: int) -> str:
    """返回 2 到 limit 之间的所有质数。"""
    if limit > 10000:
        return "Error: limit 太大，最多支持 10000"
    if limit < 2:
        return "范围内没有质数"
    sieve = list(range(limit + 1))
    sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = 0
    primes = [x for x in sieve if x]
    return f"2 到 {limit} 的质数（共 {len(primes)} 个）：{primes}"


async def text_transform(text: str, operation: str) -> str:
    """
    对文本做变换。
    operation 支持：upper / lower / reverse / title / snake_case
    """
    ops = {
        "upper": text.upper,
        "lower": text.lower,
        "reverse": lambda: text[::-1],
        "title": text.title,
        "snake_case": lambda: re.sub(r'\s+', '_', text.lower()),
    }
    fn = ops.get(operation.lower())
    if not fn:
        return f"Error: 不支持的 operation={operation!r}，可选：{list(ops.keys())}"
    return fn()


# ============================================================
# 工具注册表
# ============================================================

TOOLS = [
    ReActTool(
        name="calculator",
        description=(
            "计算数学表达式。支持加减乘除、幂运算、括号，以及 Python math 模块里的函数"
            "（如 sqrt, sin, cos, log, pi, e 等）。"
            "expression 参数传入表达式字符串，例如 'sqrt(16) + 2 * pi'。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "要计算的数学表达式"}
            },
            "required": ["expression"]
        },
        fn=calculator,
    ),
    ReActTool(
        name="current_time",
        description="获取当前日期和时间。",
        parameters={
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "时区名称，默认 local"}
            },
            "required": []
        },
        fn=current_time,
    ),
    ReActTool(
        name="word_counter",
        description="统计一段文字中的词数（英文单词 + 中文字符）和总字符数。",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要统计的文字内容"}
            },
            "required": ["text"]
        },
        fn=word_counter,
    ),
    ReActTool(
        name="list_prime_numbers",
        description="列出 2 到 limit 之间的所有质数。",
        parameters={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "上限（含），最大 10000"}
            },
            "required": ["limit"]
        },
        fn=list_prime_numbers,
    ),
    ReActTool(
        name="text_transform",
        description="对文本进行格式变换：upper/lower/reverse/title/snake_case。",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "原始文字"},
                "operation": {
                    "type": "string",
                    "enum": ["upper", "lower", "reverse", "title", "snake_case"],
                    "description": "变换操作"
                }
            },
            "required": ["text", "operation"]
        },
        fn=text_transform,
    ),
]


# ============================================================
# 漂亮的打印
# ============================================================

TOOL_ICON = {
    "calculator": "[bold yellow]calc[/bold yellow]",
    "current_time": "[bold cyan]time[/bold cyan]",
    "word_counter": "[bold green]words[/bold green]",
    "list_prime_numbers": "[bold magenta]primes[/bold magenta]",
    "text_transform": "[bold blue]text[/bold blue]",
}


def print_banner():
    print()
    print(Panel.fit(
        "[bold cyan]ReAct Agent Demo[/bold cyan]\n"
        "[dim]Reasoning + Acting -- based on provider layer[/dim]\n\n"
        "[bold]Available tools:[/bold]\n"
        + "\n".join(
            f"  [{i+1}] [yellow]{t.name}[/yellow] -- {t.description[:60]}..."
            for i, t in enumerate(TOOLS)
        ),
        title="[bold]ReAct Agent[/bold]"
    ))


async def run_task(agent: ReActAgent, question: str):  # noqa: C901
    print()
    print(Rule(f"[bold]Question[/bold]"))
    print(Panel(f"[bold white]{question}[/bold white]", border_style="blue"))

    context = Context(
        system_prompt=(
            "You are a smart assistant with multiple tools available. "
            "When asked about calculations, time, word counts, etc., use the appropriate tools. "
            "You may chain multiple tool calls to complete complex tasks. "
            "Always summarize results clearly."
        ),
        messages=[UserMessage(content=question)],
        tools=None,
    )

    total_input_tokens = 0
    total_output_tokens = 0

    import json
    async for event in agent.run(context):

        if event.type == "react_iteration":
            print()
            print(Rule(
                f"[dim]Iteration {event.iteration}  |  messages: {event.message_count}[/dim]",
                style="dim"
            ))

        elif event.type == "react_tool_call":
            tc = event.tool_call
            print(f"\n  >> [bold yellow]Tool call:[/bold yellow] [cyan]{tc.name}[/cyan]")
            print(f"     args: [dim]{json.dumps(tc.arguments, ensure_ascii=False)}[/dim]")

        elif event.type == "react_tool_result":
            status = "[red]ERROR[/red]" if event.is_error else "[green]OK[/green]"
            result_preview = event.result.replace("\n", " ")[:120]
            print(f"  {status} result: {result_preview}")

        elif event.type == "react_done":
            msg = event.message
            total_input_tokens += msg.usage.input
            total_output_tokens += msg.usage.output

            text_blocks = [b for b in msg.content if hasattr(b, 'text')]
            final_text = "\n".join(b.text for b in text_blocks).strip()

            print()
            print(Rule("[bold green]Final Answer[/bold green]", style="green"))
            print(Panel(
                final_text or "[dim](no text output)[/dim]",
                border_style="green"
            ))
            print(
                f"\n[dim]  Done | iterations: {event.iterations}"
                f" | tokens in: {total_input_tokens}  out: {total_output_tokens}[/dim]"
            )

        elif event.type == "react_error":
            print()
            print(Panel(
                f"[bold red]Error[/bold red] (iteration {event.iteration})\n{event.reason}",
                border_style="red"
            ))


# ============================================================
# 主程序
# ============================================================

DEMO_QUESTIONS = [
    # 单工具
    "现在是几点？",
    # 需要计算
    "sin(pi/6) 的值是多少？再加上 sqrt(3)/2 是多少？",
    # 多步骤：计算 + 质数
    "100 以内有多少个质数？这个数字的平方是多少？",
    # 文字处理
    "帮我把 'hello world from react agent' 转成 snake_case，再转成大写",
    # 综合：时间 + 计算
    "现在几点？2024 年到现在过了多少天（假设 2024-01-01 是第 1 天，今天是 2025-07-14）？",
]


async def main():
    print_banner()

    # 初始化模型，动态拉取 Ollama 列表
    models = builtin_models()
    await models.refresh("ollama")   # 拉取本地已安装的模型列表

    # 优先级：deepseek (有余额) → ollama 本地 → 报错
    model = models.get_model("deepseek", "deepseek-v4-pro")

    if not model:
        # 从 Ollama 里挑第一个非 embedding 的模型
        ollama_models = models.get_models("ollama")
        # bge 等 embedding 模型不支持 chat，过滤掉
        chat_models = [m for m in ollama_models if "bge" not in m.id.lower()]
        if chat_models:
            model = chat_models[0]

    if not model:
        all_models = models.get_models()
        print(f"[red]未找到可用模型，已注册：{[m.id for m in all_models]}[/red]")
        return

    print()
    print(Panel.fit(
        f"[bold]{model.name}[/bold]  ({model.provider} / {model.api})\n"
        f"Base URL: {model.base_url}",
        title="🤖 使用模型"
    ))

    agent = ReActAgent(
        models=models,
        model=model,
        tools=TOOLS,
        max_iterations=8,
        pass_through_stream_events=False,  # 改成 True 可以看底层 Stream 事件
    )

    # Run all demo questions
    for question in DEMO_QUESTIONS:
        await run_task(agent, question)
        print()
        print(Rule(style="dim"))
        await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(main())

"""
agent/eventbus.py

实现了一个基于 asyncio.Queue 的多路异步流分发器 (Pub/Sub)。
用于解决 Python 异步生成器单次消费的问题。
"""
import asyncio
from typing import AsyncGenerator, AsyncIterable, TypeVar, Generic

T = TypeVar('T')

class StreamBroadcaster(Generic[T]):
    """
    流媒体中继路由器。
    将一个单消费者的 AsyncIterable 转换为多个独立的 AsyncGenerator 管道。
    """
    def __init__(self, source: AsyncIterable[T], max_queue_size: int = 100):
        self._source = source
        self._max_queue_size = max_queue_size
        self._subscribers: list[asyncio.Queue[T | Exception | None]] = []
        self._pumping_task: asyncio.Task | None = None

    def subscribe(self) -> AsyncGenerator[T, None]:
        """
        创建一个独立的订阅管道。
        返回一个 AsyncGenerator，消费者可以使用 async for 迭代它。
        """
        queue: asyncio.Queue[T | Exception | None] = asyncio.Queue(maxsize=self._max_queue_size)
        self._subscribers.append(queue)
        return self._queue_iterator(queue)

    async def _queue_iterator(self, queue: asyncio.Queue[T | Exception | None]) -> AsyncGenerator[T, None]:
        """将 asyncio.Queue 转换为 AsyncGenerator。"""
        try:
            while True:
                item = await queue.get()
                if item is None:
                    # None 代表流正常结束
                    queue.task_done()
                    break
                if isinstance(item, Exception):
                    # 如果发生异常，向上抛出
                    queue.task_done()
                    raise item
                
                yield item
                queue.task_done()
        finally:
            # 如果消费者中途主动 break 退出，可以考虑从 subscribers 移除 queue
            if queue in self._subscribers:
                self._subscribers.remove(queue)

    async def start_pumping(self):
        """
        开启主水泵，消费底层流并派发到所有订阅的队列中。
        """
        try:
            async for item in self._source:
                # 给所有存活的消费者发数据
                for q in self._subscribers:
                    await q.put(item)
            
            # 正常结束：给所有队列发送 None 信号
            for q in self._subscribers:
                await q.put(None)
                
        except Exception as e:
            # 异常结束：给所有队列发送 Exception 信号
            for q in self._subscribers:
                await q.put(e)
            raise e

    def pump_in_background(self) -> asyncio.Task:
        """
        在后台启动水泵，返回 Task 对象以便追踪。
        """
        if self._pumping_task is not None:
            raise RuntimeError("Broadcaster is already pumping.")
        self._pumping_task = asyncio.create_task(self.start_pumping())
        return self._pumping_task

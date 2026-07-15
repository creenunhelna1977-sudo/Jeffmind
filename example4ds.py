import asyncio
import os
from dotenv import load_dotenv

from provider.providers.all import builtin_models
from provider.types import Context, UserMessage

from rich import print

# Load environment variables from .env file
load_dotenv()

async def main():
    models = builtin_models()
    
    # Using DeepSeek Chat
    model = models.get_model("deepseek", "deepseek-v4-flash")
    if not model:
        print("Model not found!")
        return

    print(f"Using model: {model.name}")
    
    context = Context(
        system_prompt="You are a helpful assistant.",
        messages=[UserMessage(content="Hello! Can you briefly introduce yourself?")]
    )

    print('=========')
    print("\n--- Streaming Response ---")
    print('=========')
    
    try:
        async for event in models.stream(model, context):
            if event.type == "text_delta":
                print(event.delta, end="", flush=True)
            elif event.type == "error":
                print(f"\nError: {event.error.error_message}")
            elif event.type == "done":
                print(f"\n\n[Done] Reason: {event.reason}")
                print(f"[Usage] Prompt: {event.message.usage.input}, Completion: {event.message.usage.output}, Total: {event.message.usage.total_tokens}")
    except Exception as e:
        print(f"\nException: {e}")

if __name__ == "__main__":
    asyncio.run(main())

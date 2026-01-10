import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from src.core.context_compressor import compressor

async def run_benchmark():
    print("🚀 Starting Context Compression Benchmark...")
    
    # 1. Simulate Long History (12 messages, 6 turns)
    history = []
    for i in range(1, 7):
        history.append(HumanMessage(content=f"User question {i}: I like sci-fi."))
        history.append(AIMessage(content=f"AI answer {i}: Here is a sci-fi book."))
        
    print(f"Original History Length: {len(history)} messages")
    
    # 2. Compress
    print("Compressing...")
    # Mock LLM generation usually takes time, so latency includes API call
    compressed = await compressor.compress_history(history)
    
    print(f"Compressed History Length: {len(compressed)} messages")
    
    # 3. Validation
    # Expected: 1 SystemMessage (Summary) + 4 Messages (Recent) = 5
    if len(compressed) == 5:
        print("✅ SUCCESS: History compressed to 5 messages.")
        print(f"Summary Content: {compressed[0].content}")
        print(f"Oldest Retained Message: {compressed[1].content}")
    else:
        print(f"❌ FAILURE: Expected 5 messages, got {len(compressed)}")
        for i, m in enumerate(compressed):
            print(f"[{i}] {type(m).__name__}: {m.content}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())

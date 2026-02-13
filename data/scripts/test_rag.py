
import asyncio
import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

from src.services.chat_service import chat_service

async def main():
    isbn = "0001047604"  # Aurora Leigh
    query = "What is the emotional tone of this book?"
    
    print(f"Testing ChatService with ISBN={isbn}, Query='{query}'...")
    print("-" * 50)
    
    try:
        # Use 'mock' provider to test flow without key
        async for chunk in chat_service.chat_stream(
            isbn=isbn, 
            user_query=query, 
            provider="mock"
        ):
            print(chunk, end="", flush=True)
        print("\n" + "-" * 50)
        print("✅ Test Completed Successfully!")
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

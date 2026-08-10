import asyncio
from ai.agents.supervisor import classify_intent

async def main():
    print("=== Testing Self-Contained Query ===")
    state_self = {
        "user_query": "What is the capital of France?",
        "chat_history": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you?"}
        ]
    }
    res_self = await classify_intent(state_self)
    print("Result:", res_self)
    
    print("\n=== Testing Ambiguous Follow-Up (Short) ===")
    state_short = {
        "user_query": "What about Germany?",
        "chat_history": [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ]
    }
    res_short = await classify_intent(state_short)
    print("Result:", res_short)

    print("\n=== Testing Ambiguous Follow-Up (Long, previously skipped) ===")
    state_long = {
        "user_query": "Can you elaborate more on what you meant in your second point regarding the CEO's statement?",
        "chat_history": [
            {"role": "user", "content": "Summarize the earnings call."},
            {"role": "assistant", "content": "1. Revenue increased. 2. The CEO stated that Q4 will see slower growth. 3. New products are launching."}
        ]
    }
    res_long = await classify_intent(state_long)
    print("Result:", res_long)

if __name__ == "__main__":
    asyncio.run(main())

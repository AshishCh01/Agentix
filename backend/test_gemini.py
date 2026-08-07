import asyncio
from google import genai
from app.config.settings import settings


async def test_gemini_connection():
    print("⏳ Testing Google Gemini API Key...")

    # Diagnostic print
    raw_key = settings.GEMINI_API_KEY
    print(f"DEBUG: Loaded key length = {len(raw_key)}")
    print(f"DEBUG: Key starts with = '{raw_key[:10]}...'")

    if not raw_key:
        print("❌ GEMINI_API_KEY is completely empty in settings!")
        return

    try:
        client = genai.Client(api_key=raw_key)

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents="Respond with 'Gemini API is connected successfully!' if you receive this message.",
        )

        print(f"✅ Response from Gemini:\n{response.text}")

    except Exception as e:
        print(f"❌ Gemini API Test Failed! Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_gemini_connection())
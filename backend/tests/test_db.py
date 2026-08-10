import asyncio
from sqlalchemy import text
from app.database.connection import AsyncSessionLocal, engine
from app.models import Base


async def test_connection():
    print("⏳ Testing Async Database Connection...")
    try:
        # 1. Test basic raw SQL execution
        async with AsyncSessionLocal() as session:
            result = await session.execute(text("SELECT 1;"))
            val = result.scalar()
            print(f"✅ Database Connection Successful! Query output: {val}")

        # 2. Check if pgvector extension is enabled
        async with AsyncSessionLocal() as session:
            vector_check = await session.execute(
                text(
                    "SELECT extname FROM pg_extension WHERE extname = 'vector';"
                )
            )
            ext = vector_check.scalar()
            if ext == "vector":
                print("✅ pgvector extension is active in Supabase!")
            else:
                print(
                    "⚠️  pgvector extension NOT found. Make sure to run 'CREATE EXTENSION vector;' in Supabase."
                )

        # 3. Verify Table Creation / Model Registration
        async with engine.begin() as conn:
            # Creates tables in Supabase if they don't exist yet
            await conn.run_sync(Base.metadata.create_all)
            print("✅ SQLAlchemy Models synchronized with database tables!")

    except Exception as e:
        print(f"❌ Connection Failed! Error: {e}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(test_connection())
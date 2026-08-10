import asyncio
from sqlalchemy import text
from app.database.connection import AsyncSessionLocal

async def main():
    db = AsyncSessionLocal()
    res = await db.execute(text("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'document_chunks';"))
    for row in res.fetchall():
        print(row)
    await db.close()

if __name__ == "__main__":
    asyncio.run(main())

import asyncio

import pytest
from sqlalchemy import text

from app.database.connection import AsyncSessionLocal, engine
from app.models import Base


@pytest.mark.asyncio
async def test_connection():
    """
    Live connectivity smoke test for the configured database.

    Previously this caught every exception, printed it, and returned
    normally -- so a broken connection string, a missing pgvector
    extension, or a failed model sync all still reported PASSED. Each step
    now asserts on the actual result instead.
    """
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(text("SELECT 1;"))
            assert result.scalar() == 1, "Basic 'SELECT 1' round-trip failed."

        async with AsyncSessionLocal() as session:
            vector_check = await session.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            )
            assert vector_check.scalar() == "vector", (
                "pgvector extension is not enabled -- run 'CREATE EXTENSION vector;' "
                "in the database."
            )

        async with engine.begin() as conn:
            # Creates tables in the database if they don't exist yet; also
            # verifies the SQLAlchemy models are valid against the live schema.
            await conn.run_sync(Base.metadata.create_all)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(test_connection())

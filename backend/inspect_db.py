import asyncio
from sqlalchemy import text
from app.database.connection import AsyncSessionLocal, engine

async def inspect_db():
    try:
        async with engine.begin() as conn:
            # Check foreign keys
            print("--- Foreign Keys ---")
            fk_sql = """
            SELECT
                tc.table_name, 
                kcu.column_name, 
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name 
            FROM 
                information_schema.table_constraints AS tc 
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                  AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name
                  AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name IN ('chat_sessions', 'documents', 'document_chunks', 'messages', 'agent_logs');
            """
            result = await conn.execute(text(fk_sql))
            for row in result.fetchall():
                print(f"{row[0]}.{row[1]} -> {row[2]}.{row[3]}")
                
            # Check Indexes
            print("\n--- Indexes ---")
            idx_sql = """
            SELECT tablename, indexname, indexdef 
            FROM pg_indexes 
            WHERE schemaname = 'public' AND tablename IN ('users', 'chat_sessions', 'documents', 'document_chunks', 'messages', 'agent_logs');
            """
            result = await conn.execute(text(idx_sql))
            for row in result.fetchall():
                print(f"Table: {row[0]}, Index: {row[1]}, Def: {row[2]}")
                
            # Check pgvector dimensions
            print("\n--- Columns ---")
            col_sql = """
            SELECT table_name, column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name IN ('users', 'chat_sessions', 'documents', 'document_chunks', 'messages', 'agent_logs');
            """
            result = await conn.execute(text(col_sql))
            for row in result.fetchall():
                print(f"{row[0]}.{row[1]} : {row[2]}")
                
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(inspect_db())

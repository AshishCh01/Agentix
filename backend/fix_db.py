import asyncio
from sqlalchemy import text
from app.database.connection import AsyncSessionLocal, engine

async def fix_db():
    print("Applying foreign keys to database...")
    try:
        async with engine.begin() as conn:
            print("Cleaning up orphaned records...")
            # Delete orphaned chat_sessions
            await conn.execute(text(
                "DELETE FROM chat_sessions WHERE user_id NOT IN (SELECT id FROM users);"
            ))
            # Delete orphaned documents
            await conn.execute(text(
                "DELETE FROM documents WHERE user_id NOT IN (SELECT id FROM users);"
            ))
            
            # Add foreign key to chat_sessions
            try:
                await conn.execute(text(
                    "ALTER TABLE chat_sessions ADD CONSTRAINT chat_sessions_user_id_fkey FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE;"
                ))
                print("Added foreign key to chat_sessions.user_id")
            except Exception as e:
                print(f"Error adding FK to chat_sessions (maybe it already exists?): {e}")

            # Add foreign key to documents
            try:
                await conn.execute(text(
                    "ALTER TABLE documents ADD CONSTRAINT documents_user_id_fkey FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE;"
                ))
                print("Added foreign key to documents.user_id")
            except Exception as e:
                print(f"Error adding FK to documents (maybe it already exists?): {e}")

    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(fix_db())

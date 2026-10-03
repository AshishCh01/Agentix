import asyncio
from supabase import create_client
from app.config.settings import settings
from app.database.connection import AsyncSessionLocal
from sqlalchemy import text

async def clear_all_data():
    print("WARNING: This will permanently delete all documents and embeddings.")
    print("Connecting to Supabase Storage...")
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    bucket = settings.SUPABASE_STORAGE_BUCKET
    
    # 1. Delete files from Supabase Storage
    try:
        # We loop to handle pagination if there are many files
        deleted_count = 0
        while True:
            res = supabase.storage.from_(bucket).list()
            if not res:
                break
            
            files = [f["name"] for f in res if f["name"] != ".emptyFolderPlaceholder"]
            if not files:
                break
                
            supabase.storage.from_(bucket).remove(files)
            deleted_count += len(files)
            
        print(f"Deleted {deleted_count} files from bucket '{bucket}'.")
    except Exception as e:
        print(f"Error deleting files from storage: {e}")
            
    # 2. Delete all records from Postgres (Cascades to chunks/embeddings)
    print("Connecting to Postgres database...")
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(text("TRUNCATE TABLE documents CASCADE;"))
            await db.commit()
        print("Successfully truncated 'documents' and 'document_chunks' tables in Postgres.")
    except Exception as e:
        print(f"Error truncating Postgres tables: {e}")
        
    print("Cleanup complete!")

if __name__ == "__main__":
    asyncio.run(clear_all_data())

import asyncio
from ai.services.parser_service import parse_document
from ai.services.chunking_service import chunk_text
from ai.services.embedding_service import embedding_service


async def run_ingestion_test():
    print("🚀 Starting Ingestion Engine Test...\n")

    # 1. Prepare sample document data
    sample_text = """
    Agentic RAG System Documentation
    =================================
    The Agentic RAG system combines Retrieval-Augmented Generation with autonomous agents.
    It allows users to upload documents in multiple formats including PDF, DOCX, TXT, and CSV.
    
    Architecture Overview:
    1. Parsing Engine: Extracts raw text from uploaded files using format-specific parsers.
    2. Chunking Service: Splits long text into overlapping chunks (e.g. 1000 characters with 200 character overlap).
    3. Embedding Service: Generates 768-dimensional vector embeddings using Google Gemini embedding models.
    4. Vector Store: Stores chunk content and embeddings in PostgreSQL using the pgvector extension.
    5. Agent Core: Uses Gemini to reason over retrieved contexts and answer user queries accurately.
    """

    file_bytes = sample_text.encode("utf-8")
    filename = "test_documentation.txt"

    # Step 1: Test Parsing
    print("--- Step 1: Testing Document Parser ---")
    try:
        parsed_text = await parse_document(file_bytes, filename)
        print(f"✅ Parsed raw text successfully! Total characters: {len(parsed_text)}\n")
    except Exception as e:
        print(f"❌ Parsing Failed: {e}")
        return

    # Step 2: Test Chunking
    print("--- Step 2: Testing Text Chunking ---")
    chunks = chunk_text(parsed_text, chunk_size=300, overlap=50)
    print(f"✅ Created {len(chunks)} chunks from parsed text.")
    for idx, chunk in enumerate(chunks):
        preview = chunk['content'].replace('\n', ' ')[:50]
        print(f"   • Chunk #{chunk['chunk_index']} ({chunk['char_count']} chars): \"{preview}...\"")
    print()

    # Step 3: Test Embedding Generation
    print("--- Step 3: Testing Embedding Generation ---")
    target_text = chunks[0]["content"] if chunks else "Default test string"
    try:
        vector = embedding_service.generate_embedding(target_text)
        print(f"✅ Successfully generated vector embedding!")
        print(f"   • Vector Dimension: {len(vector)}")
        print(f"   • Sample Vector Values: {vector[:5]}...")
    except Exception as e:
        print(f"❌ Embedding Generation Failed: {e}")
        return

    print("\n🎉 Full Ingestion Pipeline Verified Successfully!")


if __name__ == "__main__":
    asyncio.run(run_ingestion_test())
import sys
from pathlib import Path

# Add root directory to sys.path to allow importing from src
sys.path.append(str(Path(__file__).parent))

from src.pipelines.indexing import index_pdf
from src.db.store import get_store

def main():
    # File PDF test
    test_pdf_path = "data/pdfs/[Description]-LLMs-Fine-tuning.pdf"
    
    print("=" * 50)
    print("TEST STEP 1: Indexing PDF")
    print("=" * 50)
    
    try:
        # 1. Index PDF
        doc_id = index_pdf(test_pdf_path, force_reindex=True)
        print(f"\n=> Document ID: {doc_id}\n")

        print("=" * 50)
        print("TEST STEP 2: Check data in Qdrant")
        print("=" * 50)
        
        # 2. Check data in Qdrant
        store = get_store()
        docs = store.list_documents()
        print("List documents in the system:")
        for doc in docs:
            print(f" - {doc['filename']} (ID: {doc['document_id']}): {doc['total_chunks']} chunks, {doc['max_page']} pages")

        print("=" * 50)
        print("TEST STEP 3: Scroll all chunks of Document")
        print("=" * 50)
        
        # 3. Scroll chunks
        chunks = store.scroll_by_document(doc_id)
        print(f"Total chunks retrieved by scroll: {len(chunks)}")
        if chunks:
            sample = chunks[0].payload
            print("\nMetadata of the first chunk:")
            for key, value in sample.items():
                # Avoid printing overly long content
                if key == "content":
                    print(f"  {key}: {str(value)[:200]}... [truncated]")
                else:
                    print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"\nERROR: {str(e)}")

if __name__ == "__main__":
    main()

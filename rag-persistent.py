import os,time
# Set these variables before importing LlamaIndex or SentenceTransformer
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from pathlib import Path

# Core LlamaIndex Modules
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank

# Local Integration Modules
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core import PromptTemplate

def run_local_rag():
    print("=" * 60)
    print("      LOCAL ACCURACY-TUNED RAG ENGINE (OFFLINE)      ")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. INPUTS 
    # -------------------------------------------------------------------------

    DOCS_DIR='mytmp/txt'
    DB_DIR='chroma_rag'
    
    input_dir = Path(DOCS_DIR)
    db_path = Path(DB_DIR)

    LMOD="mannix/llama3.1-8b-abliterated:latest"
    #LMOD="dolphin-llama3:8b"
    # -------------------------------------------------------------------------
    # 2. LOCAL GLOBAL MODEL SETTINGS
    # -------------------------------------------------------------------------
    print("\n[1/4] Configuring local offline models...")
    
    # Setup LLM via local Ollama instance (dolphin-llama3:8b)
    # Low temperature guarantees analytical, non-creative generation
    Settings.llm = Ollama(
        model=LMOD, 
        request_timeout=300.0,
        temperature=0
    )

    # Setup Embedding Model using your locally saved model directory
    # Replace the string below with the absolute path if it isn't in your cache
    local_embedding_path = "BAAI/bge-large-en-v1.5"
    
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=local_embedding_path,
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        #local_files_only = True
    )

    # Accuracy Parsing: Clean chunk limits targeting precise sentence grouping
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)

    # -------------------------------------------------------------------------
    # 3. VECTOR STORAGE & INGESTION
    # -------------------------------------------------------------------------
    print("[2/4] Connecting to persistent ChromaDB client...")
    
    # Initialize the database engine
    chroma_client = chromadb.PersistentClient(path=str(db_path))
    chroma_collection = chroma_client.get_or_create_collection("local_rag_collection")
    
    # Map LlamaIndex data constructs onto the underlying Chroma storage engine
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

# CRITICAL FIX: Evaluate Chroma collection size first
    vector_count = chroma_collection.count()
    
    if vector_count > 0:
        # LOAD EXISTING: If weights exist, skip directory reading entirely
        print(f"[3/4] Found {vector_count} existing vectors in ChromaDB! Loading weights from storage...")
        index = VectorStoreIndex.from_vector_store(
            vector_store, 
            storage_context=storage_context
        )
    else:
        # INGEST NEW: Database is empty, parse your physical documents
        print(f"[3/4] Database empty ({vector_count} vectors). Ingesting files from: {input_dir}")
        if not input_dir.exists() or not input_dir.is_dir():
            print(f"[Error] Target directory '{input_dir}' does not exist.")
            return
        reader = SimpleDirectoryReader(input_dir=str(input_dir), required_exts=[".txt"])
        documents = reader.load_data()
        
        if not documents:
            print("[Error] No text files found and database is empty. Add data files first.")
            return
            
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        print(f"[Success] Processed and saved {len(documents)} document sources.")

    # -------------------------------------------------------------------------
    # 4. QUERY EXECUTION ENGINE & ACCURACY PIPELINE
    # -------------------------------------------------------------------------
    print("[4/4] Activating Reranker and synthesis prompts...")

    # Fast Local Reranking: Broaden the initial fetch to catch relevant keywords,
    # then squeeze it through a localized cross-encoder model to identify matches.
    local_reranker_path = "BAAI/bge-reranker-large"
    
    rerank_postprocessor = SentenceTransformerRerank(
        model=local_reranker_path,
        top_n=6  # Pass only the 3 absolute highest-scoring nodes to the LLM
    )

    query_engine = index.as_query_engine(
        similarity_top_k=12,  # Fetch 12 semantic options first
        node_postprocessors=[rerank_postprocessor],
        response_mode="tree_summarize"
    )

    # Enforced System Instruction: Locks local model output inside context bounds
    strict_rag_prompt = (
        "Context information is provided strictly below:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Analyze the provided text context and answer the question: {query_str}\n"
        "Follow these rules precisely:\n"
        "1. Answer using ONLY facts directly mentioned in the context text.\n"
        "2. Do NOT extrapolate, speculate, or utilize exterior knowledge frameworks.\n"
        "3. If the answer is completely missing from the context, respond word-for-word with: "
        "'I am unable to answer this question based on the provided documents.'\n"
        "Answer: "
    )
    # FIX: Wrap the raw string inside a PromptTemplate object
    strict_rag_prompt_template = PromptTemplate(strict_rag_prompt)

    # Apply the template object to the engine instead of a plain string
    query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": strict_rag_prompt_template} 
    )

    # -------------------------------------------------------------------------
    # 5. USER CONSOLE RUNTIME LOOP
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LOCAL RUNTIME ACTIVATED. Type 'exit' to close program.")
    print("=" * 60)

    while True:
        # Start the clock
        start_time = time.perf_counter()
        try:
            query = input("\nEnter query: ").strip()
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting pipeline safely. Goodbye!")
                break
            if not query:
                continue

            print("Processing locally...")
            response = query_engine.query(query)

            print("\n┌──[ RESPONSE ]")
            print(f"│ {response.response}")
            print("└──")
            
            print("\n[Source Citations]")
            for node in response.source_nodes:
                file_name = node.node.metadata.get('file_name', 'Unknown')
                score = f"{node.score:.4f}" if node.score is not None else "N/A"
                print(f" 📄 {file_name} (Confidence: {score})")

        except Exception as error:
            print(f"\n[Execution Exception]: {error}")

        # Stop the clock
        end_time = time.perf_counter()

        # Calculate and print execution time
        execution_time = end_time - start_time
        print(f"Query took {execution_time:.0f} seconds to complete.")
       
        # 4. Print results using exact class reflection mappings
        # Check for '.model_name' first (used by Embeddings), fallback to '.model' (used by LLMs)
        embedding_property = (
            getattr(Settings.embed_model, "model_name", None) 
            or getattr(Settings.embed_model, "model", "Unknown Embedding")
        )

        llm_property = (
            getattr(Settings.llm, "model", None) 
            or getattr(Settings.llm, "model_name", "Unknown LLM")
        )

        print("\n" + "="*50)
        print(" RUNTIME CONFIGURATION METADATA")
        print("="*50)
        print(f"📦 Active Embedding : {embedding_property}")
        print(f"🧠 Active Gen LLM    : {llm_property}")
        cdb=chroma_collection.name
        print(f"🗄️  Vector Database  : ChromaDB (Collection: {cdb})")
        print("="*50 + "\n")

if __name__ == "__main__":
    run_local_rag()


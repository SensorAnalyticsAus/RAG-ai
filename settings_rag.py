DOCS_DIR='mytmp/txt' # where user data as text files is kept 
DB_DIR='chroma_rag'  # dir name under which weights are saved
#LMOD="dolphin-llama3:8b" 
LMOD= "mannix/llama3.1-8b-abliterated:latest" #LLM
EMOD= "BAAI/bge-large-en-v1.5"   #Embedding
RRMOD="BAAI/bge-reranker-large"  #Reranker
LTEMP=0.0 #Set LMOD's temperature, max to 0.7-1.0 for higher variability
CHUNK_SIZE=512 # Chunk size, higher chunk size will add to compute
CHUNK_OVERLAP=64 # Roughly 10% of Chunk size
SIMTOP_K=12 #Fetch n semantic options first
RRTOP_K=3 #Pass only the p absolute highest-scoring nodes to the LLM
RESP_MODE="tree_summarize" # Ref docu for other response modes


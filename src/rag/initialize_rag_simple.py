import os
from pathlib import Path
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.readers import SimpleDirectoryReader
from chromadb import PersistentClient

# -----------------
DOCS_DIR = Path("./docs")
PERSIST_DIR = "vectorstore/chroma"
COLLECTION_NAME = "island_docs"

# Embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

chroma_client = PersistentClient(path=PERSIST_DIR)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, embed_model=embed_model)
query_engine = index.as_retriever(similarity_top_k=3)


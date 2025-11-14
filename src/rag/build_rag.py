import os
import pandas as pd
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
    Document,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from chromadb import PersistentClient

# -------------------------------
# 1️⃣ Load environment + config
# -------------------------------
load_dotenv()

PERSIST_DIR = "vectorstore/chroma"
COLLECTION_NAME = "island_docs"
DOCS_DIR = "app/docs"
CSV_DIR = "data/"

# -------------------------------
# 2️⃣ Embedding Model
# -------------------------------

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
# -------------------------------
# 3️⃣ Helper: Convert CSV to Documents
# -------------------------------
def csv_to_documents(csv_path):
    """Convert a CSV file to a list of LlamaIndex Documents."""
    print(f"📄 Converting CSV to text documents: {csv_path}")
    df = pd.read_csv(csv_path)
    docs = []
    for _, row in df.iterrows():
        content = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        docs.append(Document(text=content))
    return docs


# -------------------------------
# 4️⃣ Add new documents or CSVs to vector store
# -------------------------------
def add_new_data():
    """Load new .txt/.pdf files or CSVs and add them to the Chroma vector store."""
    print("🔹 Connecting to Chroma vector store...")
    chroma_client = PersistentClient(path=PERSIST_DIR)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Reuse existing index if exists
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    parser = SimpleNodeParser()
    new_nodes = []

    # Load textual documents
    if os.path.exists(DOCS_DIR):
        print(f"📂 Loading new documents from {DOCS_DIR} ...")
        documents = SimpleDirectoryReader(DOCS_DIR).load_data()
        new_nodes += parser.get_nodes_from_documents(documents)

    # Load CSV files
    if os.path.exists(CSV_DIR):
        for folder in os.listdir(CSV_DIR):
            for file in os.listdir(CSV_DIR + folder):
                if file.endswith(".csv"):
                    csv_path = os.path.join(CSV_DIR + folder, file)
                    csv_docs = csv_to_documents(csv_path)
                    new_nodes += parser.get_nodes_from_documents(csv_docs)

    if new_nodes:
        print(f"🧩 Adding {len(new_nodes)} new nodes to Chroma...")
        index.insert_nodes(new_nodes)
        index.storage_context.persist()
        print("✅ Vector store updated successfully.")
    else:
        print("⚠️ No new data found to add.")

# -------------------------------
# 6️⃣ Main execution
# -------------------------------
if __name__ == "__main__":
    print("🚀 Starting RAG pipeline with tabular + textual data...")

    # Step 1: Add new data (text or CSV)
    add_new_data()

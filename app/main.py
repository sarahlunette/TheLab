"""
FastAPI MVP: Chat + Plan generation (24h/72h) + RAG integration (Chroma + LlamaIndex)
Auth: HTTP Basic
Logs: Python logging
LLM: Local Mistral (safetensors, v0.2)
Docs store: ./docs (txt/md)
RAG vectorstore: ./vectorstore/chroma

Run:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import csv
import uuid
import datetime
import logging
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import FileResponse
from pydantic import BaseModel
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from chromadb import PersistentClient

# -----------------
# Load env vars
# -----------------
load_dotenv()
MISTRAL_MODEL_DIR = os.getenv('MISTRAL_MODEL_DIR', '../models/Mistral-7B-Instruct-v0.2')
PERSIST_DIR = "vectorstore/chroma"
COLLECTION_NAME = "island_docs"

DOCS_DIR = Path('./docs')
EXPORT_DIR = Path('./exports')
DOCS_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

API_USER = os.getenv('MVP_USER', 'admin')
API_PASS = os.getenv('MVP_PASS', 'password')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('mvp')

app = FastAPI(title='MVP Crisis Chat & Plan (RAG-enabled)')
security = HTTPBasic()
ACTION_LOGS = []

# -----------------
# Load local Mistral model
# -----------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"🧠 Loading Mistral model from {MISTRAL_MODEL_DIR} on {device}...")

tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL_DIR, local_files_only=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MISTRAL_MODEL_DIR,
    local_files_only=True,
    torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
    device_map='auto' if device == 'cuda' else None,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
logger.info("✅ Mistral model loaded successfully.")

# -----------------
# Pydantic model
# -----------------
class ChatRequest(BaseModel):
    question: str

# -----------------
# Auth helper
# -----------------
def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != API_USER or credentials.password != API_PASS:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials.username

# -----------------
# Initialize RAG
# -----------------
logger.info("⚙️ Initializing RAG components...")
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Wrap SBERT in HuggingFaceEmbedding (compatible with LlamaIndex)
sbert_model_path = "../models/all-MiniLM-L6-v2"
embed_model = HuggingFaceEmbedding(
    model_name=sbert_model_path,
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

# Persistent Chroma vectorstore
chroma_client = PersistentClient(path=PERSIST_DIR)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Initialize index
index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, embed_model=embed_model)
query_engine = index.as_retriever(similarity_top_k=3)

MAX_CONTEXT_TOKENS = 1000

def query_knowledge_base(question: str) -> str:
    retrieved_nodes = query_engine.retrieve(question)
    context = "\n".join([n.text for n in retrieved_nodes])
    inputs = tokenizer(context, return_tensors="pt")
    if inputs.input_ids.size(1) > MAX_CONTEXT_TOKENS:
        context_tokens = inputs.input_ids[:, -MAX_CONTEXT_TOKENS:]
        context = tokenizer.decode(context_tokens[0], skip_special_tokens=True)
    logger.info(f"📚 Retrieved {len(retrieved_nodes)} context chunks from RAG.")
    return context or "No relevant information found."

# -----------------
# Chat endpoint
# -----------------
@app.post("/chat")
def chat(request: ChatRequest, username: str = Depends(verify_credentials)):
    input_text = request.question
    rag_context = query_knowledge_base(input_text)
    prompt = f"""
You are an AI crisis assistant. Use the context below to answer concisely and factually.

Context:
{rag_context}

Question:
{input_text}

Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    ACTION_LOGS.append({
        'time': datetime.datetime.now().isoformat(),
        'username': username,
        'question': input_text,
        'context_used': rag_context[:1000],
        'answer': answer
    })

    return {"answer": answer, "context_used": rag_context}

# -----------------
# Plan generation endpoint
# -----------------
@app.get("/plan")
def plan(horizon: int = 24, username: str = Depends(verify_credentials)):
    if horizon == 24:
        plan_text = (
            "Plan 24h:\n"
            "- Abri / Shelter\n"
            "- Eau et nourriture\n"
            "- Recherche des personnes disparues\n"
            "- Soins médicaux urgents\n"
            "- Logistique et transport\n"
            "- Énergie et électricité\n"
            "- Premiers soutiens psychologiques\n"
        )
    elif horizon == 72:
        plan_text = (
            "Plan 72h:\n"
            "- Reconstitution des systèmes vitaux (eau, électricité)\n"
            "- Coordination ONG / Gouvernement\n"
            "- Déblaiement et sécurisation\n"
            "- Soutien psychologique\n"
            "- Remise en place des infrastructures critiques\n"
            "- Appels d’offre et reconstruction\n"
        )
    else:
        plan_text = "Plan non disponible pour cet horizon"

    filename = EXPORT_DIR / f"plan_{horizon}h_{uuid.uuid4().hex[:6]}.pdf"
    c = canvas.Canvas(str(filename), pagesize=A4)
    for i, line in enumerate(plan_text.split("\n")):
        c.drawString(50, 800 - i * 20, line)
    c.save()

    ACTION_LOGS.append({
        'time': datetime.datetime.now().isoformat(),
        'plan_horizon': horizon,
        'file': str(filename)
    })

    return FileResponse(str(filename), media_type='application/pdf', filename=f"plan_{horizon}h.pdf")

# -----------------
# Upload new document
# -----------------
@app.post("/upload_doc")
def upload_doc(file: UploadFile = File(...), username: str = Depends(verify_credentials)):
    filepath = DOCS_DIR / file.filename
    with open(filepath, 'wb') as f:
        f.write(file.file.read())
    return {"message": f"Document {file.filename} uploaded successfully."}

# -----------------
# Logs endpoints
# -----------------
@app.get("/logs")
def get_logs(username: str = Depends(verify_credentials)):
    return ACTION_LOGS

@app.get("/export_logs_csv")
def export_logs_csv(username: str = Depends(verify_credentials)):
    filename = EXPORT_DIR / f"logs_{uuid.uuid4().hex[:6]}.csv"
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['time', 'username', 'question', 'context_used', 'answer', 'plan_horizon', 'file'])
        writer.writeheader()
        for log in ACTION_LOGS:
            writer.writerow({k: log.get(k, "") for k in ['time', 'username', 'question', 'context_used', 'answer', 'plan_horizon', 'file']})
    return FileResponse(str(filename), media_type='text/csv', filename=filename.name)

"""
FastAPI MVP: Chat + Plan generation (24h/72h) + simple docs store + PDF/CSV export
Auth: HTTP Basic
Logs: Python logging
LLM: calls a local Mistral model loaded from a local folder (supports safetensors, v0.2 compatible)
Simple retrieval: keyword-based over loaded SOPs (docs stored in ./docs as .txt/.md)

Requirements (pip):
fastapi uvicorn python-multipart reportlab requests python-dotenv transformers torch accelerate safetensors tqdm
# Optional for better retrieval: scikit-learn sentence-transformers faiss-cpu

Run:
uvicorn fastapi_mistral_rag_mvp:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import io
import csv
import uuid
import time
import logging
import datetime
from pathlib import Path
from typing import List
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import FileResponse
from pydantic import BaseModel
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# -----------------
# Load env vars
# -----------------
load_dotenv()

MISTRAL_MODEL_DIR = os.getenv('MISTRAL_MODEL_DIR', '/mnt/c/Users/sarah/Desktop/TheLab_/models/Mistral-7B-Instruct-v0.2')
DOCS_DIR = Path('./docs')
EXPORT_DIR = Path('./exports')
EXPORT_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

API_USER = os.getenv('MVP_USER', 'admin')
API_PASS = os.getenv('MVP_PASS', 'password')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('mvp')

app = FastAPI(title='MVP Crisis Chat & Plan (FastAPI)')
security = HTTPBasic()
ACTION_LOGS = []

# -----------------
# Load model
# -----------------
logger.info(f"🧠 Loading Mistral model from: {MISTRAL_MODEL_DIR}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"💻 Device detected: {device}")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        MISTRAL_MODEL_DIR,
        local_files_only=True,
        use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        MISTRAL_MODEL_DIR,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map='auto' if device == 'cuda' else None,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True
    )
    logger.info(f"✅ Mistral model loaded successfully on {device}.")
except Exception as e:
    logger.error(f"❌ Failed to load Mistral model: {e}")
    raise SystemExit("Could not load local Mistral model. Ensure proper files exist.")

# -----------------
# Pydantic models
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
# Simple keyword-based retrieval
# -----------------
def retrieve_docs(question: str) -> str:
    context = ""
    for doc in DOCS_DIR.glob("*.txt"):
        with open(doc, 'r', encoding='utf-8') as f:
            content = f.read()
            if any(word.lower() in content.lower() for word in question.split()):
                context += content + "\n"
    return context

# -----------------
# Chat endpoint
# -----------------
@app.post("/chat")
def chat(request: ChatRequest, username: str = Depends(verify_credentials)):
    input_text = request.question
    context = retrieve_docs(input_text)

    inputs = tokenizer(f"{context}\nQuestion: {input_text}\nAnswer:", return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=256)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    ACTION_LOGS.append({
        'time': datetime.datetime.now().isoformat(),
        'question': input_text,
        'answer': answer
    })

    return {"answer": answer}

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

    # Export PDF
    filename = EXPORT_DIR / f"plan_{horizon}h_{uuid.uuid4().hex[:6]}.pdf"
    c = canvas.Canvas(str(filename), pagesize=A4)
    for i, line in enumerate(plan_text.split("\n")):
        c.drawString(50, 800 - i*20, line)
    c.save()

    ACTION_LOGS.append({
        'time': datetime.datetime.now().isoformat(),
        'plan_horizon': horizon,
        'file': str(filename)
    })

    return FileResponse(str(filename), media_type='application/pdf', filename=f"plan_{horizon}h.pdf")

# -----------------
# Document upload
# -----------------
@app.post("/upload_doc")
def upload_doc(file: UploadFile = File(...), username: str = Depends(verify_credentials)):
    filepath = DOCS_DIR / file.filename
    with open(filepath, 'wb') as f:
        f.write(file.file.read())
    return {"message": f"Document {file.filename} uploaded successfully."}

# -----------------
# Logs endpoint
# -----------------
@app.get("/logs")
def get_logs(username: str = Depends(verify_credentials)):
    return ACTION_LOGS

# -----------------
# CSV export of logs
# -----------------
@app.get("/export_logs_csv")
def export_logs_csv(username: str = Depends(verify_credentials)):
    filename = EXPORT_DIR / f"logs_{uuid.uuid4().hex[:6]}.csv"
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['time', 'question', 'answer', 'plan_horizon', 'file'])
        writer.writeheader()
        for log in ACTION_LOGS:
            writer.writerow({k: log.get(k, "") for k in ['time', 'question', 'answer', 'plan_horizon', 'file']})
    return FileResponse(str(filename), media_type='text/csv', filename=filename.name)

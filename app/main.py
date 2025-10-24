"""
FastAPI MVP: Chat + Plan generation (24h/72h) + RAG integration (Chroma + LlamaIndex)
Auth: HTTP Basic
Logs: Python logging
LLM: Local Mistral (safetensors, v0.2)
Docs store: ./docs (txt/md)
RAG vectorstore: ./vectorstore/chroma
Conversation Memory: LangChain ConversationBufferMemory

Run:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import csv
import uuid
import datetime
import logging
from pathlib import Path
from collections import defaultdict

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

# 🧠 LangChain memory
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

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
# Conversation memory
# -----------------
USER_MEMORIES = defaultdict(lambda: ConversationBufferMemory(return_messages=True))

# -----------------
# Chat endpoint with conversation memory
# -----------------
@app.post("/chat")
def chat(request: ChatRequest, username: str = Depends(verify_credentials)):
    input_text = request.question.strip()

    # Retrieve user-specific memory
    memory = USER_MEMORIES[username]

    # Retrieve RAG context
    rag_context = query_knowledge_base(input_text)

    # Build conversation history (last 5 messages)
    history_text = "\n".join([
        f"{msg.type.capitalize()}: {msg.content}"
        for msg in memory.chat_memory.messages[-5:]
    ])

    prompt = f"""
Tu es un **assistant IA de gestion de crise et de résilience territoriale**, chargé d’aider des décideurs locaux à **prioriser et planifier** des projets de résilience à partir de données structurées (tabulaires) et de retours d’expérience (retex).

Knowledge Base Context:
{rag_context}

Conversation History:
{history_text}

User: {input_text}

---

## 🎯 OBJECTIFS PRINCIPAUX
1. Identifier et **prioriser** les projets de résilience les plus pertinents pour le territoire.
2. Fournir pour chaque projet une **fiche décisionnelle complète** :
   - Description courte et objectifs
   - Justification issue du contexte RAG
   - Évaluation de faisabilité, impact, coût, et urgence
   - Ressources et parties prenantes clés
   - Calendrier prévisionnel (30 / 90 / 180 jours)
   - Risques et mesures d’atténuation
   - Indicateurs de suivi (KPIs)
3. Produire une **synthèse claire** et un **bloc JSON exploitable** pour automatiser la planification.

---

## 📊 CRITÈRES DE PRIORISATION
Chaque projet est évalué selon 6 critères pondérés :

| Critère | Description | Note (0–1) | Poids (w) |
|----------|-------------|------------|------------|
| E (Urgence) | Probabilité d’occurrence à court terme | 0–1 | 0.25 |
| I (Impact) | Population / infrastructures concernées | 0–1 | 0.30 |
| C (Coût) | Niveau de ressources nécessaires | 0–1 | 0.05 |
| F (Faisabilité) | Capacité technique, politique, humaine | 0–1 | 0.20 |
| L (Effet de levier) | Co-bénéfices / synergies | 0–1 | 0.15 |
| T (Temporalité) | Délai d’obtention de bénéfices | 0–1 | 0.05 |

**Score de priorité** :
Score_priorité = normalize( wE*(1-E) + wI*I + wF*F + wL*L + wT*T - wC*C )

---

## 📋 FORMAT DE SORTIE ATTENDU

### 🧾 1. Résumé pour décideur
- Trois phrases maximum, résumant les priorités et recommandations principales.

### 🧱 2. Liste des projets (Top N)
Pour chaque projet :
- `id`
- `titre_court`
- `description_brève`
- `score_priorite` (0–100)
- `confiance` (haute / moyenne / faible)
- `justification_synthese` (résumé + sources RAG)
- `sources`
- `ressources_estimees` (budget, personnel, matériel)
- `calendrier_recommandé` (30j / 90j / 180j)
- `parties_prenantes`
- `principaux_risques` + `mesures_dattenuation`
- `kpis` (≥3)
- `actions_immédiates`
- `score_components` (valeurs des 6 critères + score final)

### 💻 3. Bloc JSON machine-lisible
{
  "meta": {
    "generated_at": "<ISO8601>",
    "rag_ids_used": ["..."],
    "history_hash": "<hash>"
  },
  "summary": "...",
  "projects": [
    {
      "id": "proj_001",
      "title": "Protection des réseaux d’eau potable",
      "priority_rank": 1,
      "priority_score": 92.3,
      "confidence": "haute",
      "justification": "Basé sur 3 retex post-Irma indiquant panne réseau >48h.",
      "sources": ["RAG_doc_12", "table_infra_2017"],
      "resources_estimate": {"budget_eur": 150000, "fte": 3, "equipment": ["pompes", "générateurs"]},
      "timeline": {"30d": ["sécuriser 2 stations"], "90d": ["renforcer conduites"], "180d": ["auditer réseau complet"]},
      "stakeholders": ["Commune", "ARS", "Protection civile"],
      "risks": [{"risk": "retard d’approvisionnement", "mitigation": "prévoir stock tampon"}],
      "kpis": [{"kpi": "% foyers raccordés", "target": ">95%", "measure": "mensuel"}],
      "immediate_actions": ["contracter fournisseurs", "vérifier stock générateurs"],
      "score_components": {"E": 0.8, "I": 0.9, "C": 0.3, "F": 0.7, "L": 0.4, "T": 0.8, "computed_score": 92.3}
    }
  ],
  "data_gaps": ["manque données coût maintenance réseau"],
  "next_steps": ["vérifier inventaire matériel", "collecter données actualisées sur capacités locales"]
}

---

## ⚙️ RÈGLES DE CITATION ET TRAÇABILITÉ
- Toute donnée chiffrée ou factuelle doit mentionner sa **source RAG** (ex: `RAG_table_infra[row=12]`).
- Si une donnée est estimée, marque-la comme `ESTIMATION` et explique la méthode utilisée.
- Si des sources sont contradictoires, indique la version la plus probable et propose un test ou une collecte complémentaire.

---

## 🧭 GESTION DES CONTRAINTES ET CONTEXTE
- Respecte strictement les contraintes budgétaires, temporelles ou géographiques mentionnées par l’utilisateur.
- Si aucune contrainte n’est donnée, propose 5 projets prioritaires par défaut.
- Mentionne les données manquantes et propose des actions concrètes pour les combler.
- Ne produis jamais d’actions illégales, irréalistes ou contraires à l’éthique.

---

## ✍️ STYLE DE SORTIE
- Ton professionnel, clair et concis.
- D’abord la **réponse actionnable**, ensuite les **détails**.
- Évite le jargon technique non expliqué.
- Mentionne la **confiance** de chaque recommandation (haute / moyenne / faible).

Assistant:
"""


    # Generate response with Mistral
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Update memory
    memory.chat_memory.add_user_message(input_text)
    memory.chat_memory.add_ai_message(answer)

    # Log
    ACTION_LOGS.append({
        'time': datetime.datetime.now().isoformat(),
        'username': username,
        'question': input_text,
        'context_used': rag_context[:1000],
        'answer': answer
    })

    return {
        "answer": answer,
        "context_used": rag_context,
        "conversation_turns": len(memory.chat_memory.messages) // 2
    }

# -----------------
# Reset + History Endpoints
# -----------------
@app.delete("/chat/reset")
def reset_chat(username: str = Depends(verify_credentials)):
    USER_MEMORIES[username].clear()
    return {"message": f"Conversation reset for {username}."}

@app.get("/chat/history")
def get_history(username: str = Depends(verify_credentials)):
    memory = USER_MEMORIES[username]
    return [
        {"role": msg.type, "content": msg.content}
        for msg in memory.chat_memory.messages
    ]

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

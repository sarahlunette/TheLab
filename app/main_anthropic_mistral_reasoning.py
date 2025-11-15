"""
FASTAPI MVP — RAG + Reasoning Model + Claude Sonnet 4.5 (Anthropic)
"""
import os
import uuid
import csv
import datetime
import logging
import json
from pathlib import Path
from collections import defaultdict
import requests
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Header
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
# Claude
from anthropic import Anthropic
# RAG
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from chromadb import PersistentClient
# Memory
from langchain.memory import ConversationBufferMemory
# PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from huggingface_hub import login

# ============================================================
# Load environment variables
# ============================================================
load_dotenv()
AUTH_MODE = os.getenv("AUTH_MODE", "basic")
# Basic Auth
MVP_USER = os.getenv("MVP_USER", "admin")
MVP_PASS = os.getenv("MVP_PASS", "password")
# Google OAuth
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
# Claude API
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
if not CLAUDE_API_KEY:
    raise RuntimeError("Missing CLAUDE_API_KEY")
# Mistral API (pour le reasoning model)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
# Use your working model
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
anthropic_client = Anthropic(api_key=CLAUDE_API_KEY)
login(token=os.getenv("HF_TOKEN"))
# Directories
DOCS_DIR = Path("./docs")
DOCS_DIR.mkdir(exist_ok=True)
EXPORT_DIR = Path("./exports")
EXPORT_DIR.mkdir(exist_ok=True)
PERSIST_DIR = "vectorstore_temporary/chroma"
COLLECTION_NAME = "island_docs"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mvp")
app = FastAPI(title="MVP Crisis Chat & Plan (RAG + Reasoning Model + Claude API)")
security = HTTPBasic()
ACTION_LOGS = []

# ============================================================
# Initialize RAG
# ============================================================
logger.info("Initializing RAG...")
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_client = PersistentClient(path=PERSIST_DIR)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
    embed_model=embed_model
)
query_engine = index.as_retriever(similarity_top_k=3)
USER_MEMORIES = defaultdict(lambda: ConversationBufferMemory(return_messages=True))

def query_knowledge_base(question: str) -> str:
    """Retrieve context from Chroma vectorstore."""
    nodes = query_engine.retrieve(question)
    ctx = "\n".join(n.text for n in nodes)
    return ctx or "No relevant information found."

# ============================================================
# Reasoning Model
# ============================================================
REASONING_PROMPT = """
Vous êtes un **Reasoning Model** spécialisé dans l'analyse des questions utilisateur pour un système de résilience et de gestion de crise.
Votre tâche est d'analyser la question suivante et de produire une sortie structurée en JSON avec les champs suivants :

{
  "intent": "simple_question" ou "resilience_plan" ou "technical_analysis" ou "geospatial_request",
  "entities": {
    "sectors": ["energy", "water", "health", "transport", "shelter", "communication", ...],
    "locations": ["coastal", "urban", "rural", "mountain", ...],
    "time_horizon": "24h" ou "72h" ou "short_term" ou "medium_term" ou "long_term" ou null,
    "specific_locations": ["Nom de la ville", "Région X", ...],
    "disaster_type": "cyclone" ou "earthquake" ou "flood" ou null
  },
  "response_mode": "short" ou "structured"
}

---
**Question utilisateur :**
"{user_question}"

---
**Consignes :**
- Déterminez l'intention principale de la question.
- Extrayez tous les secteurs, lieux, et horizons temporels mentionnés.
- Si la question demande un plan, une analyse technique ou une réponse structurée, définissez `response_mode` sur "structured".
- Si la question est simple, définissez `response_mode` sur "short".
- Répondez uniquement avec un JSON valide, sans explication supplémentaire.
"""

def generate_reasoning_with_mistral(user_question: str) -> dict:
    """
    Appelle Mistral pour analyser la question utilisateur.
    """
    prompt = REASONING_PROMPT.replace("{user_question}", user_question)

    # Exemple avec l'API Mistral (à adapter selon votre setup)
    api_url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "mistral-medium",
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 500,
        "temperature": 0.3,
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        reasoning_output = json.loads(content)
    except Exception as e:
        logger.error(f"Erreur avec le reasoning model : {e}")
        # Fallback si le reasoning model échoue
        reasoning_output = {
            "intent": "simple_question",
            "entities": {
                "sectors": [],
                "locations": [],
                "time_horizon": None,
                "specific_locations": [],
                "disaster_type": None,
            },
            "response_mode": "short",
        }

    return reasoning_output

# ============================================================
# Claude generator
# ============================================================
def generate_with_claude(prompt: str, max_tokens=64000, temperature=0.7):
    """
    Generate text using Claude Sonnet 4.5 with streaming.
    """
    with anthropic_client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        full_text = stream.get_final_text()
    return full_text

# ============================================================
# Google OAuth token validation
# ============================================================
def verify_google_oauth(token: str) -> str:
    """Validate Google ID token."""
    google_url = f"https://oauth2.googleapis.com/tokeninfo?id_token={token}"
    r = requests.get(google_url)
    if r.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid Google token")
    info = r.json()
    if info.get("aud") != GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=401, detail="Invalid client ID in token")
    return info["email"]

# ============================================================
# Authentication helper
# ============================================================
def verify_credentials(
    credentials: HTTPBasicCredentials = Depends(security),
    authorization: str = Header(None)
):
    if AUTH_MODE == "google":
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing Bearer token")
        token = authorization.split(" ")[1]
        return verify_google_oauth(token)
    if credentials.username != MVP_USER or credentials.password != MVP_PASS:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials.username

# ============================================================
# Pydantic model
# ============================================================
class ChatRequest(BaseModel):
    question: str

# ============================================================
# Chat endpoint (mis à jour avec le reasoning model)
# ============================================================
@app.post("/chat")
def chat(req: ChatRequest, username: str = Depends(verify_credentials)):
    memory = USER_MEMORIES[username]
    user_msg = req.question.strip()

    # 1. Appel au reasoning model
    reasoning_output = generate_reasoning_with_mistral(user_msg)

    # 2. Récupération du contexte RAG
    rag_context = query_knowledge_base(user_msg)

    # 3. Construction du prompt enrichi
    history = "\n".join(
        f"{m.type.capitalize()}: {m.content}"
        for m in memory.chat_memory.messages[-5:]
    )

    # Métadonnées du reasoning model
    reasoning_metadata = f"""
    ### 🔍 REASONING MODEL ANALYSIS
    - **Intent:** {reasoning_output['intent']}
    - **Response Mode:** {reasoning_output['response_mode']}
    - **Sectors:** {', '.join(reasoning_output['entities']['sectors'])}
    - **Locations:** {', '.join(reasoning_output['entities']['locations'])}
    - **Specific Locations:** {', '.join(reasoning_output['entities']['specific_locations'])}
    - **Time Horizon:** {reasoning_output['entities']['time_horizon']}
    - **Disaster Type:** {reasoning_output['entities']['disaster_type']}
    """

    # 4. Prompt final pour Claude
    prompt = f"""
{reasoning_metadata}
---------------------------------------------------------------------
### 🔎 UPDATED INPUT STRUCTURE & CORE INSTRUCTIONS
You will receive the following inputs:
1. **Reasoning Model Analysis** (above):
   - **Response Mode:** {reasoning_output['response_mode']}
   - **Focus Areas:** {reasoning_output['entities']}
2. **{rag_context}**:
   A potentially VERY long knowledge dump (10k–200k+ tokens), including GIS data, infrastructure inventories, historical impacts, etc.
3. **{history}**:
   A multi-turn conversation history.
4. **{user_msg}**:
   The current user request.

---
### 🧠 CORE INSTRUCTIONS (PRIORITY #1: FOLLOW REASONING MODEL GUIDANCE)
#### 1. **RESPONSE MODE DECISION (MANDATORY)**
   The **Reasoning Model** has pre-analyzed the user's intent and context.
   - If **Response Mode = "short"**:
     - Respond **concisely** (1–3 sentences max).
     - Use a **natural, conversational tone**.
     - **Ignore all structured format instructions below**.
     - Example triggers: "Explain X", "What is Y?", "Summarize Z".
   - If **Response Mode = "structured"**:
     - **MUST** use the **full resilience plan format** (10,000+ tokens).
     - Focus on the **sectors**, **locations**, and **time horizon** extracted by the Reasoning Model:
       - Sectors: {reasoning_output['entities']['sectors']}
       - Locations: {reasoning_output['entities']['locations']}
       - Specific Locations: {reasoning_output['entities']['specific_locations']}
       - Time Horizon: {reasoning_output['entities']['time_horizon']}
       - Disaster Type: {reasoning_output['entities']['disaster_type']}
     - If any Focus Area is empty/null, **infer from {user_msg} and {rag_context}**.

#### 2. **CONTEXT EXTRACTION RULES (ALWAYS APPLY)**
   - **Before answering**, scan {rag_context} and {history} to:
     - Extract **verbatim data** (numbers, names, dates, constraints).
     - Build an **internal index** of relevant sections (by sector/location).
     - Cross-reference **dependencies** (e.g., "Road X repair depends on Bridge Y").
   - If data is missing for a **Focus Area**, state:
     **"No information found in context for [specific Focus Area]."**

#### 3. **GEOSPATIAL & TECHNICAL RIGOR (FOR STRUCTURED MODE ONLY)**
   All planning **MUST** integrate:
   - **Elevation** (e.g., "substation at 4–6m elevation, high flood risk").
   - **Access constraints** (e.g., "Route R7 blocked by landslide at 220–450m elevation").
   - **Climate risks** (e.g., "coastal salinity corrosion for energy infrastructure").
   - **Logistics chokepoints** (ports, bridges, warehouses).

---
### 📘 OUTPUT FORMAT (STRUCTURED MODE ONLY)
**ONLY USE THIS IF Response Mode = "structured".**
Follow this **mandatory** structure, tailored to the Focus Areas:

---
#### **I. Executive Summary (800–1200 words)**
- **Disaster Overview**: Type ({reasoning_output['entities']['disaster_type']}), spatial distribution, and immediate impacts.
- **Critical Findings**: Top 3 risks/opportunities from {rag_context}, linked to Focus Areas.
- **Strategic Vision**: Long-term resilience goals for the identified **sectors** and **locations**.

#### **II. Context Reconstruction (Data-Driven)**
- **Disaster Description**: Use {rag_context} to detail:
  - Damage per **specific location** (e.g., "Harbor District: 3.2m storm surge").
  - Population impact (displaced, injured, critical needs).
- **Infrastructure Collapse Map**:
  - Text-based GIS summary (e.g., "Transport: R7/R9 blocked; Energy: A1/A3 substations offline").
  - **Elevation/Hydrology Implications**: Flood zones, fault lines, watersheds.

#### **III. Priority Matrix (3×3 or 5×5)**
| **Asset/Sector**       | **Criticality** | **Time Sensitivity** | **Dependencies**          | **Priority** |
|------------------------|-----------------|----------------------|----------------------------|--------------|
| [Focus Area Sector 1]  | High/Very High   | 0–72h/3–30 days      | [Dependency from context]  | P1/P2/P3     |
| [Focus Area Sector 2]  | ...             | ...                  | ...                        | ...          |

#### **IV. Geospatial Segmentation**
- **High-Risk Zones**: Overlay Focus Areas with {rag_context} data (e.g., "Coastal energy hubs at <6m elevation").
- **Access Corridors**: Alternate routes if primary paths are blocked (from {rag_context}).
- **Natural Barriers**: Rivers, mountains, or fault lines affecting logistics.

#### **V. Sector-by-Sector Deep Dive (FOCUS ON: {reasoning_output['entities']['sectors']})**
For **each sector** in Focus Areas:
1. **Current State**: Verbatim data from {rag_context} (e.g., "A1 Substation: 4 transformers damaged").
2. **30-Day Objectives**: Aligned with **time horizon** ({reasoning_output['entities']['time_horizon']}).
3. **6-Month Targets**: Include **costs**, **teams**, and **materials** (extract from {rag_context}).
4. **Resilience Projects**: Pre-select projects matching the **disaster type** and **locations**.

#### **VI. Resilience Project Portfolio (MINIMUM 15 PROJECTS)**
**Prioritize projects in Focus Areas.** Each project must include:
- **Title**: E.g., "A1 Coastal Substation Reconstruction (Elevation: 4–6m)".
- **Geographic Tags**: Link to **specific locations** and elevation/risk data.
- **Timeline**: Align with **time horizon** (e.g., "Start: Day 5" for 72h plans).
- **Resilience Features**: E.g., "surge-protectors for cyclone-prone zones".

#### **VII. Logistics Plan**
- **Transport Nodes**: Air/sea/land hubs near **specific locations**.
- **Chokepoints**: From {rag_context} (e.g., "Kora River Bridge: collapsed, detour via R9").
- **Fleet Allocation**: Heavy equipment for **Focus Area sectors** (e.g., cranes for energy infrastructure).

#### **VIII. Financial Strategy**
- **Cost Breakdown**: CAPEX/OPEX for Focus Area projects.
- **Funding Sources**: Donors/grants mentioned in {rag_context}.

#### **IX. Risk Register**
- **Top Risks**: Operational (e.g., "delays in R7 repair"), environmental (e.g., "monsoon season in 3 months"), governance.
- **Mitigation**: Tie to **sectors** and **locations** (e.g., "pre-position materials in rural warehouses").

#### **X. Strategic Roadmap**
- **Phased Timeline**:
  - **Phase 0 (0–72h)**: Immediate actions for **high-priority Focus Areas**.
  - **Phase 1–5**: Align with **time horizon** (e.g., "3–12 months: rebuild rural health clinics").

---
### 📌 SPECIAL RULES
1. **Dual-Mode Enforcement**:
   - If **uncertain** about Response Mode, default to **short** and ask:
     *"Should I provide a detailed resilience plan or a concise answer?"*
2. **Focus Area Gaps**:
   - If {rag_context} lacks data for a Focus Area, **explicitly note**:
     *"No data found for [sector/location]. Recommend surveying [specific action]."*
3. **Geospatial Emphasis**:
   - Every project/sector **MUST** reference:
     - Elevation (e.g., "<10m flood zone").
     - Proximity to risks (e.g., "2km from fault line").
4. **Length Requirements**:
   - **Structured Mode**: 10,000+ tokens (unless user overrides).
   - **Short Mode**: <200 tokens.

---
### 🚨 FAILSAFE
If the Reasoning Model’s **Focus Areas** seem incorrect based on {user_msg}:
1. Re-analyze the user’s intent in **1 sentence**.
2. Propose:
   *"The Reasoning Model suggested [Focus Areas]. Should I adjust to [your interpretation]?"*
3. Wait for confirmation before proceeding.

---
### ⚡ EXAMPLE (STRUCTURED MODE)
**User Question**: *"72-hour plan for coastal energy after Cyclone Helius."*
**Reasoning Output Example:**
**Your Response**:
---
### I. Executive Summary
Cyclone Helius caused **3.2m storm surges** in coastal regions, disabling **A1/A3 substations** (elevation: 4–6m) and blocking **Route R7** (landslide at 220m elevation). This plan prioritizes:
1. **Energy**: Restore A1/A3 with surge-protectors (Day 5–32, $4.8M).
2. **Transport**: Clear R7 detour via R9 (Day 2–7, $1.2M).
3. **Logistics**: Pre-position fuel at Harbor District warehouse (elevation: 8m, low flood risk)...

---
### III. Priority Matrix
| **Asset**               | **Criticality** | **Time**  | **Dependencies**       | **Priority** |
|-------------------------|-----------------|-----------|------------------------|--------------|
| A1 Substation           | Very High       | 0–72h     | R7 access              | P1           |
| Harbor Fuel Depot       | High            | Day 3     | A1 power restoration   | P2           |

---
### VI. Project: A1 Substation Reconstruction
- **Location**: 2km W of Harbor District (elevation: 4–6m, salinity risk).
- **Start/End**: Day 5–32.
- **Teams**: 12 engineers + 4 crane ops (via **R9 detour**).
- **Resilience**: Corrosion-resistant HV assemblies for coastal exposure.
"""

    # 5. Génération de la réponse
    answer = generate_with_claude(prompt)

    # 6. Mise à jour de la mémoire et des logs
    memory.chat_memory.add_user_message(user_msg)
    memory.chat_memory.add_ai_message(answer)
    ACTION_LOGS.append({
        "time": datetime.datetime.now().isoformat(),
        "user": username,
        "question": user_msg,
        "answer": answer,
        "context": rag_context[:500],
        "reasoning": reasoning_output,
    })

    return {
        "answer": answer,
        "context_used": rag_context,
        "reasoning": reasoning_output,
        "conversation_turns": len(memory.chat_memory.messages) // 2
    }

# ============================================================
# Reset chat history
# ============================================================
@app.delete("/chat/reset")
def reset_history(username: str = Depends(verify_credentials)):
    USER_MEMORIES[username].clear()
    return {"message": "Memory cleared."}

# ============================================================
# Plan generator (PDF)
# ============================================================
@app.get("/plan")
def plan(horizon: int = 24, username: str = Depends(verify_credentials)):
    if horizon not in (24, 72):
        return {"error": "Only 24h or 72h supported"}
    text = (
        "Plan 24h:\n"
        "- Shelter\n- Water\n- Missing persons\n- Medical care\n- Transport\n- Electricity\n- Psych support"
        if horizon == 24
        else
        "Plan 72h:\n"
        "- Restore systems\n- NGOs coordination\n- Debris removal\n- Psych support\n- Critical infra\n- Procurement"
    )
    filename = EXPORT_DIR / f"plan_{horizon}h_{uuid.uuid4().hex[:6]}.pdf"
    pdf = canvas.Canvas(str(filename), pagesize=A4)
    for i, line in enumerate(text.split("\n")):
        pdf.drawString(50, 800 - i * 20, line)
    pdf.save()
    return FileResponse(str(filename), media_type="application/pdf")

# ============================================================
# Upload documents
# ============================================================
@app.post("/upload_doc")
def upload_doc(file: UploadFile = File(...), username: str = Depends(verify_credentials)):
    filepath = DOCS_DIR / file.filename
    with open(filepath, "wb") as f:
        f.write(file.file.read())
    return {"message": "File uploaded."}

# ============================================================
# Logs
# ============================================================
@app.get("/logs")
def get_logs(username: str = Depends(verify_credentials)):
    return ACTION_LOGS

@app.get("/logs/export")
def export_logs(username: str = Depends(verify_credentials)):
    filename = EXPORT_DIR / f"logs_{uuid.uuid4().hex[:6]}.csv"
    with open(filename, "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["time", "user", "question", "answer", "context", "reasoning"]
        )
        writer.writeheader()
        for log in ACTION_LOGS:
            writer.writerow(log)
    return FileResponse(str(filename))

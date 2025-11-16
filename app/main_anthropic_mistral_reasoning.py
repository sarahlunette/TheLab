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
from typing import Optional

import requests
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Header
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

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

# HF login (for embeddings if needed)
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

# Mistral API (reasoning model)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Claude model
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")

# Anthropic client
anthropic_client = Anthropic(api_key=CLAUDE_API_KEY)

# HF login (optional – used by HuggingFaceEmbedding)
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# Directories
DOCS_DIR = Path("./docs")
DOCS_DIR.mkdir(exist_ok=True)

EXPORT_DIR = Path("./exports")
EXPORT_DIR.mkdir(exist_ok=True)

PERSIST_DIR = "vectorstore_temporary/chroma"
COLLECTION_NAME = "island_docs"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mvp")

# FastAPI app
app = FastAPI(title="MVP Crisis Chat & Plan (RAG + Reasoning Model + Claude API)")
security = HTTPBasic()

# Simple in-memory logs
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
    embed_model=embed_model,
)

# Use retriever interface so we can call .retrieve()
query_engine = index.as_retriever(similarity_top_k=3)

# Per-user conversational memory
USER_MEMORIES = defaultdict(lambda: ConversationBufferMemory(return_messages=True))


def query_knowledge_base(question: str) -> str:
    """
    Retrieve context from Chroma vectorstore.

    Returns:
        A concatenated text context string.
        Returns "" if no relevant information is found (no placeholder sentence).
    """
    try:
        nodes = query_engine.retrieve(question)
    except Exception as e:
        logger.error(f"Error querying vector store: {e}")
        return ""

    ctx = "\n".join(n.text for n in nodes)
    return ctx  # "" if empty


# ============================================================
# Reasoning Model Prompt (Mistral)
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


def _default_structured_reasoning() -> dict:
    """
    Default reasoning output when Mistral is unavailable or fails.
    We bias toward a structured resilience plan rather than a short answer.
    """
    return {
        "intent": "resilience_plan",
        "entities": {
            "sectors": ["energy", "water", "health", "transport", "shelter"],
            "locations": ["coastal"],
            "time_horizon": "short_term",
            "specific_locations": [],
            "disaster_type": None,
        },
        "response_mode": "structured",
    }


def generate_reasoning_with_mistral(user_question: str) -> dict:
    """
    Call Mistral to analyze the user's question and return a structured reasoning dict.

    - If the API call fails or no key is set, falls back to a structured "resilience_plan".
    - Adds a simple heuristic: if the user clearly asks for a plan/prioritization/timeline,
      we force response_mode="structured" and intent="resilience_plan".
    """
    # If no Mistral key, directly use the fallback
    if not MISTRAL_API_KEY:
        logger.warning("MISTRAL_API_KEY not set. Using default structured reasoning fallback.")
        reasoning_output = _default_structured_reasoning()
    else:
        prompt = REASONING_PROMPT.replace("{user_question}", user_question)
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
            logger.error(f"Erreur avec le reasoning model (Mistral) : {e}")
            reasoning_output = _default_structured_reasoning()

    # Heuristic override based on the natural language question
    lower_q = user_question.lower()
    if any(
        w in lower_q
        for w in [
            "plan",
            "planning",
            "prioritisation",
            "prioritization",
            "priorities",
            "detailed",
            "détail",
            "timeline",
            "roadmap",
            "strategie",
            "strategy",
        ]
    ):
        # Force structured mode for planning-like questions
        reasoning_output["response_mode"] = "structured"
        if not reasoning_output.get("intent") or reasoning_output.get("intent") == "simple_question":
            reasoning_output["intent"] = "resilience_plan"

    return reasoning_output


# ============================================================
# Claude generator
# ============================================================
def generate_with_claude(prompt: str, max_tokens: int = 64000, temperature: float = 0.7) -> str:
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
    authorization: str = Header(None),
):
    """
    Authentication helper with two modes:
    - Basic Auth (default)
    - Google OAuth (Bearer token)
    """
    if AUTH_MODE == "google":
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing Bearer token")
        token = authorization.split(" ")[1]
        return verify_google_oauth(token)

    if credentials.username != MVP_USER or credentials.password != MVP_PASS:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return credentials.username


# ============================================================
# Pydantic models
# ============================================================
class ChatRequest(BaseModel):
    question: str


# ============================================================
# Chat endpoint (with reasoning model + RAG + Claude)
# ============================================================
@app.post("/chat")
def chat(req: ChatRequest, username: str = Depends(verify_credentials)):
    memory = USER_MEMORIES[username]
    user_msg = req.question.strip()

    # 1. Reasoning model (Mistral)
    reasoning_output = generate_reasoning_with_mistral(user_msg)

    # 2. RAG context
    rag_context = query_knowledge_base(user_msg)
    rag_block = rag_context if rag_context.strip() else "<<EMPTY>>"

    # 3. Conversation history (last 5 messages)
    history = "\n".join(
        f"{m.type.capitalize()}: {m.content}" for m in memory.chat_memory.messages[-5:]
    )

    # 4. Reasoning metadata as a human-readable summary
    reasoning_metadata = f"""
### 🔍 REASONING MODEL ANALYSIS (INTERNAL METADATA)
- Intent: {reasoning_output.get('intent')}
- Response Mode: {reasoning_output.get('response_mode')}
- Sectors: {', '.join(reasoning_output.get('entities', {}).get('sectors', []))}
- Locations: {', '.join(reasoning_output.get('entities', {}).get('locations', []))}
- Specific Locations: {', '.join(reasoning_output.get('entities', {}).get('specific_locations', []))}
- Time Horizon: {reasoning_output.get('entities', {}).get('time_horizon')}
- Disaster Type: {reasoning_output.get('entities', {}).get('disaster_type')}
"""

    # 5. Full prompt for Claude (rewritten with stricter no-clarification rules)
    prompt = f"""
{reasoning_metadata}
-------------------------------------------------------------------------------
### 🔎 INPUT STRUCTURE

You will receive four blocks:

1. **Reasoning Model Output (JSON-like semantics)** — already summarized above.
2. **RAG CONTEXT** — may be very long text from local documents (infrastructure, GIS, assessments).
3. **CONVERSATION HISTORY** — last turns of the chat.
4. **CURRENT USER MESSAGE** — the question you must answer now.

---

#### RAG CONTEXT
<<<
{rag_block}
>>>

#### CONVERSATION HISTORY (last turns)
<<<
{history}
>>>

#### CURRENT USER MESSAGE
<<<
{user_msg}
>>>

---

### 🧠 CORE BEHAVIOR

You are an expert assistant for **resilience, disaster response, and crisis planning** for small islands and territories.

You must ALWAYS:

1. **Use the Reasoning Model's output as primary guidance**, but you are allowed to correct it silently if it obviously misunderstands the user.
2. **Integrate RAG CONTEXT when available**:
   - Use it to extract concrete facts (numbers, names, locations, elevations, assets, costs, timelines).
   - Prefer RAG facts over your own world knowledge when they conflict.
3. **Never ask the user for confirmation or options.**
   - You must choose the best interpretation and answer directly.
   - You can briefly state your assumptions in 1–2 sentences at the beginning if needed.

---

### 📏 RESPONSE MODE LOGIC

The Reasoning Model proposes:

- Response Mode: **{reasoning_output.get('response_mode')}**
- Intent: **{reasoning_output.get('intent')}**
- Focus Entities:
  - Sectors: {reasoning_output.get('entities', {}).get('sectors', [])}
  - Locations: {reasoning_output.get('entities', {}).get('locations', [])}
  - Specific Locations: {reasoning_output.get('entities', {}).get('specific_locations', [])}
  - Time Horizon: {reasoning_output.get('entities', {}).get('time_horizon')}
  - Disaster Type: {reasoning_output.get('entities', {}).get('disaster_type')}

You must follow this logic:

1. **If the Reasoning Model's Response Mode is `"short"`**:
   - The user is likely asking a simple, focused question.
   - Respond in **1–3 short paragraphs maximum**.
   - Natural, conversational style.

2. **If the Reasoning Model's Response Mode is `"structured"`**:
   - The user is asking for a **plan, structured analysis, or multi-sector strategy**.
   - Use the **full resilience plan structure** specified below.
   - Aim for a long, detailed answer (multi-section, multi-page). You may shorten if context/length is limited, but keep the structure.

3. **If you detect that the user explicitly asks for a detailed plan, prioritization, or timeline**, but the Reasoning Model misclassified it as `"short"`:
   - Silently override to `"structured"`.
   - Do NOT ask permission. Just provide the structured plan.

---

### 🧭 HANDLING RAG GAPS

- If the **RAG CONTEXT block equals `<<EMPTY>>`**, this means there is **no local knowledge available**.
- In that case:
  - Rely on your own domain expertise and general public knowledge about similar disasters.
  - You may briefly state:  
    *"Local knowledge base was empty; I’m using general best practices and publicly known information."*
  - Do **NOT** treat this as an error and do **NOT** ask the user to upload files.

- If RAG CONTEXT exists but lacks data for a critical sector/location:
  - Explicitly note:  
    *"No specific data found in the context for [sector/location]; using standard best practices instead."*

---

### 🌍 GEOSPATIAL & TECHNICAL EMPHASIS (FOR STRUCTURED MODE)

When in **structured** mode, your plan must:

- Integrate **elevation and exposure**: coastal flooding, landslides, salinity, storm surge heights, etc.
- Highlight **chokepoints**: bridges, passes, ports, airports, key road segments.
- Consider **lifeline infrastructure**: power, water, telecom, health, transport, fuel.
- Align actions with **time horizons** (e.g., 24h, 72h, 1 week, 1 month, 6 months, 1 year).

---

### 📘 STRUCTURED OUTPUT FORMAT
Use this structure **only when Response Mode = "structured"** (or when you override to structured as explained above).

#### I. Executive Summary
- Disaster overview (type, severity, main impacts, affected areas).
- Top 3–5 priorities across **sectors** and **locations**.
- Time framing (e.g. 0–72h, 1 week, 1 month, 6+ months).
- Mention if you are relying on general knowledge or RAG-specific information.

#### II. Context Reconstruction (Data-Driven)
- Describe the disaster and its impacts per **location** and **sector** using RAG facts when available.
- Include:
  - Physical damage (buildings, roads, ports, energy infra).
  - Population impacts (displacement, casualties, vulnerable groups).
  - Key geospatial insights (elevation bands, flood zones, exposed coasts).

#### III. Priority Matrix
Provide a small matrix (3x3 or 5x5) in markdown table form. Example:

| Asset/Sector           | Criticality | Time Sensitivity | Dependencies                | Priority |
|------------------------|------------|------------------|-----------------------------|----------|
| [Example asset]        | Very High  | 0–72h            | [Road/port/fuel dependency] | P1       |

- Priorities should align with the **time horizon** and **disaster type**.

#### IV. Geospatial Segmentation
- Identify high-risk zones and safer zones (e.g., coastal <10m, mid-elevation, uplands).
- Describe access corridors, detours, and blocked routes.
- Mention natural barriers (mountains, rivers, reefs) affecting logistics.

#### V. Sector-by-Sector Deep Dive
For each **relevant sector** (e.g., energy, water, health, shelter, transport, communication):

1. **Current State**:
   - What is damaged, what is functional.
   - Use RAG data when present; otherwise, general patterns.

2. **Short-Term Objectives (0–7 days or 0–72h)**:
   - Life-saving and stabilization actions.
   - Very concrete tasks.

3. **Medium-Term Objectives (1–4 weeks or more)**:
   - Restoration and early reconstruction.

4. **Resilience Measures**:
   - Build-back-better ideas (elevating assets, redundancy, decentralization, nature-based solutions).

#### VI. Resilience Project Portfolio
- Propose **at least 10–15 concrete projects** when possible.
- For each project, include:
  - **Title**
  - **Location / geographic tags** (coastal/urban/rural, elevation if possible)
  - **Sector(s)**
  - **Approximate timeline** (start when? duration?)
  - **Main benefits** and **resilience features** (e.g., cyclone-resistant, flood-proof, redundant routing).

#### VII. Logistics & Operations Plan
- Describe how teams, materials, and fuel move around:
  - Main ports/airports, warehouses, staging areas.
  - Temporary detours if main roads/bridges are damaged.
- Include considerations like:
  - Access to remote communities.
  - Weather windows (e.g., peak of cyclone season, rainy season).

#### VIII. Financial & Partnership Strategy
- High-level cost ranges (e.g., low/medium/high; or rough budget bands).
- Possible funding sources (government, donors, IFIs, NGOs).
- Coordination mechanisms (clusters, task forces, emergency committees).

#### IX. Risk Register
- List key **operational**, **environmental**, and **governance** risks.
- Provide mitigation options aligned with proposed projects and logistics.

#### X. Strategic Roadmap
- Phase the work over time (e.g. Phase 0: 0–72h; Phase 1: 1–4 weeks; Phase 2: 1–6 months; Phase 3: beyond 6 months).
- Summarize how the island moves from emergency response to long-term resilience.

---

### 🔒 SPECIAL RULES

1. **No Clarification Requests**:
   - Do not ask the user whether they prefer “short vs detailed”.
   - Do not ask them to choose between options (e.g. "Option A / B").
   - Always choose the best mode yourself and answer directly.

2. **Silent Correction of Reasoning Model**:
   - If the Reasoning Model clearly misunderstood the user’s intent:
     - Correct it internally.
     - Adjust sectors, locations, time horizon and disaster type yourself.
     - You may start with one clarifying sentence like:  
       *"I will focus on [X sectors] in [Y locations] over [Z horizon], based on your request."*

3. **Length Guidance**:
   - Structured mode: aim for a rich, multi-section answer. If you must shorten, keep the structure but compress each section.
   - Short mode: stay under ~200–300 words unless the question needs slightly more.

Now, answer the CURRENT USER MESSAGE accordingly.
"""

    # 6. Generate the answer with Claude
    answer = generate_with_claude(prompt)

    # 7. Update memory and logs
    memory.chat_memory.add_user_message(user_msg)
    memory.chat_memory.add_ai_message(answer)

    ACTION_LOGS.append(
        {
            "time": datetime.datetime.now().isoformat(),
            "user": username,
            "question": user_msg,
            "answer": answer,
            "context": rag_context[:500],  # truncate for logs
            "reasoning": reasoning_output,
        }
    )

    return {
        "answer": answer,
        "context_used": rag_context,
        "reasoning": reasoning_output,
        "conversation_turns": len(memory.chat_memory.messages) // 2,
    }


# ============================================================
# Reset chat history
# ============================================================
@app.delete("/chat/reset")
def reset_history(username: str = Depends(verify_credentials)):
    USER_MEMORIES[username].clear()
    return {"message": "Memory cleared."}


# ============================================================
# Simple plan generator (PDF)
# ============================================================
@app.get("/plan")
def plan(horizon: int = 24, username: str = Depends(verify_credentials)):
    if horizon not in (24, 72):
        return {"error": "Only 24h or 72h supported"}

    if horizon == 24:
        text = (
            "Plan 24h:\n"
            "- Shelter\n"
            "- Water\n"
            "- Missing persons\n"
            "- Medical care\n"
            "- Transport\n"
            "- Electricity\n"
            "- Psych support"
        )
    else:
        text = (
            "Plan 72h:\n"
            "- Restore systems\n"
            "- NGOs coordination\n"
            "- Debris removal\n"
            "- Psych support\n"
            "- Critical infra\n"
            "- Procurement"
        )

    filename = EXPORT_DIR / f"plan_{horizon}h_{uuid.uuid4().hex[:6]}.pdf"
    pdf = canvas.Canvas(str(filename), pagesize=A4)
    for i, line in enumerate(text.split("\n")):
        pdf.drawString(50, 800 - i * 20, line)
    pdf.save()

    return FileResponse(str(filename), media_type="application/pdf")


# ============================================================
# Upload documents (to be ingested into RAG offline)
# ============================================================
@app.post("/upload_doc")
def upload_doc(file: UploadFile = File(...), username: str = Depends(verify_credentials)):
    filepath = DOCS_DIR / file.filename
    with open(filepath, "wb") as f:
        f.write(file.file.read())
    # Ingestion into vector store should be done by a separate process/script
    return {"message": "File uploaded.", "filename": file.filename}


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

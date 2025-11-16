"""
FASTAPI MVP — RAG + Reasoning Model (Mistral) + Claude Sonnet 4.5 + MCP Tools
Now supporting async MCP Earth Engine data ingestion directly from /chat.
"""

import os
import csv
import datetime
import logging
import json
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict

import requests
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Header
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Anthropic Claude
from anthropic import Anthropic

# RAG (Qdrant + LlamaIndex)
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Memory
from langchain.memory import ConversationBufferMemory

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# HuggingFace login
from huggingface_hub import login

# 🔥 IMPORT MCP TOOL (async)
from mcp_server.tools.earth_engine_tool import fetch_earth_engine_data

# ============================================================
# ENV & CONFIG
# ============================================================
load_dotenv()

AUTH_MODE = os.getenv("AUTH_MODE", "basic")
MVP_USER = os.getenv("MVP_USER", "admin")
MVP_PASS = os.getenv("MVP_PASS", "password")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
if not CLAUDE_API_KEY:
    raise RuntimeError("Missing CLAUDE_API_KEY")

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
anthropic_client = Anthropic(api_key=CLAUDE_API_KEY)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "island_docs")

DOCS_DIR = Path("./docs")
DOCS_DIR.mkdir(exist_ok=True)

EXPORT_DIR = Path("./exports")
EXPORT_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mvp")

# Optional logs (for /logs endpoints if you want later)
ACTION_LOGS: list[Dict[str, Any]] = []

# ============================================================
# RAG INIT (Qdrant + LlamaIndex)
# ============================================================
logger.info("Initializing RAG with Qdrant vectorstore...")

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

qdrant_client = QdrantClient(url=QDRANT_URL)

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
    embed_model=embed_model,
)

query_engine = index.as_retriever(similarity_top_k=3)

USER_MEMORIES = defaultdict(lambda: ConversationBufferMemory(return_messages=True))

# ============================================================
# RAG helper
# ============================================================
def query_knowledge_base(question: str) -> str:
    try:
        nodes = query_engine.retrieve(question)
    except Exception as e:
        logger.error(f"Error querying vector store: {e}")
        return ""
    return "\n".join(n.text for n in nodes)


# ============================================================
# Reasoning Model Prompt (Mistral)
# ============================================================
REASONING_PROMPT = """
You are a reasoning model responsible for extracting structured parameters from
the user's message so the crisis-resilience assistant can decide whether to:

A) generate a narrative/plan, OR  
B) call the geospatial MCP tool fetch_earth_engine_data.

Your output MUST be a valid JSON object with the following structure:

{
  "intent": "simple_question" | "resilience_plan" | "technical_analysis" | "geospatial_request",
  "entities": {
    "sectors": [ ... ],
    "locations": [ ... ],
    "time_horizon": "24h" | "72h" | "short_term" | "medium_term" | "long_term" | null,
    "specific_locations": [ ... ],
    "disaster_type": "cyclone" | "earthquake" | "flood" | null,

    "dataset": string | null,
    "date": "YYYY-MM-DD" | null,
    "lon": float | null,
    "lat": float | null,
    "radius": float | null
  },
  "response_mode": "short" | "structured"
}

### Extraction rules for calls to fetch_earth_engine_data:
- The MCP tool is only eligible if:
    - A dataset name is explicitly provided
    - A date is explicitly mentioned
    - Both longitude and latitude are explicitly present
- If any of dataset/date/lon/lat is missing → DO NOT invent them → keep them null.
- Coordinates MUST be real numbers. If malformed, set to null.
- Radius: extract ONLY explicit numeric values ("within 30m", "buffer 500 meters").
- Remove units → store only the raw number.
- If radius is not mentioned, leave it null (the system will default to 10).

### Intent classification rules:
- If the user mentions coordinates, radii, buffers, or asks to retrieve geospatial values:
    → intent = "geospatial_request"
- If the user asks about assessment, analysis, impact, reconstruction, planning:
    → intent = "resilience_plan"
- If the user asks a factual or explanatory question:
    → intent = "simple_question"
- When uncertain → choose the simplest faithful interpretation.

### Date handling:
- Extract a date ONLY if explicitly present.
- Convert ALL extracted dates to ISO format YYYY-MM-DD.
- If the date cannot be parsed → set to null.

### Strict Output Rules:
- Return ONLY valid JSON.
- No extra text, no comments, no markdown.

User message:
"{user_question}"
"""


def _default_structured_reasoning() -> dict:
    return {
        "intent": "resilience_plan",
        "response_mode": "structured",
        "entities": {
            "sectors": ["energy", "water"],
            "locations": [],
            "time_horizon": "short_term",
            "specific_locations": [],
            "disaster_type": None,
            "dataset": None,
            "date": None,
            "lon": None,
            "lat": None,
            "radius": None,
        },
    }


# ============================================================
# Reasoning function (Mistral)
# ============================================================
def generate_reasoning_with_mistral(user_question: str) -> dict:
    if not MISTRAL_API_KEY:
        logger.warning("MISTRAL_API_KEY not set. Using default structured reasoning.")
        return _default_structured_reasoning()

    prompt = REASONING_PROMPT.replace("{user_question}", user_question)

    try:
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "mistral-medium",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 500,
            },
            timeout=15,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        result = json.loads(content)
    except Exception as e:
        logger.error(f"[Reasoning] Error calling Mistral: {e}")
        return _default_structured_reasoning()

    # guarantee entity structure
    result.setdefault("entities", {})
    entities = result["entities"]

    # Normalize lon / lat / radius
    for key in ["lon", "lat", "radius"]:
        val = entities.get(key)
        if val is None or val == "" or val == "null":
            entities[key] = None
        else:
            try:
                entities[key] = float(val)
            except Exception:
                entities[key] = None

    # Normalize date to YYYY-MM-DD where possible
    d = entities.get("date")
    if d:
        parsed = None
        # try ISO first
        try:
            parsed = datetime.datetime.fromisoformat(d)
        except Exception:
            # try common formats if needed
            for fmt in ("%d/%m/%Y", "%m/%d/%Y"):
                try:
                    parsed = datetime.datetime.strptime(d, fmt)
                    break
                except Exception:
                    continue
        if parsed:
            entities["date"] = parsed.strftime("%Y-%m-%d")
        else:
            entities["date"] = None
    else:
        entities["date"] = None

    return result


# ============================================================
# Claude generator
# ============================================================
def generate_with_claude(prompt: str) -> str:
    with anthropic_client.messages.stream(
        model=CLAUDE_MODEL,
        temperature=0.7,
        max_tokens=64000,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        return stream.get_final_text()


# ============================================================
# Auth & FastAPI
# ============================================================
app = FastAPI(title="Crisis RAG + MCP API")
security = HTTPBasic()


def verify_credentials(
    credentials: HTTPBasicCredentials = Depends(security),
    authorization: str = Header(None),
):
    if AUTH_MODE == "google":
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "Missing Bearer token")
        # For brevity we don't validate with Google here.
        return authorization.split(" ")[1]

    if credentials.username != MVP_USER or credentials.password != MVP_PASS:
        raise HTTPException(401, "Unauthorized")

    return credentials.username


# ============================================================
# Chat endpoint
# ============================================================
class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat(req: ChatRequest, username: str = Depends(verify_credentials)):
    memory = USER_MEMORIES[username]
    user_msg = req.question.strip()

    # 1. Reasoning
    reasoning_output = generate_reasoning_with_mistral(user_msg)
    entities = reasoning_output.get("entities", {})

    # 2. Initial RAG
    rag_context = query_knowledge_base(user_msg)
    rag_block = rag_context if rag_context.strip() else "<<EMPTY>>"

    # 3. History (last 5 messages)
    history = "\n".join(
        f"{m.type.upper()}: {m.content}"
        for m in memory.chat_memory.messages[-5:]
    )

    # 4. Reasoning metadata
    reasoning_metadata = f"""
### 🔍 REASONING MODEL ANALYSIS (INTERNAL)
Intent: {reasoning_output.get('intent')}
Response Mode: {reasoning_output.get('response_mode')}
Entities: {json.dumps(entities, ensure_ascii=False)}
"""

    # 5. Optional MCP Tool call (geospatial)
    geospatial_result = None

    if (
        reasoning_output.get("intent") == "geospatial_request"
        and entities.get("dataset") is not None
        and entities.get("lon") is not None
        and entities.get("lat") is not None
        and entities.get("date") is not None
    ):
        try:
            geospatial_result = await fetch_earth_engine_data(
                dataset=entities["dataset"],
                lon=entities["lon"],
                lat=entities["lat"],
                date=entities["date"],
                radius=int(entities.get("radius") or 10),
            )
        except Exception as e:
            logger.error(f"MCP tool failed: {e}")
            geospatial_result = {"error": f"MCP tool failed: {e}"}

        # After MCP tool updates docs + vectorstore, refresh RAG
        rag_context = query_knowledge_base(user_msg)
        rag_block = rag_context if rag_context.strip() else "<<EMPTY>>"

    # 6. Build full Claude prompt
    prompt = f"""
{reasoning_metadata}
-------------------------------------------------------------------------------
### 🔎 INPUT BLOCKS

You receive four inputs:

1. **Reasoning Model Output (summarized above)** — structured guidance about the user’s intent, sectors, locations, and time horizon.
2. **RAG CONTEXT** — text retrieved from local documents (GIS, infrastructure, reports, tables, project docs).
3. **CONVERSATION HISTORY** — the last turns of the chat with this user.
4. **CURRENT USER MESSAGE** — the question to answer now.

---

#### RAG CONTEXT
<<<
{rag_block}
>>>

#### CONVERSATION HISTORY
<<<
{history}
>>>

#### CURRENT USER MESSAGE
<<<
{user_msg}
>>>

-------------------------------------------------------------------------------
### 🎯 GLOBAL ROLE

You are **RESILIENCE-GPT**, a Crisis & Resilience Strategic Planner AI for small islands, coastal territories, and fragile states. You specialize in:

- Post-disaster damage assessment and impact mapping
- Multi-sector resilience engineering and infrastructure recovery
- Critical infrastructure prioritization (power, water, health, telecom, transport)
- Humanitarian logistics and supply-chain restoration
- GIS-informed planning and geospatial reasoning (elevation, exposure, chokepoints)
- Climate risk modelling and long-term adaptation
- Economic and financial reconstruction strategies
- Long-term resilience transformation planning (1–15 years)

You must integrate relevant information from the RAG CONTEXT when available.

-------------------------------------------------------------------------------
### 🧠 MODE SELECTION (SHORT vs STRUCTURED)

The Reasoning Model suggests:
- **Intent** = {reasoning_output.get('intent')}
- **Response Mode** = {reasoning_output.get('response_mode')}

Behavior:

1. If `response_mode = "short"` and the user is asking a simple, factual, or conceptual question:
   - Answer in 1–3 short paragraphs, conversational and clear.

2. If `response_mode = "structured"` or the user explicitly asks for a plan / strategy / roadmap / prioritization:
   - Produce a multi-section, highly detailed resilience plan.
   - Focus on prioritization and project-level detail.

You must not ask the user for clarification; choose the best interpretation and answer directly.

-------------------------------------------------------------------------------
### 🧭 RAG INTEGRATION & GAP HANDLING

- If RAG CONTEXT is non-empty: extract concrete facts and use them.
- If RAG CONTEXT is `<<EMPTY>>`: rely on best practices for similar territories.
- Explicitly state when you rely on generic assumptions.

-------------------------------------------------------------------------------
### 📘 STRUCTURED OUTPUT FORMAT (ONLY IF STRUCTURED MODE)

[... keep your detailed section structure here if you want ...]
(Executive Summary, Context Reconstruction, Priority Matrix, Sector Plans, Project Portfolio, Logistics, Finance, Risks, Roadmap.)

In short mode, answer briefly without the full structure.

Now answer the CURRENT USER MESSAGE accordingly.
"""

    # 7. Prompt length safety
    MAX_PROMPT_CHARS = 600_000
    safe_prompt = prompt[:MAX_PROMPT_CHARS]

    # 8. Ask Claude
    answer = generate_with_claude(safe_prompt)

    # 9. Memory update & logs
    memory.chat_memory.add_user_message(user_msg)
    memory.chat_memory.add_ai_message(answer)

    ACTION_LOGS.append(
        {
            "time": datetime.datetime.now().isoformat(),
            "user": username,
            "question": user_msg,
            "answer": answer,
            "context": rag_context[:500],
            "reasoning": reasoning_output,
        }
    )

    # 10. Response
    return {
        "answer": answer,
        "context_used": rag_context,
        "reasoning": reasoning_output,
        "extracted_date": entities.get("date"),
        "extracted_lon": entities.get("lon"),
        "extracted_lat": entities.get("lat"),
        "extracted_radius": entities.get("radius"),
        "geospatial_data_used": geospatial_result,
        "conversation_turns": len(memory.chat_memory.messages) // 2,
    }


# ============================================================
# Reset history
# ============================================================
@app.delete("/chat/reset")
def reset_history(username: str = Depends(verify_credentials)):
    USER_MEMORIES[username].clear()
    return {"message": "Memory cleared."}

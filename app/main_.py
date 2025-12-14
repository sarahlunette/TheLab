"""
FASTAPI MVP — RAG + Reasoning Model (Mistral) + Claude Sonnet 4.5 + MCP Tools
Now supporting async MCP Earth Engine data ingestion directly from /chat.
"""

import os
import uuid
import csv
import datetime
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any

import requests
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Header
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from collections import defaultdict

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

# 🔥 IMPORT MCP TOOL (now async!)
from mcp_server.tools.earth_engine_tool import fetch_earth_engine_data

# Load .env
load_dotenv()

# ------------------------------------------------------------
# Auth mode
# ------------------------------------------------------------
AUTH_MODE = os.getenv("AUTH_MODE", "basic")
MVP_USER = os.getenv("MVP_USER", "admin")
MVP_PASS = os.getenv("MVP_PASS", "password")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

# ------------------------------------------------------------
# Claude API
# ------------------------------------------------------------
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
anthropic_client = Anthropic(api_key=CLAUDE_API_KEY)

# ------------------------------------------------------------
# Mistral (reasoning)
# ------------------------------------------------------------
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# ------------------------------------------------------------
# Qdrant / RAG
# ------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "island_docs")

DOCS_DIR = Path("./docs")
DOCS_DIR.mkdir(exist_ok=True)

EXPORT_DIR = Path("./exports")
EXPORT_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mvp")

# ------------------------------------------------------------
# Initialize RAG
# ------------------------------------------------------------
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

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


# ------------------------------------------------------------
# Query vectorstore
# ------------------------------------------------------------
def query_knowledge_base(question: str) -> str:
    try:
        nodes = query_engine.retrieve(question)
    except:
        return ""
    return "\n".join(n.text for n in nodes)


# ------------------------------------------------------------
# Reasoning model prompt
# ------------------------------------------------------------
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


def _default_structured_reasoning():
    return {
        "intent": "resilience_plan",
        "response_mode": "structured",
        "entities": {
            "sectors": ["energy", "water"],
            "locations": [],
            "date": None,
            "lon": None,
            "lat": None,
            "radius": None,
        },
    }


# ------------------------------------------------------------
# Reasoning function (cleaned)
# ------------------------------------------------------------
def generate_reasoning_with_mistral(user_question: str) -> dict:
    if not MISTRAL_API_KEY:
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
        result = json.loads(response.json()["choices"][0]["message"]["content"])
    except:
        return _default_structured_reasoning()

    # guarantee entity structure
    result.setdefault("entities", {})
    entities = result["entities"]

    # Normalize fields
    for key in ["lon", "lat", "radius"]:
        try:
            entities[key] = float(entities.get(key)) if entities.get(key) else None
        except:
            entities[key] = None

    # Date normalization (if needed)
    d = entities.get("date")
    try:
        if d:
            parsed = datetime.datetime.fromisoformat(d)
            entities["date"] = parsed.strftime("%Y-%m-%d")
    except:
        entities["date"] = None

    return result


# ------------------------------------------------------------
# Claude generator
# ------------------------------------------------------------
def generate_with_claude(prompt: str) -> str:
    with anthropic_client.messages.stream(
        model=CLAUDE_MODEL,
        temperature=0.7,
        max_tokens=64000,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        return stream.get_final_text()


# ------------------------------------------------------------
# Auth
# ------------------------------------------------------------
app = FastAPI(title="Crisis RAG + MCP API")
security = HTTPBasic()


def verify_credentials(
    credentials: HTTPBasicCredentials = Depends(security),
    authorization: str = Header(None),
):
    if AUTH_MODE == "google":
        if not authorization:
            raise HTTPException(401, "Missing Bearer token")
        token = authorization.split(" ")[1]
        return token  # Google validation skipped in this snippet.

    if credentials.username != MVP_USER or credentials.password != MVP_PASS:
        raise HTTPException(401, "Unauthorized")

    return credentials.username


# ------------------------------------------------------------
# CHAT ENDPOINT — Fully fixed to use MCP Earth Engine tool
# ------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat(req: ChatRequest, username: str = Depends(verify_credentials)):
    memory = USER_MEMORIES[username]
    user_msg = req.question.strip()

    # 1. Reasoning model
    reasoning_output = generate_reasoning_with_mistral(user_msg)
    entities = reasoning_output["entities"]

    # 2. Initial RAG
    rag_context = query_knowledge_base(user_msg)

    # 3. MCP Tool call (if coordinates present)
    geospatial_result = None

    if (
        reasoning_output.get("intent") == "geospatial_request"
        and entities.get("lon") is not None
        and entities.get("lat") is not None
        and entities.get("date") is not None
    ):
        try:
            geospatial_result = await fetch_earth_engine_data(
                dataset="external_api_data",
                lon=entities["lon"],
                lat=entities["lat"],
                date=entities["date"],
                radius=entities.get("radius") or 10,
            )
        except Exception as e:
            geospatial_result = {"error": f"MCP tool failed: {e}"}

        # Re-query updated RAG after vectorstore rebuild
        rag_context = query_knowledge_base(user_msg)

    # 4. Build Claude prompt
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

You can handle **very long** contexts (tens of thousands of tokens) and you must integrate all relevant information from the RAG CONTEXT when it is available.

-------------------------------------------------------------------------------
### 🧠 MODE SELECTION (SHORT vs STRUCTURED)

The Reasoning Model suggests:
- **Intent** = {reasoning_output.get('intent')}
- **Response Mode** = {reasoning_output.get('response_mode')}

Your behavior:

1. If `response_mode = "short"` **and** the user is asking a simple, factual, or conceptual question:
   - Answer in **1–3 short paragraphs**, conversational and clear.
   - No heavy structure, no long report.

2. If `response_mode = "structured"` **or** the user explicitly asks for a plan / strategy / roadmap / prioritization:
   - Produce a **multi-section, highly detailed resilience plan**.
   - Use the structured format defined below.
   - Focus on **prioritization** and **project-level detail**.

3. If the Reasoning Model misclassified the question (e.g. user clearly asks for a plan but `response_mode` is "short"):
   - Silently override and use **structured mode**.
   - Do **not** ask the user for clarification.

You may briefly state your assumptions (1–2 sentences) at the top of the answer if useful.

-------------------------------------------------------------------------------
### 🧭 RAG INTEGRATION & HANDLING GAPS

- When RAG CONTEXT is **non-empty**:
  - Extract concrete facts: locations, assets, damages, capacities, existing projects, budgets, timelines, constraints.
  - Verbally reference those facts in your plan (e.g. “according to the provided report, the hospital in District A has X beds…”).
  - Prefer RAG facts over general knowledge when there is a conflict.

- When RAG CONTEXT is `<<EMPTY>>`:
  - Rely on best practices, typical island geographies, and generic disaster patterns.
  - You may mention:  
    *“Local knowledge base is empty; I will use general best practices for small islands in similar crises.”*
  - Do **not** treat this as an error.

- When RAG lacks data for a crucial sector or location:
  - Explicitly note that and use standard patterns instead (e.g. generic hospital recovery steps, generic port reopening strategies).

-------------------------------------------------------------------------------
### 🌍 GEOSPATIAL & TECHNICAL FOCUS (STRUCTURED MODE)

If you are in **structured** mode, your plan must:

- Consider **elevation bands**, floodplains, storm surge risk, landslides, erosion, and salinity.
- Identify **critical chokepoints**: bridges, passes, key road segments, airports, ports, ferry landings, fuel depots.
- Map **lifeline infrastructure**: power plants, substations, water intakes, treatment plants, main reservoirs, health facilities, telecom hubs.
- Reflect **island and archipelago constraints**: limited redundancy, dependency on ports/airports, fuel supply vulnerability, limited skilled workforce.
- Integrate **time horizons** (0–72h, 1–2 weeks, 1–3 months, 3–12 months, 1–5 years, 5–15 years).

-------------------------------------------------------------------------------
### 📘 STRUCTURED OUTPUT FORMAT (USE ONLY IN STRUCTURED MODE)

When in **structured** mode, follow this structure as closely as possible.
You can compress sections if needed, but keep all headings.

#### I. Executive Summary
- Disaster overview (type, severity, main impacts, geographic scope).
- Top 5–10 priorities across **sectors** and **locations**.
- High-level phases (0–72h, 2 weeks, 3 months, 1 year, 5+ years).
- Mention whether you rely mostly on RAG data, general knowledge, or both.

#### II. Context Reconstruction (Data-Driven)
- Describe the situation by **zones** or **regions** (e.g. coastal strip, main town, interior villages).
- Summarize:
  - Physical damage (buildings, roads, ports, airports, energy assets).
  - Population impacts (displacement, vulnerable groups, access constraints).
  - Key geospatial elements (elevation, flood zones, landslide-prone slopes, exposed coastlines).
- Use any concrete data from RAG (names, numbers, categories).

#### III. Priority Matrix
Provide a concise **priority matrix** in markdown, focusing on early decisions.

Example:

| Asset / Sector                          | Criticality | Time Sensitivity | Dependencies                      | Priority |
|-----------------------------------------|------------|------------------|-----------------------------------|----------|
| Main drinking-water system (Town A)     | Very High  | 0–72h            | Power, road access, fuel          | P1       |
| Coastal substation / mini-grid (Zone B) | High       | 1–2 weeks        | Spare parts, skilled electricians | P1       |
| Bridge over River X                     | High       | 2–8 weeks        | Heavy machinery, geotech study    | P2       |

Explain in text **why** some assets are P1 vs P2 vs P3.

#### IV. Geospatial Segmentation & Access
- Segment the territory into meaningful zones (e.g. “North coastal corridor”, “Central plateau”, “South hills”, “Outer islands”).
- For each zone, describe:
  - Exposure (flood, wind, landslide, drought, isolation).
  - Main access routes and backups.
  - Critical facilities within.

#### V. Sector-by-Sector Deep Assessment & Actions
For each relevant sector (at least: **Energy, WASH, Health, Transport, Shelter, Food Systems, Communication, Education, Environment**):

1. **Current State**
   - What is likely damaged or disrupted, using RAG facts when available.
   - Main bottlenecks and failure modes.

2. **Short-Term Objectives (0–72h / first 7 days)**
   - Life-saving actions, stabilization of critical services.
   - Concrete operations (e.g. “deploy 3 rapid-repair teams to restore 70% of low-voltage feeders in Town A”).

3. **Medium-Term Objectives (2–12 weeks)**
   - Progressive restoration and early reconstruction.
   - Temporary solutions (e.g. generators, bladders, modular bridges).

4. **Long-Term Resilience Measures (3 months – 15 years)**
   - Build-back-better interventions:
     - Elevating assets, floodproofing, cyclone-resistant standards.
     - Microgrids, distributed storage, backup water sources.
     - Nature-based solutions (mangroves, dunes, watershed restoration).
   - Regulatory and governance measures if relevant.

#### VI. Resilience Project Portfolio (10–15 Projects Minimum When Feasible)
List a **portfolio of concrete projects**. For each project, include:

- **Project ID / Title**
- **Sector(s)** (e.g. Energy + Health)
- **Location / Zone** (with any elevation, coastal, or risk descriptors from RAG if present).
- **Objective & Rationale**
- **Approximate Timeline**
  - Start window (e.g. “within 1 month”, “after 3 months”).
  - Duration estimate.
- **Key Activities**
- **Dependencies**
  - (e.g. “port access restored”, “bridge X passable”, “specialized equipment delivered”).
- **Resource Needs**
  - Workforce (profiles and rough numbers).
  - Equipment and key materials.
- **Rough Cost Band**
  - E.g. “low (<1M)”, “medium (1–10M)”, “high (>10M)”, or broad ranges in EUR/USD.
- **Resilience Co-Benefits**
  - How this project reduces future disaster risk or speeds up recovery next time.

Focus strongly on **prioritization** and **project-level clarity**.

#### VII. Logistics & Supply Chain / Operations Plan
- Describe how materials, staff, and fuel move through the territory:
  - Ports, airports, main warehouses, staging hubs.
  - Road/bridge constraints and alternate routes.
- Include:
  - How to reach remote or isolated communities.
  - How to sequence reopening of critical corridors.
  - How seasonal/weather factors (rainy season, cyclone peak, swell) affect operations.

#### VIII. Financial & Partnership Strategy
- Group projects and actions into **programs** (e.g. “Critical Lifelines Program”, “Resilient Housing Program”, “Climate-Resilient Mobility Program”).
- Indicate:
  - Rough budget envelopes for each program.
  - Potential funding sources (domestic budget, development banks, donors, NGOs, private sector).
  - Implementation modalities (government-led, NGO consortia, PPP, community-driven).

#### IX. Risk Register & Mitigation
- Identify key **operational risks** (access, weather, supply chain), **social risks** (inequitable aid, conflict), and **governance risks** (coordination failure, corruption).
- For each important risk, propose 1–3 mitigation measures.

#### X. Strategic Roadmap (Phasing)
- Structure the overall effort into phases:
  - **Phase 0: 0–72h**
  - **Phase 1: Day 3 – Week 2**
  - **Phase 2: Weeks 3–12**
  - **Phase 3: 3–12 months**
  - **Phase 4: 1–5 years**
  - **Phase 5: 5–15 years**
- Summarize for each phase:
  - Key goals.
  - Main classes of projects launched or completed.
  - Institutional / governance milestones.

-------------------------------------------------------------------------------
### 🔒 SPECIAL RULES

1. **Do NOT ask the user for clarification or to choose between options.**
   - You must infer the best interpretation and produce a direct answer.

2. **You may briefly state your assumptions at the top** (1–2 sentences) if needed.

3. **In structured mode, aim for a long, rich answer.**
   - If token limits constrain you, keep all headings but compress subpoints.

4. **In short mode**, answer in a focused way, without the full structure.

Now, based on all of the above, answer the CURRENT USER MESSAGE.
"""

    MAX_PROMPT_CHARS = 600_000
    prompt = prompt[:MAX_PROMPT_CHARS]

    # 5. Ask Claude
    answer = generate_with_claude(prompt)

    # 6. Memory update
    memory.chat_memory.add_user_message(user_msg)
    memory.chat_memory.add_ai_message(answer)

    # 7. Response
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


# ------------------------------------------------------------
# Reset history
# ------------------------------------------------------------
@app.delete("/chat/reset")
def reset_history(username: str = Depends(verify_credentials)):
    USER_MEMORIES[username].clear()
    return {"message": "Memory cleared."}

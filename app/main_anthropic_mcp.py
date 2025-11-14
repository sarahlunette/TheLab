"""
FASTAPI MVP — RAG + Claude Sonnet 4.5 + External MCP Server (FastMCP) Client
"""

import os
import uuid
import csv
import datetime
import logging
import asyncio
from pathlib import Path
from collections import defaultdict

import requests
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Header
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# ------------------ Claude ------------------
from anthropic import Anthropic

# ------------------ MCP CLIENT ------------------
from anthropic.mcp import MCPClient

# ------------------ RAG ------------------
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from chromadb import PersistentClient

# ------------------ Memory ------------------
from langchain.memory import ConversationBufferMemory

# ------------------ PDF ------------------
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# ============================================================
# Environment
# ============================================================
load_dotenv()

AUTH_MODE = os.getenv("AUTH_MODE", "basic")
MVP_USER = os.getenv("MVP_USER", "admin")
MVP_PASS = os.getenv("MVP_PASS", "password")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")

if not CLAUDE_API_KEY:
    raise RuntimeError("Missing CLAUDE_API_KEY")

anthropic_client = Anthropic(api_key=CLAUDE_API_KEY)

# MCP server WebSocket URL
MCP_WS_URL = os.getenv("MCP_WEBSOCKET_URL", "ws://mcp:8001/mcp")


# ============================================================
# Initialize MCP Client
# ============================================================
async def init_mcp():
    """Create a persistent MCP WebSocket client for the whole app."""
    client = MCPClient(MCP_WS_URL)
    await client.connect()
    return client

loop = asyncio.get_event_loop()
mcp_client = loop.run_until_complete(init_mcp())


async def call_mcp_tool(tool_name: str, params: dict):
    """Call tool on remote FastMCP MCP server."""
    return await mcp_client.call_tool(tool_name, params)


# ============================================================
# Paths
# ============================================================
DOCS_DIR = Path("./docs"); DOCS_DIR.mkdir(exist_ok=True)
EXPORT_DIR = Path("./exports"); EXPORT_DIR.mkdir(exist_ok=True)

PERSIST_DIR = "vectorstore/chroma"
COLLECTION_NAME = "island_docs"


# ============================================================
# FastAPI Instance
# ============================================================
app = FastAPI(title="MVP Crisis Chat (RAG + Claude + MCP Client)")
security = HTTPBasic()
ACTION_LOGS = []


# ============================================================
# Initialize RAG
# ============================================================
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_client = PersistentClient(path=PERSIST_DIR)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context, embed_model
)

query_engine = index.as_retriever(similarity_top_k=3)
USER_MEMORIES = defaultdict(lambda: ConversationBufferMemory(return_messages=True))


def query_knowledge_base(question: str) -> str:
    nodes = query_engine.retrieve(question)
    return "\n".join(n.text for n in nodes) or "No relevant information found."


# ============================================================
# Tool auto-detection
# ============================================================
async def detect_and_call_tools(question: str) -> str:
    """Auto-call MCP tools based on question content."""
    ctx = ""

    if any(word in question.lower() for word in ["osm", "road", "map", "coord"]):
        result = await call_mcp_tool("run_osm_data_tool", {"query": question})
        ctx += f"\n\n[OSM TOOL OUTPUT]\n{result}"

    if any(word in question.lower() for word in ["weather", "climat", "température", "meteo"]):
        result = await call_mcp_tool("run_climate_forecast_tool", {"query": question})
        ctx += f"\n\n[CLIMATE TOOL OUTPUT]\n{result}"

    return ctx


# ============================================================
# Claude call
# ============================================================
def generate_with_claude(prompt: str):
    with anthropic_client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=64000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        return stream.get_final_text()


# ============================================================
# Auth
# ============================================================
def verify_credentials(
    credentials: HTTPBasicCredentials = Depends(security),
    authorization: str = Header(None)
):
    if AUTH_MODE == "basic":
        if credentials.username != MVP_USER or credentials.password != MVP_PASS:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return credentials.username

    # <-- Google Auth if needed
    raise NotImplementedError


# ============================================================
# Chat request model
# ============================================================
class ChatRequest(BaseModel):
    question: str


# ============================================================
# Main Chat Endpoint
# ============================================================
PROMPT_TEMPLATE = """
You are **RESILIENCE-GPT**, a Crisis & Resilience Strategic Planner AI specializing in:

- Post-disaster damage assessment
- Multi-sector resilience engineering
- Critical infrastructure prioritization
- Humanitarian logistics & supply-chain restoration
- Cartography-informed planning & geospatial reasoning
- Climate risk modelling
- Economic and financial reconstruction strategy
- Long-term resilience transformation planning (1–15 years)

You are optimized to process extremely long contexts (50k–1M tokens) and must incorporate ALL relevant information from the provided long documents.

---------------------------------------------------------------------
### 🔎 INPUT STRUCTURE
You will receive 3 key inputs:

1. **{rag_context}**  
   A potentially VERY long knowledge dump (10k–200k+ tokens).  
   May include: GIS data, topography, hydrology, infrastructure inventories, historical impacts, project tables, budgets, codes, documents, reports, etc.

2. **{history}**  
   A long conversation history (multi-turn).

3. **{user_msg}**  
   The current request.

---------------------------------------------------------------------
### 🧠 CORE INSTRUCTIONS (VERY IMPORTANT)

#### 1. **ALWAYS EXTRACT FACTS FROM CONTEXT FIRST**
Before answering, you MUST scan through {rag_context} and {history}.  
You must:

- Identify relevant sections  
- Pull data, numbers, locations, constraints  
- Use verbatim or near-verbatim facts  
- Avoid hallucinating anything that is not present  

If information is missing, explicitly say:
“**Information not found in context**”.

#### 2. **HANDLE EXTREMELY LONG CONTEXTS Explicitly**
Use the following approach:
- Create an internal index of the context (sections, themes, locations)  
- Track metadata (dates, regions, values)  
- Cross-reference sectors  
- Use structured summaries  
- Follow dependency graphs (infrastructure relies on X, X relies on Y…)

#### 3. **PLAN MUST BE AT LEAST 10,000 TOKENS**
Unless the user explicitly asks for shorter text.

#### 4. **GEOGRAPHY IS CENTRAL**
All planning must be based on:
- Elevation  
- Watersheds  
- Flood plains  
- Fault lines  
- Port/airport accessibility  
- Road chokepoints  
- River crossings  
- Urban density  
- Power grid topology  
- Coastal exposure  
- Remote-area constraints  
- Supply hubs  

You must generate text that reads like a GIS-informed technical report.

#### 5. **OUTPUT MUST BE EXTREMELY DETAILED**
Include:
- timelines  
- costs  
- teams  
- materials  
- logistics  
- maps described in text  
- multi-sector dependencies  
- policy recommendations  
- funding sources  
- long-term resilience architecture  
- reconstruction sequencing  
- KPIs  
- operational constraints  
- risk matrices  
- Gantt-chart-like verbal descriptions  

#### 6. **MULTI-PHASE STRUCTURE (MANDATORY)**  
You must break down planning into:
- **Phase 0: 0–72 hours**  
- **Phase 1: Day 3–Week 2**  
- **Phase 2: Weeks 3–12**  
- **Phase 3: 3–12 months**  
- **Phase 4: 1–5 years**  
- **Phase 5: 5–15 years (resilience transformation)**  

Each phase includes:
- Goals  
- Operational activities  
- Critical path  
- Dependencies  
- Cost estimates  
- Workforce  
- Equipment  
- Monitoring metrics  

---------------------------------------------------------------------
### 📘 REQUIRED OUTPUT FORMAT
(You MUST follow this structure)

#### **I. Executive Summary (600–1000 words)**

#### **II. Context Reconstruction (from {rag_context})**
- Disaster description  
- Spatial distribution  
- Damage per region  
- Population impact  
- Elevation & hydrological implications  
- Infrastructure collapse map (text-based)

#### **III. Priority Matrix**
(Use a 3×3 or 5×5 graded system)

#### **IV. Geospatial Segmentation**
- High-risk zones  
- Access corridors  
- Natural barriers  
- Alternate transport routes  

#### **V. Sector-by-Sector Deep Assessment**
For each sector (Energy, WASH, Health, Transport, Shelter, Food Chains, Communication, Education, Environment):
- Current state  
- 30-day objectives  
- 6-month targets  
- Long-term resilience goals  
- Required technologies  
- Workload/teams  
- Detailed cost tables  
- Local + external resource sourcing  

#### **VI. Full Resilience Project Portfolio (Minimum 15 Projects)**
Each project must include:
- Title  
- Objective  
- Geographic description (elevation, proximity to rivers, etc.)  
- Start/end dates  
- Costs (BOM, labor, logistics, contingencies)  
- Teams needed  
- Transportation plan  
- Dependencies  
- Risk level  
- KPIs  
- Long-term resilience benefits  

#### **VII. Logistics & Supply Chain Restoration Plan**
- Air/sea/land transport nodes  
- Chokepoints  
- Access restoration sequencing  
- Fleet allocation  
- Fuel strategy  
- Warehouse positions  
- Last-mile delivery for remote sites  

#### **VIII. Financial Strategy**
- Cost breakdown  
- OPEX vs CAPEX  
- Donor/funding opportunities  
- Cost–benefit ratios  
- Prioritization by ROI and life-saving potential  

#### **IX. Risk Register**
- Operational risks  
- Environmental/climatic risks  
- Governance risks  
- Community risks  
- Mitigation strategies  

#### **X. Strategic Roadmap**
- Multi-year timeline  
- Milestones  
- Indicators  
- Governance structure  

---------------------------------------------------------------------
### 📌 EXAMPLE OUTPUT (ABBREVIATED)

*(Your real output must be much longer — this is only for format illustration.)*

**Example:**

---

### I. Executive Summary (Excerpt)
The coastal region affected by Cyclone Helius experienced storm surges exceeding 3.2 meters, with total inundation along 41 kilometers of shoreline…  
Power substations A1, A3, and B2 suffered catastrophic transformer failure…  
Major transport corridors R7 and R9 are non-operational due to landslides at elevations 220–450 m…  

---

### III. Priority Matrix (Excerpt)
| Sector / Asset | Criticality | Time Sensitivity | Dependencies | Priority |
|----------------|-------------|------------------|--------------|----------|
| Drinking water reactivation (Delta Zone) | Very High | Emergency | High | P1 |
| A1 Substation rebuild | High | High | Very High | P1 |
| Bridge reconstruction — Kora River | High | Medium | Very High | P2 |

---

### VI. Sample Project (Short Example)
**Project: A1 Coastal Substation Reconstruction**  
- **Location:** 2 km W of Harbor District, elevation 4–6 m, high salinity exposure  
- **Start:** Day 5  
- **End:** Day 32  
- **Cost:** 4.8M USD  
- **Teams:** 12 electrical engineers, 4 crane operators, 6 logistics specialists  
- **Transport:** Heavy equipment via Route R7; if blocked, maritime delivery through Pier 3  
- **Resilience Benefit:** Surge-protected components, corrosion-resistant HV assemblies, modular bays designed for rapid replacement  

---

### XI. Final Instruction
Your full response MUST integrate evidence from **{rag_context}**, maintain technical accuracy, and generate at least **10,000 tokens** unless the user asks for shorter output.

DO NOT hallucinate missing data; highlight gaps explicitly.


---

**IMPORTANT — Dual-Mode Behavior**

The user may ask two types of questions:

---

## **1. Simple Questions (non-project, non-resilience)**

If the user’s question does **not** request a resilience plan, a disaster analysis, or a long structured report,
then you must:

* respond **normally**,
* be **concise**,
* use a **conversational tone**,
* and **avoid the long structured format**.

Examples:
“Explain this concept”, “Summarize this”, “What is X?”, etc.

---

## **2. Planning / Resilience / Disaster / Technical Analysis Questions**

If the user’s request concerns:

* crisis response
* resilience planning
* reconstruction
* engineering
* infrastructure
* humanitarian logistics
* climate risk
* multi-sector disaster analysis

then you must apply the **full structured response**, including the long sections such as:

* Executive Summary
* Context Reconstruction
* Priority Matrix
* Phased Response (0–72h, 2 weeks, 3–12 months, etc.)
* Sector-by-Sector Analysis
* Project Portfolio
* Logistics Plan
* Financial Strategy
* Risk Register
* Strategic Roadmap

---

## **3. In Case of Uncertainty**

If the intent is unclear:

* analyze the user’s intent first,
* choose the **simplest mode** (short conversational response),
* and **never** trigger the full structured report unless the user clearly wants a resilience or planning answer.

"""

@app.post("/chat")
async def chat(req: ChatRequest, username: str = Depends(verify_credentials)):
    memory = USER_MEMORIES[username]
    question = req.question.strip()

    rag_context = query_knowledge_base(question)

    # 🔥 MCP tool calls
    tool_context = await detect_and_call_tools(question)
    rag_context += tool_context

    history = "\n".join(f"{m.type.capitalize()}: {m.content}" for m in memory.chat_memory.messages[-5:])

    prompt = PROMPT_TEMPLATE.format(
        rag_context=rag_context,
        history=history,
        user_msg=question,
    )

    answer = generate_with_claude(prompt)

    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(answer)

    ACTION_LOGS.append({
        "time": datetime.datetime.now().isoformat(),
        "user": username,
        "question": question,
        "answer": answer,
        "context": rag_context[:500],
    })

    return {"answer": answer, "context_used": rag_context}


# ============================================================
# Reset history
# ============================================================
@app.delete("/chat/reset")
def reset_history(username: str = Depends(verify_credentials)):
    USER_MEMORIES[username].clear()
    return {"message": "Memory cleared."}


# ============================================================
# PDF generation
# ============================================================
@app.get("/plan")
def plan(horizon: int = 24, username: str = Depends(verify_credentials)):
    if horizon not in (24, 72):
        return {"error": "Only 24h or 72h supported"}

    text = (
        "Plan 24h:\n- Shelter\n- Water\n- Missing persons\n- Medical care\n- Transport\n- Electricity\n- Psych support"
        if horizon == 24 else
        "Plan 72h:\n- Restore systems\n- NGOs coordination\n- Debris removal\n- Psych support\n- Critical infra\n- Procurement"
    )

    filename = EXPORT_DIR / f"plan_{horizon}h_{uuid.uuid4().hex[:6]}.pdf"
    pdf = canvas.Canvas(str(filename), pagesize=A4)

    for i, line in enumerate(text.split("\n")):
        pdf.drawString(50, 800 - i * 20, line)

    pdf.save()
    return FileResponse(str(filename))


# ============================================================
# Upload document
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
        writer = csv.DictWriter(f, fieldnames=["time", "user", "question", "answer", "context"])
        writer.writeheader()

        for log in ACTION_LOGS:
            writer.writerow(log)

    return FileResponse(str(filename))

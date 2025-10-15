"""
FastAPI App — CodeLlama Inference API
--------------------------------------
Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import torch
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = os.getenv("CODELLAMA_MODEL", "codellama/CodeLlama-7b-hf")
API_USER = os.getenv("API_USER", "admin")
API_PASS = os.getenv("API_PASS", "password")

os.environ["TRANSFORMERS_OFFLINE"] = os.getenv("TRANSFORMERS_OFFLINE", "0")  # "1" = offline mode

device = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CodeLlama-API")

# -----------------------------
# Auth setup
# -----------------------------
security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != API_USER or credentials.password != API_PASS:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials.username

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(title="CodeLlama API", version="1.0.0")
logger.info(f"🚀 Starting CodeLlama API with model '{MODEL_NAME}' on {device}")

# -----------------------------
# Load model + tokenizer
# -----------------------------
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("✅ Tokenizer loaded successfully.")
except Exception as e:
    logger.critical(f"❌ Could not load tokenizer: {e}")
    sys.exit(1)

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    )
    model.eval()
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.critical(f"❌ Could not load model: {e}")
    sys.exit(1)

# -----------------------------
# Pydantic schema
# -----------------------------
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    do_sample: bool = True

class GenerateResponse(BaseModel):
    generated_code: str

# -----------------------------
# Inference endpoint
# -----------------------------
@app.post("/generate", response_model=GenerateResponse)
def generate_code(request: GenerateRequest, username: str = Depends(verify_credentials)):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                do_sample=request.do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )

        code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"👩‍💻 User '{username}' generated {len(code)} chars of code.")
        return GenerateResponse(generated_code=code)

    except Exception as e:
        logger.error(f"❌ Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def root():
    return {"message": "✅ CodeLlama API is running", "model": MODEL_NAME, "device": device}

# ============================================================
# 1) Base image: PyTorch + CUDA + Python 3.10
# ============================================================
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# ============================================================
# 2) System setup
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# 3) Create app folder
# ============================================================
WORKDIR /app

# ============================================================
# 4) Copy requirements first for Docker caching
# ============================================================
COPY requirements.txt /app/requirements.txt

# ============================================================
# 5) Install Python dependencies (GPU optimized)
# ============================================================
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Optional: HuggingFace cache directory
ENV HF_HOME=/root/.cache/huggingface

# ============================================================
# 6) Copy the application
# ============================================================
COPY . /app

# ============================================================
# 7) Expose FastAPI port
# ============================================================
EXPOSE 8000

# ============================================================
# 8) Start the API with uvicorn
# ============================================================
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

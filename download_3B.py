from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
load_dotenv()

# Path where you want to save the model
model_dir = os.path.expanduser("models/Mistral-3B-Quantized")
token = os.getenv('HF_TOKEN')
# ------------------------
# Tokenizer
# ------------------------
model_name = "TheBloke/Mistral-3B-Instruct-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

# ------------------------
# Model (quantized)
# ------------------------
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    token=token
)
model.save_pretrained(model_dir)

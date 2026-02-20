import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. Modelo base
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 2. Ruta a tus adaptadores LoRA
ADAPTER_PATH = "modelo_chat"

# 3. Carpeta donde se guardará el modelo final
OUTPUT_PATH = "./model_nurse_final"

# Cargar tokenizer base
print("Cargando tokenizer base...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

print("Cargando modelo base TinyLlama...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32
)
# 
print("Cargando adaptadores entrenados (LoRA)...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("Fusionando modelo base + LoRA...")
model = model.merge_and_unload()

print("Guardando modelo completo...")
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print(" MODELO COMPLETO CREADO CON ÉXITO")

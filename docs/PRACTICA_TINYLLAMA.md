# Práctica: Entrenamiento y Fusión de TinyLlama Nurse

Este documento detalla el proceso completo de creación del asistente TinyLlama Nurse, desde el entrenamiento inicial hasta el chat interactivo final.

---

## 🏥 Parte 1: Entrenamiento del Modelo (Aprendizaje)

### Contexto
TinyLlama es un proyecto de código abierto que tiene como objetivo "compactar" el modelo de lenguaje Llama 2 de Meta en una versión extremadamente pequeña y eficiente. Mientras que los modelos de IA tradicionales requieren servidores masivos, TinyLlama está diseñado para funcionar en dispositivos con recursos limitados.

El objetivo es inyectar conocimientos específicos de enfermería institucional utilizando el archivo `primeros_auxilios_500.jsonl`.

### 1.1 Configuración del Entorno
Instalación de dependencias necesarias:
```bash
pip install -U transformers datasets accelerate peft sentencepiece torch
```

### 1.2 Importación de Librerías y Definiciones
```python
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_FILE = "primeros_auxilios_500.jsonl"
MAX_LENGTH = 256

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")
```

### 1.3 Tokenizer y Modelo Base
El **tokenizer** convierte el texto humano en códigos numéricos. Establecemos el `pad_token` igual al `eos_token` para manejar longitudes de secuencia consistentes.

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
model.to(device)
```

### 1.4 Configuración LoRA (Low-Rank Adaptation)
LoRA permite ajustar el modelo modificando solo una pequeña fracción de los parámetros (capas `q_proj` y `v_proj`), lo que ahorra memoria VRAM.

```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

### 1.5 Preparación de Datos
Cargamos el dataset y lo tokenizamos agregando las etiquetas de "Usuario" y "Asistente".

```python
dataset = load_dataset("json", data_files=DATA_FILE, split="train")
dataset = dataset.select(range(min(500, len(dataset))))

def tokenize(example):
    text = f"Usuario: {example['instruction']}\nAsistente: {example['output']}"
    return tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LENGTH)

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
```

### 1.6 Entrenamiento
Definimos los argumentos de entrenamiento optimizados para hardware local.

```python
training_args = TrainingArguments(
    output_dir="./tinyllama_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    fp16=True if device == "cuda" else False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()
```

### 1.7 Guardado de Adaptadores
Al final, solo guardamos los "adaptadores" (los cambios aprendidos), no el modelo completo.
```python
model.save_pretrained("modelo_chat")
tokenizer.save_pretrained("modelo_chat")
```

---

## 🔗 Parte 2: Fusión del Modelo (Merge)

En esta fase, tomamos el modelo base y le integramos permanentemente los adaptadores de enfermería.

### 2.1 Carga y Fusión
Utiliza el script `02_create_model_nurse.py`.

```python
from peft import PeftModel

# Modelo base y Adaptadores
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(base_model, "./modelo_chat")

# Fusión permanente (Merge and Unload)
model = model.merge_and_unload()

# Guardado final
OUTPUT_PATH = "./model_nurse_final"
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)
```

### 2.2 Compresión (Exportación)
Para compartir el modelo, utiliza `03_export_model_nurse.py`.

---

## 💬 Parte 3: Uso del Modelo (Chat / Inferencia)

### 3.1 Carga del Asistente
```python
MODEL_PATH = "./model_nurse_final"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32, device_map="auto")
model.eval()
```

### 3.2 Función de Chat y Bucle de Interacción
```python
def chat_enfermeria(pregunta):
    prompt = f"Eres un asistente experto en enfermería universitaria.\nUsuario: {pregunta}\nEnfermero:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, temperature=0.6, top_p=0.9, do_sample=True)
        
    respuesta = tokenizer.decode(output[0], skip_special_tokens=True)
    return respuesta.split("Enfermero:")[-1].strip()

while True:
    p = input("Tú: ")
    if p.lower() in ["salir", "exit"]: break
    print(f"\nChat-Enfermería: {chat_enfermeria(p)}\n")
```

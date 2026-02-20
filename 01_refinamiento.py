# Refinamiento del modelo TinyLlama-1.1B-Chat-v1.0 con un conjunto de datos específico
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

# Configuración del modelo y los datos para el refinamiento
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_FILE = "primeros_auxilios_500.jsonl"
MAX_LENGTH = 256

# Comprobar el dispositivo (GPU o CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("=========================================")
print("Usando dispositivo:", device.upper())
if device == "cuda":
    print("Tarjeta gráfica detectada:", torch.cuda.get_device_name(0))
print("=========================================\n")

# Cargar el tokenizador y configurar el token de padding
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Cargar el modelo base en precisión media (fp16) para ahorrar VRAM
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
model.to(device)

# Configuración de LoRA (entrena solo una fracción de los parámetros)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print("\nParámetros a entrenar:")
model.print_trainable_parameters()
print("\n")

# Cargar el conjunto de datos desde el archivo JSONL
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

# Seleccionar las muestras (con límite de seguridad al tamaño real del dataset)
dataset = dataset.select(range(min(500, len(dataset))))

# Función para tokenizar el texto con el formato Usuario/Asistente
def tokenize(example):
    text = f"Usuario: {example['instruction']}\nAsistente: {example['output']}"
    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

# Aplicar tokenización
tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# Configurar el data collator para el entrenamiento
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Configurar los argumentos de entrenamiento (Optimizados para tu RTX 2060)
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

# Crear el entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Iniciar el proceso de refinamiento
print("Iniciando el entrenamiento... ¡Paciencia y a dejar trabajar a la gráfica!")
trainer.train()

# Guardar el modelo refinado
model.save_pretrained("modelo_chat")
tokenizer.save_pretrained("modelo_chat")
print("\n¡Éxito! Modelo refinado guardado correctamente en la carpeta 'modelo_chat'")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "model_nurse_final"

print("Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

print("Cargando modelo de enfermería...")
# detectar si hay GPU y elegir dtype apropiado (evita usar el parámetro deprecated `torch_dtype`)
_use_cuda = torch.cuda.is_available()
_dtype = torch.float16 if _use_cuda else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=_dtype,
    device_map="auto" if _use_cuda else None,
)

model.eval()
print(" Modelo cargado correctamente")



def chat_enfermeria(pregunta, max_tokens=200):
    prompt = f"""
Eres un asistente experto en enfermería universitaria.
Respondes de forma clara, técnica y pedagógica.

Usuario: {pregunta}
Enfermero:
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    # mover inputs al mismo dispositivo donde está el modelo (evita mezcla cpu/cuda)
    try:
        model_device = model.device
    except Exception:
        model_device = next(model.parameters()).device
    inputs = inputs.to(model_device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    texto = tokenizer.decode(output[0], skip_special_tokens=True)
    respuesta = texto.split("Enfermero:")[-1].strip()
    return respuesta
print("Escribe 'salir' para terminar\n")

while True:
    pregunta = input("Tú: ")

    if pregunta.lower() in ["salir", "exit", "quit"]:
        print("Hasta luego 👋")
        break

    respuesta = chat_enfermeria(pregunta)
    print(f"\nChat-Enfermería: {respuesta}\n")

# 🩺 TinyLlama Nurse: Guía Completa de IA para Enfermería

<p align="center">
  <img src="https://img.shields.io/badge/Model-TinyLlama--1.1B-orange?style=for-the-badge&logo=huggingface" alt="Model">
  <img src="https://img.shields.io/badge/Domain-Nursing%20Protocols-red?style=for-the-badge" alt="Domain">
  <img src="https://img.shields.io/badge/Platform-Windows%20Local-lightgrey?style=for-the-badge&logo=windows" alt="Platform">
  <img src="https://img.shields.io/badge/Optimized-RTX%20GPU-green?style=for-the-badge" alt="GPU">
</p>

---

## 📖 Introducción y Contexto

**TinyLlama Nurse** es una versión especializada del modelo de lenguaje de código abierto **TinyLlama-1.1B**. Su propósito es funcionar como un asistente experto en enfermería universitaria, capaz de responder de forma técnica y pedagógica sobre protocolos de salud y primeros auxilios.

---

## 🚀 Guía de Ejecución (Orden Recomendado)

Sigue los scripts en orden numérico para completar el proceso:

### ⚙️ Paso 0: Preparación
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 🧠 Paso 1: Entrenamiento (Fine-Tuning)
*   **Script**: `01_refinamiento.py`
*   **Función**: Inyecta 500 casos de enfermería al modelo usando LoRA.
*   **Resultado**: Crea la carpeta de adaptadores `./modelo_chat`.

### 🔗 Paso 2: Fusión (Merge Weights)
*   **Script**: `02_create_model_nurse.py`
*   **Función**: Une el modelo base con el conocimiento de enfermería.
*   **Resultado**: Crea el modelo final en `./model_nurse_final`.

### 📦 Paso 3: Exportación (ZIP)
*   **Script**: `03_export_model_nurse.py`
*   **Función**: Comprime el modelo en un archivo ZIP de alta capacidad (Zip64).
*   **Resultado**: Genera `model_nurse_final.zip`.

---

## 📂 Organización del Proyecto

| Archivo / Carpeta | Propósito |
| :--- | :--- |
| `01_refinamiento.py` | Script de entrenamiento LoRA (Inicio). |
| `02_create_model_nurse.py` | Script para fusionar los pesos del modelo. |
| `03_export_model_nurse.py` | Exportador con barra de progreso y soporte >4GB. |
| `medalpaca_training/` | **Contenido Médico Avanzado** (ver sección abajo). |
| `docs/PRACTICA_TINYLLAMA.md` | Documentación técnica completa y académica. |
| `model_nurse_final/` | Directorio con el modelo final fusionado. |
| `modelo_chat/` | Adaptadores generados tras el entrenamiento. |

---

## 🏥 Sección Especial: Medalpaca Training
La carpeta `medalpaca_training/` contiene herramientas para un entrenamiento médico mucho más profundo y masivo.

*   **¿Qué hace?**: Permite trabajar con datasets de medicina en español de más de 130,000 registros.
*   **Contenido**:
    *   `descargar_medalpaca.py`: Descarga y procesa datasets médicos.
    *   `refinamiento_medalpaca.py`: Entrenamiento masivo optimizado para GPUs con `gradient_checkpointing`.
*   **Uso**: Ideal si buscas un nivel de conocimiento médico profesional más allá de los primeros auxilios.

---

## 🛠️ Requisitos
*   **GPU**: NVIDIA (6GB VRAM mínimo).
*   **Software**: Python 3.10+, PyTorch con CUDA.
*   **Almacenamiento**: ~10GB libres.

---

<p align="center">
  <i>Iniciativa de formación en IA y Salud - Universidad</i>
</p>

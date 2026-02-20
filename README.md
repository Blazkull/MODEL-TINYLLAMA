# 🩺 TinyLlama Nurse: Sistema de Entrenamiento y Fusión de IA Médica

<p align="center">
  <img src="https://img.shields.io/badge/Model-TinyLlama--1.1B--Chat-orange?style=for-the-badge&logo=huggingface" alt="Model">
  <img src="https://img.shields.io/badge/Optimization-LoRA%20(PEFT)-blue?style=for-the-badge" alt="Optimization">
  <img src="https://img.shields.io/badge/Domain-Enfermería%20y%20Salud-red?style=for-the-badge" alt="Domain">
</p>

Este repositorio contiene un entorno completo para el refinamiento local del modelo **TinyLlama-1.1B** mediante técnicas de **Fine-Tuning con LoRA**, orientado específicamente a la creación de un asistente virtual para enfermería universitaria.

---

## 🚀 Flujo de Trabajo y Scripts detallados

A continuación, se detalla la lógica interna de cada componente del sistema en su orden de ejecución:

### 🧠 1. Entrenamiento: `01_refinamiento.py`
Este script es el corazón del proceso de aprendizaje. Su función es "inyectar" conocimiento específico sin alterar drásticamente la estructura del modelo original.

*   **Logic & Hardware**:
    *   Detecta automáticamente si cuentas con una **GPU NVIDIA (CUDA)**. Si se detecta, activa la precisión de punto flotante de 16 bits (`fp16=True`), lo que duplica la velocidad y reduce el consumo de VRAM.
    *   Carga el modelo base en formato de 16 bits (`torch.float16`) para optimizar recursos en arquitecturas como la RTX 2060.
*   **Técnica LoRA (PEFT)**:
    *   Configura una matriz de bajo rango (`r=8`, `lora_alpha=16`) que se enfoca exclusivamente en las capas de atención `q_proj` y `v_proj`. Esto permite que solo se entrenen unos pocos millones de parámetros, ahorrando gigabytes de memoria.
*   **Procesamiento de Datos**:
    *   Lee el archivo `primeros_auxilios_500.jsonl` y formatea cada entrada como un diálogo `Usuario / Asistente`.
    *   Aplica un límite de **256 tokens** por secuencia para mantener la eficiencia.

### 🔗 2. Fusión de Pesos: `02_create_model_nurse.py`
Cuando el entrenamiento termina, los cambios (adaptadores) viven en una carpeta separada. Este script se encarga de integrarlos permanentemente al modelo base.

*   **Proceso de Merge**:
    *   Carga el modelo base limpio en un espacio de memoria de 32 bits (`float32`) para garantizar la máxima fidelidad durante la fusión.
    *   Carga los adaptadores aprendidos desde la carpeta `./modelo_chat`.
    *   Ejecuta el comando `model.merge_and_unload()`, que realiza una suma matemática de los pesos de los adaptadores sobre los pesos originales.
*   **Resultado**: Crea un modelo "standalone" en `./model_nurse_final` que ya no requiere de la librería `PEFT` para funcionar, siendo mucho más rápido en inferencia.

### 📦 3. Empaquetado y Exportación: `03_export_model_nurse.py`
Debido a que los modelos de lenguaje pueden pesar varios gigabytes, un script de compresión estándar podría fallar o tardar demasiado sin dar feedback.

*   **Compresión Inteligente**:
    *   **Modo Granular**: Primero escanea todos los archivos del modelo para calcular el peso total exacto.
    *   **Barra de Progreso (tqdm)**: Muestra en tiempo real cuántos Gigabytes se han comprimido y a qué velocidad (MB/s).
    *   **Soporte Large File**: Utiliza el estándar **Zip64** y lectura por bloques (chunks de 1MB) para manejar archivos de más de 4GB sin saturar la memoria RAM del sistema.

---

## 🏥 Especialización Médica: `medalpaca_training/`

Esta carpeta contiene una versión "Premium" de los scripts para investigadores que deseen llevar el modelo a un nivel de conocimiento médico profesional.

### 📥 Procesador: `descargar_medalpaca.py`
*   Conecta con el Hugging Face Hub para descargar el dataset `somosnlp/spanish_medica_llm`.
*   **Formateo Inteligente**: Clasifica automáticamente si el dato es un caso clínico (clinic_case) o una pregunta médica simple, asignándole un "System Prompt" adecuado para guiar la respuesta de la IA.

### 🌡️ Entrenador Maestro: `refinamiento_medalpaca.py`
*   Diseñado para procesar más de **130,000 registros médicos**.
*   **Optimizaciones Extremas**: 
    *   Activa `gradient_checkpointing_enable()`, lo que permite entrenar modelos grandes en GPUs con poca memoria a cambio de un ligero coste en velocidad de CPU.
    *   Usa un `lr_scheduler_type="cosine"`, que reduce la velocidad de aprendizaje de forma suave, permitiendo que el modelo aprenda detalles médicos finos sin "olvidar" lo anterior.
    *   Aumenta la longitud de contexto a **512 tokens**.

---

## 🛠️ Requisitos del Sistema
*   **Sistema Operativo**: Windows 10/11 con PowerShell.
*   **Entorno**: Python 3.10 o superior (recomendado 3.11).
*   **Hardware**: 
    *   Mínimo: 16GB RAM + CPU.
    *   **Recomendado**: NVIDIA GPU con 6GB+ VRAM (ej. RTX 2060, 3060, 4060).
*   **Librerías Críticas**: `transformers`, `torch` (con soporte CUDA), `peft`, `datasets`, `tqdm`.

---

## 📂 Árbol de Archivos Importante

```text
E:\IA-UNIVERSIDAD\TINYLLAMA\
├── 01_refinamiento.py        <-- Entrenamiento de Enfermería (PASO 1)
├── 02_create_model_nurse.py  <-- Fusión de Modelo Final (PASO 2)
├── 03_export_model_nurse.py  <-- Compresor con barra de progreso (PASO 3)
├── requirements.txt          <-- Dependencias necesarias
├── docs/
│   └── PRACTICA_TINYLLAMA.md <-- Guía académica detallada
└── medalpaca_training/       <-- Módulo avanzado de medicina masiva
    ├── descargar_medalpaca.py
    └── refinamiento_medalpaca.py
```

---
<p align="center">
  <i>Desarrollado para la formación académica en Inteligencia Artificial y Ciencias de la Salud.</i>
</p>

## Créditos y contacto
- Proyecto realizado por Jhoan Acosta - Blazkull.
- Para dudas o mejoras, abre un issue o contacta al autor.

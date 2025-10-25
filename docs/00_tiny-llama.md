# Ficha Técnica — TinyLlama-1.1B-Chat-v1.0

## Información General

- **Nombre del modelo:** TinyLlama-1.1B-Chat-v1.0
- **Repositorio:** [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- **Autor / Organización:** jzhang38 / TinyLlama Project
- **Tamaño del modelo:** ~1.1 mil millones de parámetros
- **Arquitectura base:** LLaMA 2 (Mismo tokenizer y arquitectura)
- **Tipo:** Modelo de lenguaje causal (Causal Language Model)
- **Licencia:** Apache 2.0
- **Idioma principal:** Inglés
- **Uso previsto:** Chat conversacional, generación de texto, QA básica, asistentes ligeros

## Descripción del Proyecto

TinyLlama es un proyecto de preentrenamiento de un modelo compacto de **1.1B parámetros** basado en la arquitectura **LLaMA 2**.
Su objetivo es ofrecer un modelo eficiente en memoria y cómputo que pueda integrarse fácilmente en entornos con recursos limitados (GPU pequeñas, dispositivos locales o entornos educativos).

El modelo **TinyLlama-1.1B-Chat-v1.0** es una versión ajustada (fine-tuned) para tareas conversacionales.
Fue entrenado inicialmente con un subconjunto del dataset **UltraChat**, compuesto por diálogos sintéticos generados por ChatGPT, y posteriormente alineado con el dataset **UltraFeedback** mediante **DPOTrainer (Direct Preference Optimization)**.

## Entrenamiento

- **Modelo base:** TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
- **Receta de ajuste:** basada en Zephyr (Hugging Face TRL)
- **Datasets de entrenamiento:**
  - UltraChat (diálogos sintéticos)
  - UltraFeedback (64k prompts con rankings de GPT-4)
- **Objetivo:** Optimizar la coherencia conversacional y la alineación con preferencias humanas.

## Requerimientos del Entorno

- **Transformers:** `>=4.34`
- **Accelerate:** recomendado para inferencia eficiente.
- **Hardware mínimo sugerido:**

  - GPU con ≥8 GB VRAM (modo 4-bit)
  - Soporte para `torch.bfloat16` o `torch.float16`

## Parámetros Importantes de Inferencia

| Parámetro        | Descripción                           | Valor recomendado |
| ---------------- | ------------------------------------- | ----------------- |
| `max_new_tokens` | Longitud máxima de salida generada    | 256–512           |
| `temperature`    | Aleatoriedad en la generación         | 0.7               |
| `top_k`          | Número de tokens candidatos           | 50                |
| `top_p`          | Probabilidad acumulada para muestreo  | 0.95              |
| `torch_dtype`    | Precisión del tensor                  | `torch.bfloat16`  |
| `device_map`     | Distribución del modelo en hardware   | `"auto"`          |
| `load_in_4bit`   | Reducción de memoria con quantización | `True` (opcional) |

## Ejemplo de Uso

```python
from transformers import pipeline
import torch

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

messages = [
    {"role": "system", "content": "You are a friendly chatbot."},
    {"role": "user", "content": "Explain what a transformer is in simple terms."},
]

prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=200, temperature=0.7)
print(outputs[0]["generated_text"])
```

## Capacidades y Limitaciones

**Capacidades:**

- Comprensión y generación de texto en inglés.
- Conversaciones multi-turno básicas.
- Instrucciones generales y explicaciones sencillas.
- Compatible con formato de chat template de `transformers`.

**Limitaciones:**

- No maneja contextos extensos (limitado a ~4k tokens).
- No soporta razonamiento complejo o multitarea avanzada.
- Desempeño limitado fuera del inglés.
- Puede producir respuestas inconsistentes o poco precisas en tareas técnicas.

## Notas Técnicas

- Compatible con `bitsandbytes` para carga en 4-bit.
- Puede ejecutarse localmente en GPU de gama media o en Google Colab.
- Diseñado para pruebas de inferencia ligera, desarrollo de prototipos y enseñanza de LLMs.

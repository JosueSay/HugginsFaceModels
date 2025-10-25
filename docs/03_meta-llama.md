# Ficha Técnica — Meta-Llama-2-7B-hf

## Información General

- **Nombre del modelo:** Llama-2-7B-hf
- **Repositorio:** [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- **Desarrollador:** Meta Platforms, Inc.
- **Serie:** Llama 2
- **Tamaño del modelo:** 7 mil millones de parámetros (7B)
- **Tipo de modelo:** Modelo generativo preentrenado (Causal Language Model)
- **Variantes disponibles:** 7B, 13B y 70B (pretrained y fine-tuned “chat”)
- **Licencia:** [Llama 2 Community License Agreement](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)
- **Idioma principal:** Inglés
- **Uso previsto:** Investigación y desarrollo comercial limitado bajo los términos de la licencia Meta.
- **Fecha de entrenamiento:** Enero 2023 – Julio 2023

## Descripción del Modelo

**Llama 2** es una familia de modelos de lenguaje de código abierto desarrollada por **Meta AI**, diseñada para generación de texto y aplicaciones conversacionales.
Los modelos fueron entrenados desde cero utilizando una mezcla de **2 billones de tokens** de fuentes públicas en línea, excluyendo cualquier dato de usuarios de Meta.

Las variantes “chat” fueron afinadas con técnicas de:

- **SFT (Supervised Fine-Tuning)** sobre datasets de instrucciones públicas.
- **RLHF (Reinforcement Learning with Human Feedback)** con más de **1 millón de ejemplos anotados manualmente**.

El modelo **Llama-2-7B-hf** corresponde a la versión preentrenada base, convertida al formato **Hugging Face Transformers**.

## Arquitectura

- **Tipo:** Transformer autoregresivo.
- **Capas:** 32.
- **Contexto máximo:** 4,096 tokens.
- **Mecanismo de atención:** Self-Attention estándar (sin GQA en 7B/13B, GQA presente en 70B).
- **Normalización:** RMSNorm.
- **Activación:** SwiGLU.
- **Embedding compartido:** Sí (word embedding atado).
- **Tokenización:** Byte-Pair Encoding (BPE).

## Entrenamiento

| Atributo                            | Valor                                                                      |
| ----------------------------------- | -------------------------------------------------------------------------- |
| **Tokens totales de entrenamiento** | 2.0T                                                                       |
| **Fuentes de datos**                | Mezcla de corpus públicos y sintéticos (no incluye datos de usuarios Meta) |
| **Batch global**                    | 4 millones de tokens                                                       |
| **Duración estimada (7B)**          | 184,320 horas GPU                                                          |
| **Hardware usado**                  | NVIDIA A100 80GB                                                           |
| **Emisiones estimadas (7B)**        | 31.22 tCO₂eq (compensadas al 100%)                                         |

El entrenamiento y fine-tuning se realizaron en el **Meta Research Super Cluster** y en infraestructura de nube de terceros.

## Desempeño y Evaluación

Comparaciones de desempeño en benchmarks académicos (valores aproximados):

| Categoría                   | Llama 1 7B | Llama 2 7B | Llama 2 Chat 7B |
| --------------------------- | ---------- | ---------- | --------------- |
| **Commonsense Reasoning**   | 60.8       | 63.9       | —               |
| **Reading Comprehension**   | 58.5       | 61.3       | —               |
| **World Knowledge**         | 46.2       | 48.9       | —               |
| **MATH (GSM8K/MATH)**       | 6.95       | 14.6       | —               |
| **TruthfulQA (%)**          | 27.4       | 33.3       | 57.0            |
| **ToxiGen (toxicidad %) ↓** | 23.0       | 21.3       | 0.0             |

> Las versiones **Llama-2-Chat** muestran mejoras significativas en seguridad y utilidad, superando a modelos abiertos equivalentes.

## Uso Previsto

**Casos de uso recomendados:**

- Aplicaciones de conversación (“chatbots”) con formato estructurado.
- Generación y resumen de texto en inglés.
- Traducción y reformulación básica.
- Adaptación mediante fine-tuning o prompt engineering.

**Fuera de alcance:**

- Uso en otros idiomas distintos del inglés.
- Aplicaciones médicas, legales, financieras o críticas.
- Entrenamiento o mejora de otros LLMs (prohibido por licencia).
- Cualquier uso que viole el [Acceptable Use Policy](https://ai.meta.com/llama/use-policy).

## Ejemplo de Uso

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "Explain the concept of reinforcement learning in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, top_p=0.9)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Requerimientos del Entorno

- **Transformers:** `>=4.33`
- **Torch:** `>=2.0`
- **Hardware recomendado:**
  - GPU ≥24 GB VRAM (A100 o RTX 4090 para carga completa)
  - Opcional: `bitsandbytes` para ejecución en 4-bit o 8-bit
- **Acceso:** requiere aceptar la licencia en el sitio de Meta y registro de contacto (nombre, afiliación, país, correo electrónico).

## Licencia

**Tipo:** Llama 2 Community License
**Características principales:**

- Licencia gratuita, no exclusiva, no transferible.
- Se permite uso comercial, con restricción para empresas >700 millones de usuarios mensuales.
- No se permite usar los modelos ni sus resultados para entrenar otros LLMs.
- Obligatorio incluir el aviso:

  ```bash
  Llama 2 is licensed under the LLAMA 2 Community License, 
  Copyright (c) Meta Platforms, Inc. All Rights Reserved.
  ```

- Obligatorio cumplir con la [Acceptable Use Policy](https://ai.meta.com/llama/use-policy).

## Consideraciones Éticas y Limitaciones

- Modelo entrenado únicamente en inglés; desempeño reducido en otros idiomas.
- Puede generar información inexacta o sesgada.
- No apto para decisiones críticas o entornos regulados.
- Se recomienda realizar pruebas de seguridad y filtrado antes de despliegue.

## Referencias y Recursos

- **Documentación oficial:** [ai.meta.com/llama](https://ai.meta.com/llama/)
- **Licencia:** [Llama 2 Community License](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)
- **Artículo técnico:** *Llama 2: Open Foundation and Fine-Tuned Chat Models (Meta AI, 2023)*

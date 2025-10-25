# Ficha Técnica — Qwen2.5-0.5B-Instruct

## Información General

* **Nombre del modelo:** Qwen2.5-0.5B-Instruct
* **Repositorio:** [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
* **Autor / Organización:** Qwen Team — Alibaba Cloud
* **Serie:** Qwen2.5
* **Tamaño del modelo:** ~0.49 mil millones de parámetros (0.36B no embebidos)
* **Arquitectura base:** Transformer con RoPE, SwiGLU, RMSNorm, Attention QKV bias y embeddings compartidos
* **Tipo:** Causal Language Model (Instruction-tuned)
* **Idiomas soportados:** Multilingüe (29 idiomas, incluyendo inglés, chino, español, francés, alemán, portugués, japonés, coreano, árabe, vietnamita, tailandés, ruso, entre otros)
* **Licencia:** Qwen License (ver en repositorio)
* **Uso previsto:** Asistentes conversacionales, generación estructurada, razonamiento básico, codificación y tareas de instrucción

## Descripción del Modelo

**Qwen2.5** es la nueva generación de modelos de lenguaje desarrollados por **Alibaba Cloud**, diseñada para mejorar las capacidades de su predecesor Qwen2.
La versión **0.5B-Instruct** es la variante más ligera y ajustada con instrucciones, ideal para entornos con recursos limitados, tareas de evaluación y despliegues locales o educativos.

Entre las mejoras más destacadas frente a Qwen2:

* Mayor conocimiento general y especializado en **matemáticas y programación**.
* Mejor seguimiento de instrucciones y generación de texto largo (hasta 8K tokens generados).
* Soporte para **contextos extendidos de hasta 128K tokens**.
* Mejor manejo de datos estructurados (tablas, JSON) y prompts complejos (sistema + usuario).
* Ampliado soporte **multilingüe** con consistencia en respuestas entre idiomas.

## Especificaciones Técnicas

| Atributo                          | Valor                                     |
| --------------------------------- | ----------------------------------------- |
| **Número de parámetros totales**  | 0.49B                                     |
| **Parámetros no embebidos**       | 0.36B                                     |
| **Número de capas**               | 24                                        |
| **Cabezas de atención (GQA)**     | 14 para Q, 2 para KV                      |
| **Contexto máximo de entrada**    | 32,768 tokens                             |
| **Longitud máxima de generación** | 8,192 tokens                              |
| **Soporte extendido**             | Hasta 128K tokens (variante base Qwen2.5) |
| **Framework**                     | PyTorch / Transformers                    |
| **Compatibilidad**                | `transformers >= 4.37.0`                  |

## Entrenamiento

* **Etapas:** Preentrenamiento + Postentrenamiento (Instruct tuning).
* **Objetivo:** Mejorar el rendimiento en tareas instruccionales, multilingües y de formato estructurado.
* **Técnicas empleadas:** Instrucción supervisada (SFT) y afinamiento con datasets de prompts y respuestas humanos.
* **Hardware utilizado:** Clúster de GPUs (escala industrial, no especificado por modelo).

## Capacidades y Usos

**Capacidades:**

* Comprensión y generación multilingüe coherente.
* Seguimiento avanzado de instrucciones complejas.
* Generación de texto estructurado (JSON, tablas, listas, código).
* Soporte robusto para prompts de múltiples roles (`system`, `user`, `assistant`).
* Eficiencia alta en entornos de inferencia ligera.

**Limitaciones:**

* Puede presentar degradación en calidad al procesar contextos >32K tokens.
* No incluye filtros de seguridad moral ni alineamiento ético avanzado (RLHF limitado).
* Razonamiento profundo o factual limitado por el tamaño (0.5B).
* Rendimiento inferior frente a variantes mayores (Qwen2.5-7B o 14B).

## Ejemplo de Uso

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language models."

messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated = model.generate(**inputs, max_new_tokens=512)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated)]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
```

## Requerimientos del Entorno

* **Transformers:** `>=4.37.0`
* **Torch:** `>=2.1` (con soporte `torch_dtype="auto"`)
* **Hardware recomendado:**

  * CPU o GPU con ≥6 GB VRAM
  * Inferencia posible en GPU media o CPU moderna (menor velocidad)
* **Opcional:** `accelerate`, `bitsandbytes` para carga optimizada en 4-bit o 8-bit

## Evaluación y Rendimiento

Los resultados de evaluación detallados se publicaron en el [blog oficial de Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/), mostrando mejoras sustanciales en:

* Razonamiento matemático y programación.
* Comprensión de instrucciones complejas.
* Desempeño multilingüe.
* Eficiencia de inferencia y throughput.

## Citas

```bibtex
@misc{qwen2.5,
    title = {Qwen2.5: A Party of Foundation Models},
    url = {https://qwenlm.github.io/blog/qwen2.5/},
    author = {Qwen Team},
    month = {September},
    year = {2024}
}

@article{qwen2,
    title={Qwen2 Technical Report},
    author={An Yang et al.},
    journal={arXiv preprint arXiv:2407.10671},
    year={2024}
}
```

---

¿Quieres que te prepare este documento como archivo `.md` descargable (`Qwen2.5-0.5B-Instruct.md`) o seguimos con **Meta Llama (Llama-2-7B-hf)**?

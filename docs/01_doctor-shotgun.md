# Ficha Técnica — Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct (Norobara-ZLoss-8x7B)

## Información General

- **Nombre del modelo:** Norobara-ZLoss-8x7B
- **Repositorio:** [Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct](https://huggingface.co/Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct)
- **Autor / Organización:** Doctor-Shotgun
- **Tamaño del modelo:** 1.1 mil millones de parámetros
- **Arquitectura base:** TinyLlama (compatible con LLaMA 2)
- **Tipo:** Modelo de lenguaje causal con ajuste instructivo (instruct-tuned)
- **Idioma principal:** Inglés
- **Uso previsto:** Decodificación especulativa, experimentación en generación de texto, tareas de instrucción multi-turno
- **Licencia:** No especificada explícitamente (presumiblemente similar a TinyLlama base: Apache 2.0)

## Descripción del Modelo

El modelo **Norobara-ZLoss-8x7B** es una versión ajustada del **TinyLlama-1.1B-32k**, optimizada para **instrucciones multi-turno** y diseñada principalmente para **decodificación especulativa** (speculative decoding).
El formato de instrucción sigue un esquema similar al de **Alpaca**, adaptado para diálogos iterativos de usuario-modelo.

El modelo conserva la arquitectura base de TinyLlama (1.1B parámetros), pero incorpora entrenamiento adicional sobre conjuntos de datos instructivos de código abierto. Está orientado a pruebas y experimentación en contextos de investigación o desarrollo de prototipos ligeros.

## Formato de Entrada Esperado

El modelo utiliza un formato estructurado basado en **bloques de instrucción y respuesta**:

```bash
### Instruction:
{prompt del sistema}

### Input:
{mensaje del usuario}

### Response:
{respuesta generada}

### Input:
{siguiente mensaje del usuario}

### Response:
{respuesta generada}
```

Este formato permite mantener coherencia en conversaciones multi-turno sin depender de plantillas de chat de `transformers`.

## Entrenamiento

- **Modelo base:** TinyLlama-1.1B-32k
- **Duración del entrenamiento:** 3 épocas (~3.5 horas)
- **Hardware utilizado:** 1 GPU A100
- **Tipo de ajuste:** Fine-tuning completo (no LoRA)
- **Datasets utilizados:** Conjuntos de instrucciones open-source (no especificados)
- **Propósito:** Mejorar rendimiento en generación de texto instructiva y soporte a contextos extendidos (hasta 32k tokens).
- **Duración de contexto:** ~32,000 tokens

## Requerimientos del Entorno

- **Transformers:** `>=4.34`
- **Accelerate:** recomendado
- **Bitsandbytes:** opcional (para inferencia en 4-bit)
- **Hardware sugerido:**

  - GPU con ≥8 GB VRAM (modo cuantizado)
  - Soporte para `torch.bfloat16` o `torch.float16`

## Parámetros Relevantes de Inferencia

| Parámetro        | Descripción                             | Valor recomendado |
| ---------------- | --------------------------------------- | ----------------- |
| `max_new_tokens` | Longitud máxima de texto generado       | 256–512           |
| `temperature`    | Controla la aleatoriedad de salida      | 0.7               |
| `top_k`          | Número de candidatos considerados       | 50                |
| `top_p`          | Probabilidad acumulativa para muestreo  | 0.95              |
| `torch_dtype`    | Tipo de precisión numérica              | `torch.bfloat16`  |
| `device_map`     | Asignación automática del modelo        | `"auto"`          |
| `load_in_4bit`   | Activar cuantización para menor memoria | `True` (opcional) |

## Ejemplo de Uso

```python
from transformers import pipeline
import torch

pipe = pipeline(
    "text-generation",
    model="Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = """### Instruction:
You are a helpful assistant specialized in simple technical explanations.

### Input:
Explain what speculative decoding is in simple terms.

### Response:
"""

outputs = pipe(prompt, max_new_tokens=200, temperature=0.7, top_p=0.95)
print(outputs[0]["generated_text"])
```

## Capacidades y Limitaciones

**Capacidades:**

- Soporta contextos extendidos de hasta 32k tokens.
- Compatible con tareas instructivas multi-turno.
- Ligero y rápido para inferencia en entornos con GPU moderadas.
- Adecuado para experimentos con decodificación especulativa.

**Limitaciones:**

- Sin alineación ética o filtros de seguridad.
- Puede generar contenido tóxico, sesgado o inapropiado.
- No optimizado para razonamiento complejo ni precisión factual.
- Solo entrenado en inglés.

## Riesgos y Consideraciones Éticas

El modelo **no aplica alineación de seguridad ni mitigación de sesgos**.
De hecho, se entrenó con ejemplos provenientes de *toxic-DPO*, lo que implica que puede producir lenguaje ofensivo o inapropiado.
Se recomienda uso únicamente en contextos controlados o de investigación.

## Notas Técnicas

- Puede utilizarse como modelo base para experimentos de **speculative decoding** junto con modelos más grandes.
- Ideal para evaluar rendimiento de inferencia en contextos largos y pipeline multi-turno.
- No diseñado para producción ni despliegue en entornos abiertos.

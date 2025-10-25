# Ficha técnica

## Módulos usados

- **os, time**: utilidades del sistema, cronómetro de inferencia.
- **psutil**: métricas de CPU/RAM del sistema.
- **dotenv (load_dotenv)**: carga `HF_TOKEN` desde `.env`.
- **huggingface_hub (login, whoami)**: autenticación a Hugging Face.
- **torch**: detección de dispositivo (CPU/GPU/MPS), tipos numéricos, memoria GPU.
- **transformers (AutoTokenizer, AutoModelForCausalLM, pipeline, TextIteratorStreamer)**: tokenizador, modelo causal, pipeline de generación y streaming de tokens.
- **rich (Live, Table)**: tabla “en vivo” durante la generación.
- **pandas**: registro y exportación de métricas a CSV.

## Funciones definidas

- **loginHF(hfToken: str)**
  
  Autentica contra Hugging Face.
  
  *Entrada:* token (string). *Salida:* — (side-effect de sesión).

- **getDevice() → str**
  
  Detecta `cuda`/`mps`/`cpu` para ejecución.

- **getSystemUsage() → dict**
  
  CPU% y RAM%/MB actuales con `psutil`.

- **getGpuUsage() → dict**
  
  Nombre de GPU y VRAM usada/reservada (MB) con `torch.cuda.*` si hay GPU.

- **fmtTinyLlama(msg, system)** / **fmtQwen(msg, system)**
  
  Devuelven lista de mensajes `{role, content}` para usar **chat template**.

- **fmtDoctorShotgun(msg, system)**
  
  Devuelve string con formato **Alpaca multi-turn** (`### Instruction/Input/Response`).

- **fmtLlama2(msg)**
  
  Devuelve string plano (modelo base no-chat).

- **MODEL_PRESETS: dict**
  
  Por `model_id`:

  - `prompt_fmt` (una de las funciones anteriores)
  - `use_chat_template: bool`
  - `gen_kwargs` por defecto: `max_new_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `do_sample`.

- **loadModel(modelId, use4bit=False, dtype=None) → (generator, tokenizer)**
  
  Carga `AutoTokenizer` y `AutoModelForCausalLM` (cuantización 4-bit opcional) y crea `pipeline("text-generation")`.
  
  *Entrada:* id del modelo y flags. *Salida:* pipeline y tokenizer.

- **buildPrompt(modelId, userText, systemText=None, tokenizer=None) → str**
  
  Aplica el formateo correcto según preset; si `use_chat_template`, usa `tokenizer.apply_chat_template`.

- **runInferenceBatch(modelId, userText, systemText=None, overrideGenKwargs=None, use4bit=False) → (text, metrics)**
  
  Inferencia “no streaming” mediante `pipeline`.
  
  Devuelve texto y métricas: `input_tokens`, `output_tokens`, `tiempo_seg`, `tok_s`, parámetros efectivos y `cpu/ram/vram` antes/después.

- **runInferenceStreaming(modelId, userText, systemText=None, overrideGenKwargs=None, use4bit=False, csvPath=None) → (text, metrics)**
  
  Inferencia con **streaming** usando `TextIteratorStreamer` y visualización en vivo con `rich.Live`.
  
  Si `csvPath` está definido, **anexa** métricas a CSV.

- **benchmarkModels(models, prompt, systemText=None, runs=1, use4bit=True, csvPath="metrics.csv", override=None) → DataFrame**
  
  Ejecuta secuencialmente N corridas por modelo (con streaming) y consolida métricas en CSV y DataFrame.

- **loadResults(csvPath="metrics.csv") → DataFrame**
  
  Carga métricas desde CSV.

- **plotQuick(df)**
  
  Gráficas rápidas (matplotlib) de `tiempo_seg` y `tok_s` por modelo.

## Parámetros relevantes (qué hacen y por qué importan)

- **`model_id`**: identifica el modelo en Hugging Face (TinyLlama, Qwen, Doctor-Shotgun).
- **`use4bit`** (loadModel): habilita cuantización 4-bit vía `bitsandbytes` para reducir VRAM/ RAM; útil en equipos con recursos limitados.
- **`torch_dtype`** (`bfloat16`/`float16`): reduce memoria y acelera en GPU compatibles.
- **`device_map="auto"`**: coloca el modelo automáticamente en GPU/MPS/CPU.
- **`max_new_tokens`**: longitud máxima de salida; impacta tiempo total y memoria.
- **`temperature`**: variabilidad/creatividad; menor es más determinista.
- **`top_p`, `top_k`**: control de muestreo; influyen en diversidad y estabilidad.
- **`repetition_penalty`**: reduce repeticiones en la salida.
- **`do_sample`**: habilita muestreo estocástico (en lugar de greedy).
- **`apply_chat_template`** (implícito en `buildPrompt`): asegura el formato correcto para modelos chat (Qwen/TinyLlama) y evita degradación por prompts mal formateados.
- **`csvPath`** (streaming/benchmark): ruta para logging reproducible de métricas.
- **`overrideGenKwargs`**: permite ajustar por corrida los hiperparámetros de generación sin cambiar los presets.
- **`HF_TOKEN`**: credencial de acceso; necesaria para modelos o cuotas que requieran autenticación.

## Flujo

1. **Autenticación**: `loginHF(HF_TOKEN)` y carga de `.env`.
2. **Elección de modelo**: definir `model_id` desde la lista de IDs.
3. **Construcción de prompt**: `buildPrompt(model_id, userText, systemText, tokenizer)` (se invoca dentro de las funciones de inferencia).
4. **Inferencia**:

   - Rápida: `runInferenceBatch(...)`
   - En vivo: `runInferenceStreaming(..., csvPath="metrics.csv")`
5. **Benchmark**: `benchmarkModels([...], prompt, runs=1, csvPath="metrics.csv")`.
6. **Análisis**: `loadResults(...)` y `plotQuick(df)`.

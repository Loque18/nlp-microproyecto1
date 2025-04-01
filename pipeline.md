
# 🎼 Pipeline para Micro Proyecto 1 - Generación de Música con Redes Neuronales

## 1. Selección de compositor e instrumento
- Elegir un compositor del dataset (ej: Beethoven).
- Extraer los archivos MIDI correspondientes.
- Usar `pretty_midi` para cargar los archivos y seleccionar un instrumento específico (ej: Piano).

## 2. Extracción y preprocesamiento de datos
- Para cada nota:
  - `pitch`: nota musical (`note.pitch`)
  - `step`: `start` actual - `start` anterior
  - `duration`: `end` - `start`
  - `velocity` (opcional): `note.velocity`
- Estructurar los datos como secuencias para entrenamiento con tamaño de contexto `c`:
  ```
  X = [Nota_{t−c}, ..., Nota_{t−1}]
  Y = Nota_{t}
  ```

## 3. Diseño del modelo (con PyTorch)
- Modelo secuencial con:
  - Embedding para `pitch`
  - LSTM para la secuencia
  - Cabezas de salida para:
    - `pitch` (clasificación)
    - `step`, `duration` (regresión)
    - `velocity` (opcional)
  
## 4. Función de pérdida compuesta
- `CrossEntropyLoss` para `pitch`
- `MSELoss` para `step`, `duration`, y `velocity` (si se usa)
- Combinación:
  ```
  total_loss = loss_pitch + alpha * loss_step + beta * loss_duration + gamma * loss_velocity
  ```

## 5. Entrenamiento del modelo
- División de datos: 80% entrenamiento / 20% validación
- Entrenamiento por batches
- Guardar el modelo final

## 6. Generación de música
- Usar un contexto inicial de `c` notas reales
- Generar 200 notas usando el modelo entrenado
- Convertir las notas a objetos `pretty_midi.Note`
- Guardar las notas generadas en archivos `.midi` y luego convertirlos a `.wav`

## 7. Entregables
- Jupyter Notebook con código, explicación y arquitectura
- Imagen del diagrama de arquitectura por separado
- Tres archivos `.wav` con 200 notas generadas cada uno

--- 

# 🎼 ArmonIA - Pipeline general

## 1. Preparación de los datos

### 1.1 Selección del compositor e instrumento
- Elegir un compositor del dataset (por ejemplo: Beethoven).
- Extraer los archivos `.mid` correspondientes a dicho compositor.
- Usar la librería `pretty_midi` para cargar los archivos MIDI.
- Seleccionar un instrumento específico (por ejemplo: Piano), filtrando las pistas que lo contienen.

## 2. Extracción y preprocesamiento de notas

### 2.1 Extracción de características
Para cada nota extraída del instrumento seleccionado:
- `pitch`: nota MIDI (`note.pitch`) — **variable categórica**
- `start`: tiempo de inicio (`note.start`)
- `end`: tiempo de finalización (`note.end`)
- `step`: diferencia entre `start` actual y `start` anterior
- `duration`: `end` - `start`
- `velocity` (opcional): `note.velocity`


### 2.2 Construcción de dataset secuencial
- Se construyen secuencias de entrenamiento con un **contexto de tamaño `c`**:
  ```python
  X = [Nota_{t−c}, ..., Nota_{t−1}]
  Y = Nota_{t}
  ```

### 2.3 Normalización de variables continuas


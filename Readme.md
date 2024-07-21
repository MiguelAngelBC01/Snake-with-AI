# IA de Snake con Deep Q-Learning

## Archivos

- `agent.py`: Aquí está el cerebro de la IA, es decir, el agente que toma decisiones.
- `game.py`: Este archivo contiene la implementación del juego de Snake.
- `model.py`: Aquí está el modelo de la red neuronal que utiliza nuestro agente.

## `agent.py`

### ¿Qué hace este archivo?

En `agent.py` definimos la clase `Agente`, que es responsable de aprender y mejorar jugando al Snake.

### Principales funciones:

- **Inicialización (`__init__`)**:

  - Aquí inicializamos nuestra red neuronal y el entrenador.
  - Cargamos un modelo preentrenado y la memoria de juego si ya existen.

- **Obtener el estado (`get_state`)**:

  - Codifica el estado actual del juego, incluyendo la dirección de la serpiente, peligros circundantes y la ubicación de la comida.

- **Memoria (`remember`, `train_long_memory`, `train_short_memory`)**:

  - Usamos una deque para almacenar las experiencias del agente.
  - Entrenamos el modelo utilizando las memorias a corto y largo plazo.

- **Acción (`get_action`)**:

  - Decidimos qué acción tomar usando una política epsilon-greedy, que equilibra entre explorar nuevas acciones y explotar el conocimiento actual.

- **Guardar y cargar modelo (`save_memory`, `load_memory`, `save_model`, `load_model`)**:

  - Manejamos la persistencia del modelo y la memoria del agente.

- **Entrenamiento (`train`)**:
  - Aquí es donde ocurre la magia. El agente juega, aprende de sus experiencias y mejora con el tiempo.

## `game.py`

### ¿Qué hace este archivo?

En `game.py` está la lógica del juego de Snake, implementada con Pygame.

### Principales funciones:

- **Inicialización (`__init__`)**:

  - Configura la ventana del juego y carga la puntuación más alta.

- **Mecánicas del juego**:

  - `reset`: Reinicia el juego.
  - `_place_food`: Coloca comida en el tablero.
  - `play_step`: Realiza un paso en el juego basado en la acción del agente.
  - `is_collision`: Verifica colisiones.
  - `_update_ui`: Actualiza la interfaz gráfica.
  - `_move`: Mueve la serpiente según la acción decidida.

- **Guardar y cargar estado del juego (`save_state`, `load_state`)**:
  - Guarda y carga el estado del juego para continuar desde donde lo dejaste.

## `model.py`

### ¿Qué hace este archivo?

En `model.py` definimos la arquitectura de la red neuronal y el proceso de entrenamiento.

### Principales componentes:

- **Clase `Linear_QNet`**:

  - Una red neuronal con una capa oculta.
  - `forward`: Define la pasada hacia adelante.
  - `save` y `load`: Manejan la persistencia del modelo.

- **Clase `QTrainer`**:
  - Maneja el entrenamiento de la red neuronal.
  - `train_step`: Realiza un paso de entrenamiento actualizando los pesos de la red.

## Cómo probarlo

1. Deberás de tener las siguientes dependencias instaladas:

   ```bash
   pip install torch pygame numpy
   ```

2. Después de instalar las dependencias ejecuta:

   ```bash
   python agent.py
   ```

3. El modelo viene con un cierto nivel de entrenamiento pero si desea ver el entrenamiento desde 0, borre el contenido de las carpetas game_state, model y memory.

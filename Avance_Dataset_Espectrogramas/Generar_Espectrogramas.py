import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURACIÓN ---
DIR_ENTRADA = "DataSet_Ejemplo"
DIR_SALIDA_ESPECTROGRAMAS = "Espectrogramas_128x128"

# Recorte de frecuencias para aislar voces/gritos humanos
FMIN = 300
FMAX = 8000
N_MELS = 128 # Número de bandas de frecuencia (Eje Y)
RESOLUCION = 128 # Tamaño final en píxeles para la CNN (128x128)

# Crear la carpeta de salida si no existe
if not os.path.exists(DIR_SALIDA_ESPECTROGRAMAS):
    os.makedirs(DIR_SALIDA_ESPECTROGRAMAS)

# --- PROCESAMIENTO PRINCIPAL ---
archivos_audio = [f for f in os.listdir(DIR_ENTRADA) if f.endswith('.wav')]

print(f"Iniciando la conversión de {len(archivos_audio)} audios a Espectrogramas limpios de 128x128...")

for archivo in archivos_audio:
    ruta_audio = os.path.join(DIR_ENTRADA, archivo)
    
    # 1. Cargar el audio
    y, sr = librosa.load(ruta_audio, sr=None)
    
    # Calcular duración real del archivo de audio en segundos
    duracion_audio = librosa.get_duration(y=y, sr=sr)
    
    # 2. Generar el Espectrograma de Mel
    espectrograma_mel = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_mels=N_MELS, 
        fmin=FMIN, 
        fmax=FMAX
    )
    
    # 3. Convertir a Decibelios
    espectrograma_mel_db = librosa.power_to_db(espectrograma_mel, ref=np.max)
    
    # Calcular cuánto tiempo real representa la matriz del espectrograma
    n_frames = espectrograma_mel_db.shape[1]
    duracion_espectrograma = librosa.frames_to_time(n_frames, sr=sr)
    
    # 4. Configurar lienzo estricto para CNN (128x128 píxeles)
    fig = plt.figure(figsize=(RESOLUCION/100, RESOLUCION/100), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # Dibujar el espectrograma (aspect='auto' lo ajusta al cuadrado)
    librosa.display.specshow(
        espectrograma_mel_db, 
        sr=sr, 
        x_axis='time', 
        y_axis='mel', 
        fmin=FMIN, 
        fmax=FMAX,
        cmap='magma',
        ax=ax
    )
    
    # 5. Guardar la imagen
    nombre_imagen = f"{archivo.split('.')[0]}.png"
    ruta_imagen = os.path.join(DIR_SALIDA_ESPECTROGRAMAS, nombre_imagen)
    
    fig.savefig(ruta_imagen, dpi=100)
    plt.close(fig)
    
    # Log informativo por consola
    print(f"Guardado: {nombre_imagen} | Audio: {duracion_audio:.2f}s -> Representado: {duracion_espectrograma:.2f}s")

print("\n¡Dataset de espectrogramas generado con éxito!")
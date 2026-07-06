# Generación de Dataset Sintético y Espectrogramas (SED)

Este repositorio contiene los códigos fuente y las muestras de datos resultantes utilizados en el desarrollo metodológico del sistema de Detección de Eventos Acústicos (SED). Aquí se documenta el proceso de mezcla paramétrica aditiva (audio) y la transformación al dominio tiempo-frecuencia (visión artificial).

## Estructura del Repositorio

El material principal se encuentra organizado dentro del directorio **`Avance_Dataset_Espectrogramas/`**, el cual contiene las siguientes subcarpetas:

### 1. `Dataset_Ejemplo/` (Muestras de Audio)
Contiene **4 archivos de audio (`.wav`)**. 
> **Nota de optimización:** Generar y subir el dataset completo de 20 audios resulta excesivamente pesado para el repositorio. Por lo tanto, se han subido únicamente estos 4 audios específicos, ya que son exactamente **los mismos que se utilizaron para la sección de ejemplos visuales y casos de estudio en el informe presentado**. Esto permite al revisor escuchar las muestras que corresponden a los escenarios evaluados en el documento.

### 2. `Espectrogramas_128x128_Ejemplo/` (Representaciones Visuales)
A diferencia de los audios, las imágenes bidimensionales son mucho más ligeras. Por ello, en esta carpeta **sí se encuentran los 20 espectrogramas totales** generados por el algoritmo. Todos los espectrogramas están estandarizados a una resolución exacta de 128x128 píxeles (sin márgenes), listos para la ingesta en una Red Neuronal Convolucional (CNN).

---

## Códigos Fuente (Scripts)

En la raíz del proyecto también se adjuntan los scripts de procesamiento digital de señales (DSP) desarrollados en Python:

* **`Generar_Dataset.py`**
  Script encargado de la síntesis acústica. Lee los bancos de sonidos ambientales y anomalías (gritos), aplica convolución (RIR) para reverberación espacial y ejecuta el modelo matemático de superposición paramétrica usando coeficientes $\alpha$ y $\beta$ para cumplir con un SNR (Relación Señal-Ruido) objetivo.

* **`Generar_Espectrogramas.py`**
  Script encargado de la extracción de características visuales. Toma los audios sintéticos previamente generados y los transforma en Espectrogramas de Mel. Aplica un filtro pasa-banda estricto (300 Hz a 8000 Hz) para resaltar la firma de la voz humana y exporta los tensores resultantes en formato de imagen optimizada.

---

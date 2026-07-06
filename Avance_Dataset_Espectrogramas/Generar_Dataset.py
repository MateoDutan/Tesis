import os
import random
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import convolve
from datetime import datetime

# --- RUTAS DE CARPETAS ---
DIR_AMBIENTALES = "Audios"
DIR_GRITOS = "Gritos"
DIR_RIR = "RIRs"       
DIR_SALIDA = "DataSet_Ejemplo"
ARCHIVO_INFORME = os.path.join(DIR_SALIDA, "Informe_Generacion_Dataset.tex")

# Crear carpetas si no existen
for carpeta in [DIR_AMBIENTALES, DIR_GRITOS, DIR_RIR, DIR_SALIDA]:
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)

# Obtener listas de archivos
archivos_ambientales = [f for f in os.listdir(DIR_AMBIENTALES) if f.endswith(('.wav', '.mp3'))]
archivos_gritos = [f for f in os.listdir(DIR_GRITOS) if f.endswith(('.wav', '.mp3'))]
archivos_rir = [f for f in os.listdir(DIR_RIR) if f.endswith(('.wav', '.mp3'))]

# --- CONFIGURACIÓN METODOLÓGICA ---
CANTIDAD_A_GENERAR = 20
FRECUENCIA_MUESTREO = 44100
MIN_SNR_DB = -10.0
MAX_SNR_DB = 5.0

print("Iniciando generación sintética con perfilado vectorial avanzado...")

def calcular_rms(senal_vector):
    if len(senal_vector) == 0:
        return 1e-10
    return np.sqrt(np.mean(senal_vector**2)) + 1e-10

cont_con_grito = 0
cont_sin_grito = 0
registro_audios = []

for i in range(CANTIDAD_A_GENERAR):
    metadata = {
        "ID": f"{i+1:04d}",
        "Etiqueta": "",
        "N": "N/A",
        "M": "N/A",
        "Inicio (s)": "N/A", # NUEVO CAMPO
        "Peak R": "N/A",
        "Peak G": "N/A",
        "RMS R": "N/A",
        "RMS G": "N/A",
        "SNR": "N/A",
        "Alfa": "N/A",
        "Beta": "N/A",
        "Clip": "No"
    }
    
    amb_file = random.choice(archivos_ambientales)
    r_n, _ = librosa.load(os.path.join(DIR_AMBIENTALES, amb_file), sr=FRECUENCIA_MUESTREO)
    len_r = len(r_n)
    metadata["N"] = str(len_r)
    
    agregar_grito = random.choice([True, False])
    
    if agregar_grito:
        grito_file = random.choice(archivos_gritos)
        g_n, _ = librosa.load(os.path.join(DIR_GRITOS, grito_file), sr=FRECUENCIA_MUESTREO)
        
        # Convolución RIR
        if archivos_rir:
            rir_file = random.choice(archivos_rir)
            h_n, _ = librosa.load(os.path.join(DIR_RIR, rir_file), sr=FRECUENCIA_MUESTREO)
            g_rev_n = convolve(g_n, h_n, mode='full')
        else:
            g_rev_n = g_n
            
        # VAD empírico
        g_activo, _ = librosa.effects.trim(g_rev_n, top_db=30)
        longitud_activa = len(g_activo)
        len_g = len(g_rev_n)
        metadata["M"] = str(longitud_activa)
        
        # Sincronización temporal
        if len_r > len_g:
            n_0 = random.randint(0, len_r - len_g)
        else:
            n_0 = 0
            
        # CÁLCULO DEL TIEMPO EN SEGUNDOS
        inicio_segundos = n_0 / FRECUENCIA_MUESTREO
        metadata["Inicio (s)"] = f"{inicio_segundos:.2f}"
            
        r_segmento_activo = r_n[n_0 : n_0 + longitud_activa]
        
        # PERFILADO VECTORIAL
        rms_grito = calcular_rms(g_activo)
        rms_ruido = calcular_rms(r_segmento_activo)
        peak_grito = np.max(np.abs(g_activo))
        peak_ruido = np.max(np.abs(r_segmento_activo))
        
        metadata["Peak G"] = f"{peak_grito:.4f}"
        metadata["Peak R"] = f"{peak_ruido:.4f}"
        metadata["RMS G"] = f"{rms_grito:.5f}"
        metadata["RMS R"] = f"{rms_ruido:.5f}"
        
        snr_0 = 20 * np.log10(rms_grito / rms_ruido)
        
        # Mezcla Paramétrica
        snr_target = random.uniform(MIN_SNR_DB, MAX_SNR_DB)
        betha_db = random.uniform(-3.0, 3.0) 
        alpha_db = betha_db + snr_target - snr_0
        
        alpha = 10 ** (alpha_db / 20)
        betha = 10 ** (betha_db / 20)
        
        metadata["SNR"] = f"{snr_target:+.2f}"
        metadata["Alfa"] = f"{alpha:.4f}"
        metadata["Beta"] = f"{betha:.4f}"
        
        # Zero-Padding y Suma
        g_padded = np.zeros(len_r)
        fin_idx = min(n_0 + len_g, len_r)
        g_padded[n_0 : fin_idx] = g_rev_n[: fin_idx - n_0]
        
        y_n = (alpha * g_padded) + (betha * r_n)
        
        max_amplitud = np.max(np.abs(y_n))
        if max_amplitud > 1.0:
            y_n = y_n / max_amplitud
            metadata["Clip"] = f"Sí"
            
        etiqueta = "anomalia"
        cont_con_grito += 1
        # INFO EXTRA ACTUALIZADA PARA CONSOLA
        info_extra = f"(SNR: {snr_target:+.1f}dB | α={alpha:.2f} | Inserción: {inicio_segundos:.2f}s)"
        
    else:
        # Flujo Normal
        betha_db = random.uniform(-3.0, 3.0)
        betha = 10 ** (betha_db / 20)
        metadata["Beta"] = f"{betha:.4f}"
        
        peak_ruido = np.max(np.abs(r_n))
        rms_ruido = calcular_rms(r_n)
        metadata["Peak R"] = f"{peak_ruido:.4f}"
        metadata["RMS R"] = f"{rms_ruido:.5f}"
        
        y_n = betha * r_n
        
        max_amplitud = np.max(np.abs(y_n))
        if max_amplitud > 1.0:
            y_n = y_n / max_amplitud
            metadata["Clip"] = f"Sí"
            
        etiqueta = "normal"
        cont_sin_grito += 1
        info_extra = ""
    
    metadata["Etiqueta"] = etiqueta
    nombre_salida = f"audio_{i+1:04d}_{etiqueta}.wav"
    registro_audios.append(metadata)
    
    ruta_salida = os.path.join(DIR_SALIDA, nombre_salida)
    sf.write(ruta_salida, y_n, FRECUENCIA_MUESTREO)
    print(f"[{i+1}/{CANTIDAD_A_GENERAR}] Creado: {nombre_salida} {info_extra}")

# =====================================================================
# GENERACIÓN DEL INFORME EN FORMATO LATEX (.tex)
# =====================================================================
print("\nGenerando informe analítico en formato LaTeX...")

with open(ARCHIVO_INFORME, "w", encoding="utf-8") as f:
    f.write("\\textbf{Resumen Estadístico del Dataset:}\n")
    f.write("\\begin{itemize}\n")
    f.write(f"    \\item \\textbf{{Total de Muestras Generadas:}} {len(registro_audios)}\n")
    f.write(f"    \\item \\textbf{{Clase Anómala (Presencia de Grito):}} {cont_con_grito}\n")
    f.write(f"    \\item \\textbf{{Clase Normal (Ausencia de Grito):}} {cont_sin_grito}\n")
    f.write("\\end{itemize}\n\n")
    
    # Inicio de la tabla LaTeX
    f.write("\\begin{table}[h!]\n")
    f.write("\\centering\n")
    f.write("\\resizebox{\\textwidth}{!}{\n")
    # Aumentamos a 13 columnas agregando 'c|'
    f.write("\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|}\n")
    f.write("\\hline\n")
    # Agregamos la cabecera \textbf{Inicio (s)}
    f.write("\\textbf{ID} & \\textbf{Clase} & \\textbf{N} & \\textbf{M} & \\textbf{Inicio (s)} & \\textbf{RMS (R)} & \\textbf{RMS (G)} & \\textbf{Peak (R)} & \\textbf{Peak (G)} & \\textbf{SNR} & \\textbf{$\\alpha$} & \\textbf{$\\beta$} & \\textbf{Clip} \\\\ \\hline\n")
    
    # Llenado de filas iterando el registro
    for reg in registro_audios:
        etiqueta_latex = reg["Etiqueta"].replace("_", "\\_")
        
        # Se agrega reg['Inicio (s)'] en la construcción de la fila
        fila = f"{reg['ID']} & {etiqueta_latex} & {reg['N']} & {reg['M']} & {reg['Inicio (s)']} & {reg['RMS R']} & {reg['RMS G']} & {reg['Peak R']} & {reg['Peak G']} & {reg['SNR']} & {reg['Alfa']} & {reg['Beta']} & {reg['Clip']} \\\\ \\hline\n"
        f.write(fila)
        
    # Cierre de la tabla y del entorno
    f.write("\\end{tabular}\n")
    f.write("}\n")
    f.write("\\vspace{0.2cm}\n")
    f.write("\\caption{Registro de parámetros matriciales, tiempos de inserción y coeficientes de mezcla aditiva por cada archivo acústico generado. $N$ y $M$ corresponden a la longitud de los vectores en muestras.}\n")
    f.write("\\label{tab:bitacora_generacion}\n")
    f.write("\\end{table}\n")

print(f"\n¡Proceso completado! Archivo final guardado en: {ARCHIVO_INFORME}")
import pandas as pd
import glob

# 1. Buscar todos los datasets maestros individuales
ruta_base = '/home/proyecto/SistemasArchivos/'
archivos = glob.glob(f'{ruta_base}dataset_maestro_*_ia.csv')

dataframes = []

for archivo in archivos:
    # 2. Leer cada CSV
    df = pd.read_csv(archivo)
    
    # 3. Extraer el nombre del proyecto del nombre del archivo 
    # Ej: de 'dataset_maestro_Proyecto2_ia.csv' saca 'Proyecto2'
    nombre_proyecto = archivo.split('_maestro_')[1].split('_ia')[0]
    
    # 4. Crear la nueva columna
    df['Proyecto'] = nombre_proyecto
    dataframes.append(df)

# 5. Concatenar y guardar el dataset final
dataset_global = pd.concat(dataframes, ignore_index=True)
ruta_salida = f'{ruta_base}dataset_maestro_GLOBAL_ia.csv'
dataset_global.to_csv(ruta_salida, index=False)

print(f"¡Fusión exitosa!")
print(f"Dataset Global creado con {len(dataset_global)} filas en total.")
print(f"Proyectos incluidos: {dataset_global['Proyecto'].unique()}")

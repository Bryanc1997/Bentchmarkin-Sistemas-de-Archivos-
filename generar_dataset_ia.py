import pandas as pd
import os
import glob

# 1. Definir las rutas a tus métricas extraídas
base_dir = '/home/proyecto/SistemasArchivos'
sistemas = {
    'EXT4': f'{base_dir}/resutados_ext4/Proyecto8/metrics', # Asegúrate de apuntar a los archivos limpios
    'XFS': f'{base_dir}/resultados_xfs/Proyecto8/metrics',
    'ZFS': f'{base_dir}/resultados_zfs/Proyecto8/metrics'
}

dataframes = []

for fs_name, path in sistemas.items():
    # Buscar todos los archivos csv de telemetría generados
    csv_files = glob.glob(os.path.join(path, '*_pidstat*.csv'))
    
    for file in csv_files:
        # Saltar archivos vacíos o con muy pocos bytes (ej. importaciones rápidas)
        if os.path.getsize(file) < 150: 
            continue
            
        try:
            # skiprows=2 omite la línea de "Linux 4.18..." y la línea en blanco
            df = pd.read_csv(file, skiprows=2, on_bad_lines='skip')
            
            # Limpiar los nombres de las columnas (quitar el '#' y los espacios)
            df.columns = df.columns.str.replace('#', '').str.strip()
            
            # Crear las variables predictoras (Features) para la IA
            df['FileSystem'] = fs_name
            # Extraer la fase desde el nombre del archivo (ej: de 'dada2_Colitis' saca 'dada2')
            df['Pipeline_Phase'] = os.path.basename(file).split('_')[0]
            
            dataframes.append(df)
        except Exception as e:
            print(f"Omitiendo {file} por error de lectura: {e}")

# 2. Concatenar todo en un solo DataFrame
dataset_maestro = pd.concat(dataframes, ignore_index=True)

# 3. Filtrar ruido del OS y herramientas de monitoreo
# Solo nos interesan los comandos reales (qiime, python, fastp, bash que los ejecuta)
ruido = ['pidstat', 'iostat', 'time', 'htop', 'rcu_sched', 'kworker', 'tuned']
dataset_maestro = dataset_maestro[~dataset_maestro['Command'].str.contains('|'.join(ruido), na=False, case=False)]

# 4. Exportar el CSV final para el entrenamiento de IA
ruta_salida = f'{base_dir}/dataset_maestro_Proyecto8_ia.csv'
dataset_maestro.to_csv(ruta_salida, index=False)

print(f"¡Dataset consolidado con éxito!")
print(f"Total de filas para entrenamiento: {len(dataset_maestro)}")
print(f"Guardado en: {ruta_salida}")

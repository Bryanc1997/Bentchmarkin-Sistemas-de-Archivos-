import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. Cargar el dataset maestro perfecto
ruta_dataset = '/home/proyecto/SistemasArchivos/dataset_maestro_GLOBAL_FINAL.csv'
print("Cargando dataset...")
df = pd.read_csv(ruta_dataset)

# 2. Preprocesamiento (Limpieza final para la IA)
# Eliminar columnas que no aportan patrones predictivos (Time, UID, PID)
columnas_a_eliminar = ['Time', 'UID', 'PID']
df = df.drop(columns=columnas_a_eliminar)

# Convertir variables de texto (Command, Pipeline_Phase, Proyecto) a números
le_command = LabelEncoder()
df['Command'] = le_command.fit_transform(df['Command'])

le_phase = LabelEncoder()
df['Pipeline_Phase'] = le_phase.fit_transform(df['Pipeline_Phase'])

le_proyecto = LabelEncoder()
df['Proyecto'] = le_proyecto.fit_transform(df['Proyecto'])

# 3. Definir "X" (Features/Variables predictoras) y "y" (Target/Objetivo)
# Queremos que la IA aprenda a identificar el FileSystem por su rendimiento
X = df.drop(columns=['FileSystem'])
y = df['FileSystem']

# 4. Dividir datos: 80% para entrenar, 20% para examen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Entrenando modelo con {len(X_train)} registros...")

# 5. Instanciar y Entrenar el Modelo (Random Forest)
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
modelo_rf.fit(X_train, y_train)

# 6. Predicción y Evaluación
predicciones = modelo_rf.predict(X_test)

print("\n=== RESULTADOS DEL MODELO RANDOM FOREST ===")
print(f"Precisión global (Accuracy): {accuracy_score(y_test, predicciones):.4f}\n")
print("Reporte detallado por Sistema de Archivos:")
print(classification_report(y_test, predicciones))

# Opcional: Ver qué métricas fueron las más importantes para el modelo
importancias = pd.DataFrame(
    {'Metrica': X.columns, 'Importancia': modelo_rf.feature_importances_}
).sort_values(by='Importancia', ascending=False)

print("\nLas 5 métricas más críticas en la bioinformática:")
print(importancias.head(5))

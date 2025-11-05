import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
# Librería para Winsorización (Capado de outliers)
from scipy.stats.mstats import winsorize 

# 1. Cargar el dataset
df = pd.read_csv('dataset_estudiantes.csv')

# 2. Análisis Exploratorio: Estructura y Nulos
print("--- Estructura Inicial del Dataset ---")
print(df.info())
print("\n--- Conteo de Valores Nulos ---")
print(df.isnull().sum())

# 3. Creación de variables objetivo
# La variable 'aprobado' ya está en el dataset.
# Verificamos si hay que recalcularla o si está lista para usar.
# Si el enunciado pide "1 si nota_final ≥ 60, 0 en caso contrario", asumimos que está lista.
# Eliminamos duplicados si existen, aunque por simplicidad, seguimos con el preprocesamiento.
# Definición de columnas por tipo
num_cols = ['horas_estudio_semanal', 'nota_anterior', 'tasa_asistencia', 'horas_sueno', 'edad']
cat_nominal_cols = ['horario_estudio_preferido', 'estilo_aprendizaje']
cat_ordinal_cols = {'nivel_dificultad': {'Fácil': 1, 'Medio': 2, 'Difícil': 3}}
cat_binaria_cols = {'tiene_tutor': {'Sí': 1, 'No': 0}}

# --- 1. Imputación de Nulos (Previo al Capado) ---

# Imputación de la moda para categóricas
for col in cat_nominal_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)
    
# Imputación de la mediana para numéricas
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# --- 2. Gestión de Outliers (Winsorización) ---
for col in num_cols:
    # Capado (Winsorización) al 5% y 95%
    df[col] = winsorize(df[col], limits=[0.05, 0.05])

# --- 3. Codificación de Variables Categóricas ---

# Codificación Ordinal (Label Encoding) para nivel_dificultad
df['nivel_dificultad'] = df['nivel_dificultad'].map(cat_ordinal_cols['nivel_dificultad'])

# Codificación Binaria (Label Encoding) para tiene_tutor
df['tiene_tutor'] = df['tiene_tutor'].map(cat_binaria_cols['tiene_tutor'])

# Codificación Nominal (One-Hot Encoding)
df = pd.get_dummies(df, columns=cat_nominal_cols, drop_first=True, dtype=int)

# --- 4. División de Variables (X e y) ---

# Variables Predictoras (X): todas excepto las variables objetivo
X = df.drop(columns=['nota_final', 'aprobado'])
# Variables Objetivo (y)
y_reg = df['nota_final']
y_clas = df['aprobado']

# --- 5. Estandarización de Variables Numéricas ---
# Estandarizador (se entrena SOLO en las columnas numéricas de X)
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

print("\n--- Preprocesamiento Finalizado. Primeras 5 Filas de X ---")
print(X.head())
print("\n--- Columnas y Tipos de Datos (Todas numéricas) ---")
print(X.info())

# División para Regresión
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# 2.2. Entrenamiento del Modelo
model_reg = LinearRegression()
model_reg.fit(X_train_reg, y_train_reg)

# 2.3. Predicciones y Evaluación
y_pred_reg = model_reg.predict(X_test_reg)

# Métricas de Evaluación
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
coefficients = pd.Series(model_reg.coef_, index=X.columns).sort_values(ascending=False)

print("--- Evaluación del Modelo de Regresión Lineal ---")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Coeficiente de Determinación (R²): {r2:.2f}")
print("\n--- Top 5 Coeficientes (Importancia) ---")
print(coefficients.head())

# División para Clasificación (con estratificación)
X_train_clas, X_test_clas, y_train_clas, y_test_clas = train_test_split(
    X, y_clas, test_size=0.2, random_state=42, stratify=y_clas 
)

# 3.2. Entrenamiento del Modelo
# Nota: La Regresión Logística de sklearn incluye regularización L2 por defecto.
model_clas = LogisticRegression(random_state=42, solver='liblinear') 
model_clas.fit(X_train_clas, y_train_clas)

# 3.3. Predicciones y Evaluación
y_pred_clas = model_clas.predict(X_test_clas)
y_prob_clas = model_clas.predict_proba(X_test_clas)[:, 1]

# Métricas de Evaluación
accuracy = accuracy_score(y_test_clas, y_pred_clas)
conf_matrix = confusion_matrix(y_test_clas, y_pred_clas)
report = classification_report(y_test_clas, y_pred_clas)

print("\n--- Evaluación del Modelo de Regresión Logística ---")
print(f"Precisión General (Accuracy): {accuracy:.2f}")
print("\n--- Matriz de Confusión ---")
print(conf_matrix)
print("\n--- Reporte de Clasificación ---")
print(report)

import json

def generar_infografia_html(resultados, nombre_archivo="resultados_proyecto_ia.html"):
    """
    Genera el archivo HTML de la infografía insertando las métricas de ML
    directamente en la sección JavaScript.

    Args:
        resultados (dict): Un diccionario con las métricas de ambos modelos.
        nombre_archivo (str): Nombre del archivo HTML a generar.
    """
    # 1. Convertir el diccionario de resultados de Python a una cadena JSON
    # 'indent=4' lo hace legible en el archivo HTML.
    resultados_json = json.dumps(resultados, indent=4)

    # 2. Definir la plantilla HTML completa con un marcador de posición {MODEL_RESULTS_JSON}
    # NOTA: Todo el código CSS y JS de la infografía se ha puesto en una única
    # string multilinea de Python.
    html_template = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Proyecto IA/ML: Rendimiento Estudiantil</title>
    
    <style>
        /* INICIO DEL CÓDIGO CSS */
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f4f7f6;
            color: #333;
            margin: 0;
            padding: 0;
            line-height: 1.6;
        }

        header {
            background-color: #007bff;
            color: white;
            padding: 30px 20px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        header h1 {
            margin: 0;
            font-size: 2.2em;
        }

        header p {
            margin-top: 5px;
            font-size: 1.1em;
        }

        .container {
            display: flex;
            justify-content: space-around;
            padding: 40px 20px;
            gap: 30px;
        }

        .card {
            background-color: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            width: 45%;
            min-width: 350px;
            transition: transform 0.3s;
        }

        .card:hover {
            transform: translateY(-5px);
        }

        .regression {
            border-top: 5px solid #28a745; /* Verde para Regresión */
        }

        .classification {
            border-top: 5px solid #ffc107; /* Amarillo para Clasificación */
        }

        .card h2 {
            color: #007bff;
            margin-top: 0;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }

        .card .description {
            font-style: italic;
            color: #666;
            margin-bottom: 20px;
        }

        .metrics {
            display: flex;
            justify-content: space-between;
            margin-bottom: 30px;
        }

        .metric-item {
            text-align: center;
            padding: 15px;
            border-radius: 8px;
            background-color: #f8f9fa;
            width: 48%;
        }

        .metric-item h3 {
            margin: 0 0 5px 0;
            font-size: 0.9em;
            color: #007bff;
        }

        .metric-item .value {
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
            margin: 0;
        }

        /* Estilos de Coeficientes */
        .feature-importance h3 {
            color: #333;
            border-bottom: 1px dashed #ccc;
            padding-bottom: 5px;
            margin-bottom: 15px;
        }

        .feature-importance ul {
            list-style: none;
            padding: 0;
        }

        .feature-importance li {
            padding: 8px 0;
            border-bottom: 1px dotted #eee;
            display: flex;
            justify-content: space-between;
            font-size: 1.05em;
        }

        .feature-importance li:last-child {
            border-bottom: none;
        }

        .coefficient-value {
            font-weight: bold;
            color: #28a745;
        }

        /* Estilos de Clasificación */
        .classification-report {
            margin-top: 20px;
        }

        .report-detail {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px dotted #eee;
            font-size: 1.05em;
        }

        .f1-value {
            font-weight: bold;
            color: #ffc107;
        }

        /* Matriz de Confusión */
        #confusion_matrix {
            width: 100%;
            margin-top: 10px;
            border-collapse: collapse;
            font-size: 0.9em;
        }

        #confusion_matrix th, #confusion_matrix td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }

        #confusion_matrix th {
            background-color: #007bff;
            color: white;
        }

        #confusion_matrix td {
            font-weight: bold;
            background-color: #e9ecef;
        }
        /* FIN DEL CÓDIGO CSS */
    </style>
</head>
<body>
    <header>
        <h1>🎓 Análisis Predictivo del Rendimiento Estudiantil</h1>
        <p>Preprocesamiento, Regresión Lineal y Regresión Logística</p>
    </header>

    <div class="container">
        <section class="card regression">
            <h2>Modelo 1: Regresión Lineal (Nota Final)</h2>
            <p class="description">Predice la nota continua (0-100).</p>
            <div class="metrics">
                <div class="metric-item">
                    <h3>R² (Varianza Explicada)</h3>
                    <p id="r2_score" class="value">Cargando...</p>
                </div>
                <div class="metric-item">
                    <h3>Error Cuadrático Medio (MSE)</h3>
                    <p id="mse_score" class="value">Cargando...</p>
                </div>
            </div>
            <div class="feature-importance">
                <h3>Factores Clave (Top 5 Coeficientes)</h3>
                <ul id="reg_coefficients"><li>Cargando...</li></ul>
            </div>
        </section>

        <section class="card classification">
            <h2>Modelo 2: Regresión Logística (Aprobado/Suspenso)</h2>
            <p class="description">Clasifica si el alumno aprueba (1) o suspende (0).</p>
            <div class="metrics">
                <div class="metric-item">
                    <h3>Precisión General (Accuracy)</h3>
                    <p id="accuracy_score" class="value">Cargando...</p>
                </div>
                <div class="metric-item confusion-matrix-container">
                    <h3>Matriz de Confusión</h3>
                    <table id="confusion_matrix"></table>
                </div>
            </div>
            <div class="classification-report">
                <h3>Detalle de Clases (F1-Score)</h3>
                <div class="report-detail">
                    <p>Clase 0 (Suspenso) - F1:</p> <span id="f1_0" class="f1-value">Cargando...</span>
                </div>
                <div class="report-detail">
                    <p>Clase 1 (Aprobado) - F1:</p> <span id="f1_1" class="f1-value">Cargando...</span>
                </div>
            </div>
        </section>
    </div>

    <script>
        /* INICIO DEL CÓDIGO JAVASCRIPT */
        // --- DATOS INYECTADOS POR PYTHON ---
        const MODEL_RESULTS = {MODEL_RESULTS_JSON};

        function displayResults() {
            const regResults = MODEL_RESULTS.regression;
            const clasResults = MODEL_RESULTS.classification;

            // --- 1. Regresión Lineal ---
            document.getElementById('r2_score').textContent = regResults.R2.toFixed(3);
            document.getElementById('mse_score').textContent = regResults.MSE.toFixed(2);

            const coefList = document.getElementById('reg_coefficients');
            coefList.innerHTML = ''; // Limpiar
            regResults.coefficients.forEach(item => {
                const li = document.createElement('li');
                const valueSpan = document.createElement('span');
                
                // Asignar color basado en si el coeficiente es positivo o negativo
                valueSpan.className = 'coefficient-value';
                valueSpan.style.color = item.value >= 0 ? '#28a745' : '#dc3545'; // Verde para positivo, Rojo para negativo

                valueSpan.textContent = item.value.toFixed(2);
                li.innerHTML = `${item.name}: `;
                li.appendChild(valueSpan);
                coefList.appendChild(li);
            });

            // --- 2. Regresión Logística ---
            document.getElementById('accuracy_score').textContent = clasResults.accuracy.toFixed(3);
            document.getElementById('f1_0').textContent = clasResults.f1_score.class_0.toFixed(2);
            document.getElementById('f1_1').textContent = clasResults.f1_score.class_1.toFixed(2);

            // Matriz de Confusión (Generación de tabla)
            const matrixTable = document.getElementById('confusion_matrix');
            matrixTable.innerHTML = `
                <tr>
                    <th></th>
                    <th>Predicho: 0 (Suspenso)</th>
                    <th>Predicho: 1 (Aprobado)</th>
                </tr>
                <tr>
                    <th>Real: 0 (Suspenso)</th>
                    <td>${clasResults.conf_matrix[0][0]} (VN)</td>
                    <td>${clasResults.conf_matrix[0][1]} (FP)</td>
                </tr>
                <tr>
                    <th>Real: 1 (Aprobado)</th>
                    <td>${clasResults.conf_matrix[1][0]} (FN)</td>
                    <td>${clasResults.conf_matrix[1][1]} (VP)</td>
                </tr>
            `;
        }

        // Ejecutar la función al cargar la página
        window.onload = displayResults;
        /* FIN DEL CÓDIGO JAVASCRIPT */
    </script>
</body>
</html>
"""
    # 3. Sustituir el marcador de posición por la cadena JSON real
    html_final = html_template.replace("{MODEL_RESULTS_JSON}", resultados_json)

    # 4. Escribir el contenido final en el archivo
    with open(nombre_archivo, "w", encoding="utf-8") as f:
        f.write(html_final)
    
    print(f"✅ Infografía generada con éxito: {nombre_archivo}. ¡Ábrela en tu navegador!")


# --- EJEMPLO DE USO ---
# **ESTE ES EL CÓDIGO QUE EJECUTARÍAS EN TU PROYECTO PYTHON**
# (Asumiendo que has calculado estas métricas previamente)

# 1. Definir las variables con los resultados de Python
# Asegúrate de que las 5 variables más importantes y sus coeficientes estén en la lista.
resultados_reales = {
    "regression": {
        "R2": 0.35,  # R2 real
        "MSE": 52.95,  # MSE real
        "coefficients": [
            {"name": "nota_anterior", "value": 2.35},
            {"name": "horas_estudio_semanal", "value": 3.48},
            {"name": "tasa_asistencia", "value": 1.76},
            {"name": "estilo_aprendizaje_Kinestésico", "value": 1.19},
            {"name": "estilo_aprendizaje_Lectura/Escritura", "value": 0.87} 
        ]
    },
    "classification": {
        "accuracy": 0.91,  # Accuracy real
        "conf_matrix": [
            [3, 17],  # VN, FP
            [1, 179]   # FN, VP
        ],
        "f1_score": {
            "class_0": 0.25,  # F1-Score Suspenso
            "class_1": 0.95   # F1-Score Aprobado
        }
    }
}

# 2. Llamar a la función para generar el archivo
generar_infografia_html(resultados_reales)
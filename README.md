# 📚 Proyecto de Machine Learning: Análisis Predictivo del Rendimiento Estudiantil

## 🎯 Objetivo del Proyecto

El objetivo principal de este proyecto es analizar el rendimiento académico de un grupo de estudiantes utilizando el conjunto de datos `dataset_estudiantes.csv` y desarrollar dos modelos predictivos clave:

1.  **Regresión Lineal:** Para predecir la **Nota Final** (`nota_final`), una variable continua entre 0 y 100.
2.  **Regresión Logística:** Para clasificar si el estudiante **Aprueba o Suspende** (`aprobado`), donde 1 = Aprobado (>=60) y 0 = Suspenso.

***

## ⚙️ Estructura del Repositorio

| Archivo | Descripción |
| :--- | :--- |
| `README.md` | Documentación principal del proyecto. |
| `11.ProyectoML.py` | Script de Python que contiene todo el flujo de trabajo: preprocesamiento, entrenamiento de modelos y generación del reporte final. |
| `dataset_estudiantes.csv` | Conjunto de datos original utilizado para el entrenamiento. |
| `resultados_proyecto_ia.html` | **Infografía de Resultados Finales:** Archivo HTML generado automáticamente por el script de Python, que visualiza las métricas clave de ambos modelos. |

***

## 🛠️ Tecnologías y Librerías

El proyecto ha sido desarrollado en **Python** utilizando un stack estándar de Machine Learning:

* **Lenguaje:** Python 3.x
* **Datos y Álgebra Lineal:** `Pandas`, `NumPy`.
* **Modelado y Métricas:** `Scikit-learn` (modelos, preprocesamiento y evaluación).
* **Reporte:** Módulo `json` (para integrar los resultados de Python en el reporte HTML/JavaScript).

***

## 📋 Metodología del Proyecto

El proyecto se estructuró siguiendo los pasos esenciales de la Ciencia de Datos para asegurar la robustez de los modelos:

### 1. Preprocesamiento de Datos

* **Análisis Exploratorio de Datos (EDA):** Inspección de la calidad del dataset, tipos de datos y distribución de las variables.
* **Gestión de Valores Nulos y Atípicos (Outliers):** Tratamiento de datos faltantes (nulos) y mitigación del impacto de valores atípicos.
* **Codificación de Variables Categóricas:** Transformación de variables como `horario_estudio_preferido`, `estilo_aprendizaje` y `nivel_dificultad` a formato numérico (e.g., mediante **One-Hot Encoding**) para ser compatibles con los modelos lineales.
* **Estandarización/Escalado:** Las variables numéricas fueron escaladas (e.g., con `StandardScaler`) para asegurar que ninguna característica dominara el entrenamiento del modelo.
* **División de Datos:** El conjunto de datos se dividió en conjuntos de entrenamiento y prueba (típicamente 80/20) para validar la capacidad de generalización de los modelos.

### 2. Entrenamiento y Evaluación de Modelos

| Modelo | Variable Objetivo | Tipo de Problema | Métrica de Regresión | Métrica de Clasificación |
| :--- | :--- | :--- | :--- | :--- |
| **Regresión Lineal** | `nota_final` | Regresión | R^2, MSE | N/A |
| **Regresión Logística** | `aprobado` | Clasificación | N/A | Accuracy, Matriz de Confusión, F1-Score |

***

## 📊 Resultados Obtenidos

Los resultados del entrenamiento y evaluación del conjunto de prueba se resumen a continuación. Los resultados completos y la matriz de confusión se encuentran en el archivo `resultados_proyecto_ia.html`.

### Modelo 1: Regresión Lineal (Nota Final)

| Métrica | Valor Final |
| :--- | :--- |
| **Coeficiente de Determinación (R^2)** | **[INSERTAR VALOR R2]** |
| **Error Cuadrático Medio (MSE)** | **[INSERTAR VALOR MSE]** |

**Factores Clave (Top 5 Coeficientes):**
1.  `[Nombre Variable 1]`: [Valor Coeficiente 1]
2.  `[Nombre Variable 2]`: [Valor Coeficiente 2]
3.  `[Nombre Variable 3]`: [Valor Coeficiente 3]
4.  `[Nombre Variable 4]`: [Valor Coeficiente 4]
5.  `[Nombre Variable 5]`: [Valor Coeficiente 5]

### Modelo 2: Regresión Logística (Aprobado/Suspenso)

| Métrica | Valor Final |
| :--- | :--- |
| **Precisión General (Accuracy)** | **[INSERTAR VALOR ACCURACY]** |
| **F1-Score (Aprobado)** | **[INSERTAR VALOR F1-CLASE 1]** |
| **F1-Score (Suspenso)** | **[INSERTAR VALOR F1-CLASE 0]** |

***

## 🚀 Cómo Ejecutar el Proyecto

Para replicar el entorno, entrenar los modelos y generar el reporte HTML:

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://docs.github.com/es/repositories/creating-and-managing-repositories/quickstart-for-repositories](https://docs.github.com/es/repositories/creating-and-managing-repositories/quickstart-for-repositories)
    cd [nombre-del-repositorio]
    ```
2.  **Instalar dependencias:**
    ```bash
    pip install pandas numpy scikit-learn
    ```
3.  **Ejecutar el script principal:**
    ```bash
    python 11.ProyectoML.py
    ```
4.  Una vez finalizada la ejecución, abre el archivo **`resultados_proyecto_ia.html`** en cualquier navegador web para ver la infografía interactiva con los resultados finales.

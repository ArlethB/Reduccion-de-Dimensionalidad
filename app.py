"""
Aplicacion de Streamlit: PCA + K-Means + SVM sobre el dataset MNIST
Actividad Individual - Reduccion de Dimensionalidad y Clasificacion con PCA, K-Means y SVM
Estudiante: Arleth Adyani Chevez Bonilla - Cuenta: 20221900251
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="MNIST: PCA + K-Means + SVM",
    layout="wide",
)

RANDOM_STATE = 42


# ============================================================
# CARGA Y PREPARACION DE DATOS
# ============================================================

@st.cache_data
def cargar_datos():
    """Carga train.csv del dataset MNIST (Kaggle Digit Recognizer)."""
    try:
        df = pd.read_csv("train.csv")
        return df
    except FileNotFoundError:
        return None


@st.cache_data
def preparar_datos(df, n_muestra=8000, test_size=0.2):
    """Normaliza los pixeles y separa entrenamiento y prueba de forma estratificada."""
    X = df.drop("label", axis=1).values / 255.0
    y = df["label"].values

    if len(X) > n_muestra:
        indices = np.random.RandomState(RANDOM_STATE).choice(len(X), n_muestra, replace=False)
        X = X[indices]
        y = y[indices]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


@st.cache_data
def entrenar_pipeline(X_train, X_test, y_train, y_test, n_components, n_clusters, kernel, C):
    """Ajusta PCA sobre entrenamiento, K-Means y SVM, evaluando SVM sobre datos de prueba."""
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    clusters_train = kmeans.fit_predict(X_train_pca)

    svm = SVC(kernel=kernel, C=C, gamma="scale", random_state=RANDOM_STATE)
    svm.fit(X_train_pca, y_train)
    y_pred_test = svm.predict(X_test_pca)

    return {
        "pca": pca,
        "X_train_pca": X_train_pca,
        "X_test_pca": X_test_pca,
        "kmeans": kmeans,
        "clusters_train": clusters_train,
        "svm": svm,
        "y_pred_test": y_pred_test,
    }


@st.cache_data
def comparar_componentes(X_train, X_test, y_train, y_test, kernel, C, lista_componentes):
    """Compara el accuracy del SVM segun distintos numeros de componentes PCA."""
    resultados = []
    for n in lista_componentes:
        pca_temp = PCA(n_components=n, random_state=RANDOM_STATE)
        X_train_temp = pca_temp.fit_transform(X_train)
        X_test_temp = pca_temp.transform(X_test)

        svm_temp = SVC(kernel=kernel, C=C, gamma="scale", random_state=RANDOM_STATE)
        svm_temp.fit(X_train_temp, y_train)
        pred_temp = svm_temp.predict(X_test_temp)

        resultados.append({
            "n_componentes": n,
            "varianza_explicada": pca_temp.explained_variance_ratio_.sum(),
            "accuracy": accuracy_score(y_test, pred_temp),
        })
    return pd.DataFrame(resultados)


# ============================================================
# VISUALIZACIONES
# ============================================================

def graficar_varianza(pca):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    varianza_acumulada = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(range(1, len(varianza_acumulada) + 1), varianza_acumulada, marker="o", markersize=4)
    ax.axhline(y=0.90, color="orange", linestyle="--", label="90% de varianza")
    ax.axhline(y=0.95, color="red", linestyle="--", label="95% de varianza")
    ax.set_xlabel("Numero de componentes")
    ax.set_ylabel("Varianza explicada acumulada")
    ax.set_title("Varianza explicada por PCA")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def graficar_proyeccion_2d(X, etiquetas, titulo, etiqueta_barra):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=etiquetas, cmap="tab10", alpha=0.6, s=18)
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    ax.set_title(titulo)
    plt.colorbar(scatter, ax=ax, label=etiqueta_barra)
    plt.tight_layout()
    return fig


def graficar_matriz_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar_kws={"label": "Cantidad"})
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Digito real")
    ax.set_title("Matriz de confusion (conjunto de prueba)")
    plt.tight_layout()
    return fig


# ============================================================
# INTERFAZ
# ============================================================

st.title("Clasificacion de Digitos MNIST con PCA, K-Means y SVM")
st.caption("Arleth Adyani Chevez Bonilla — Cuenta: 20221900251")
st.markdown("---")

df = cargar_datos()

if df is None:
    st.error("No se encontro el archivo train.csv en el directorio de la aplicacion.")
    st.markdown(
        """
        **Para usar esta aplicacion:**
        1. Coloca un archivo `train.csv` con columnas `label, pixel0, pixel1, ..., pixel783`
           en la misma carpeta que `app.py`.
        2. Si estas desplegando en Streamlit Community Cloud, sube ese `train.csv`
           junto con el resto del repositorio.
        """
    )
    st.stop()

st.sidebar.header("Configuracion")

n_components = st.sidebar.slider(
    "Numero de componentes PCA", min_value=2, max_value=100, value=30, step=1,
    help="Dimensiones a las que se reducen las 784 caracteristicas originales (28x28 pixeles)."
)

n_clusters = st.sidebar.slider(
    "Numero de clusters (K-Means)", min_value=2, max_value=15, value=10, step=1,
    help="Se recomiendan 10, ya que coincide con los digitos 0 al 9."
)

kernel = st.sidebar.selectbox("Kernel SVM", ["rbf", "linear", "poly", "sigmoid"])
C = st.sidebar.slider("Parametro C (SVM)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

n_muestra = st.sidebar.select_slider(
    "Tamano de muestra a usar", options=[1000, 2000, 3000, 4000, 5000], value=5000,
    help="Tu dataset tiene 5000 imagenes en total. Reduce este valor si la aplicacion tarda demasiado en entrenar."
)

entrenar = st.sidebar.button("Entrenar modelo", use_container_width=True)

if "resultados" not in st.session_state:
    st.session_state["resultados"] = None

if entrenar:
    with st.spinner("Preparando datos y entrenando PCA, K-Means y SVM..."):
        X_train, X_test, y_train, y_test = preparar_datos(df, n_muestra=n_muestra)
        resultados = entrenar_pipeline(
            X_train, X_test, y_train, y_test, n_components, n_clusters, kernel, C
        )
        st.session_state["resultados"] = resultados
        st.session_state["datos"] = (X_train, X_test, y_train, y_test)
        st.session_state["config"] = (n_components, n_clusters, kernel, C)

if st.session_state["resultados"] is None:
    st.info("Configura los parametros en el panel izquierdo y presiona **Entrenar modelo** para comenzar.")
    st.markdown(
        """
        ### Que hace esta aplicacion

        1. **PCA** reduce las 784 caracteristicas de cada imagen (28x28 pixeles) al numero
           de componentes que elijas, conservando la mayor varianza posible.
        2. **K-Means** agrupa las imagenes en clusters usando unicamente los datos reducidos,
           sin usar la etiqueta real del digito.
        3. **SVM** se entrena con los mismos componentes de PCA para clasificar el digito,
           y se evalua sobre un conjunto de prueba que el modelo nunca vio.
        4. Al final se puede elegir una imagen especifica del conjunto de prueba y ver
           la clase que predice el modelo para esa imagen en particular.
        """
    )
else:
    resultados = st.session_state["resultados"]
    X_train, X_test, y_train, y_test = st.session_state["datos"]
    n_components_actual, n_clusters_actual, kernel_actual, C_actual = st.session_state["config"]

    pca = resultados["pca"]
    kmeans = resultados["kmeans"]
    svm = resultados["svm"]
    y_pred_test = resultados["y_pred_test"]

    # -------------------- 1. PCA --------------------
    st.header("1. PCA - Reduccion de dimensionalidad")

    col1, col2, col3 = st.columns(3)
    col1.metric("Dimensiones originales", "784")
    col2.metric("Dimensiones reducidas", n_components_actual)
    col3.metric("Varianza explicada", f"{pca.explained_variance_ratio_.sum():.2%}")

    st.pyplot(graficar_varianza(pca))

    st.subheader("Proyeccion 2D de los digitos (coloreada por digito real)")
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit(X_train)
    X_train_2d = pca_2d.transform(X_train)
    st.pyplot(graficar_proyeccion_2d(
        X_train_2d, y_train,
        "Proyeccion PCA 2D de los digitos de entrenamiento", "Digito"
    ))

    # -------------------- 2. K-Means --------------------
    st.header("2. K-Means - Clustering")

    clusters_train = resultados["clusters_train"]
    sil = silhouette_score(resultados["X_train_pca"], clusters_train)

    col1, col2 = st.columns(2)
    col1.metric("Numero de clusters", n_clusters_actual)
    col2.metric("Silhouette score", f"{sil:.4f}")

    st.subheader("Clusters proyectados en 2D")
    st.pyplot(graficar_proyeccion_2d(
        X_train_2d, clusters_train,
        f"Clusters generados por K-Means (k={n_clusters_actual})", "Cluster"
    ))

    st.subheader("Tabla cruzada: cluster asignado vs digito real")
    st.caption(
        "Cada fila es un cluster de K-Means y cada columna un digito real. "
        "Si un cluster esta dominado por un solo digito, K-Means logro separarlo bien."
    )
    tabla_cruzada = pd.crosstab(clusters_train, y_train, rownames=["cluster"], colnames=["digito_real"])
    st.dataframe(tabla_cruzada, use_container_width=True)

    # -------------------- 3. SVM --------------------
    st.header("3. SVM - Clasificacion supervisada")
    st.caption(f"Evaluado sobre el conjunto de prueba ({len(y_test)} muestras que el modelo no vio en entrenamiento)")

    accuracy_test = accuracy_score(y_test, y_pred_test)

    col1, col2, col3 = st.columns(3)
    col1.metric("Kernel", kernel_actual)
    col2.metric("Accuracy (prueba)", f"{accuracy_test:.4f}")
    col3.metric("Muestras de prueba", len(y_test))

    st.pyplot(graficar_matriz_confusion(y_test, y_pred_test))

    st.subheader("Reporte de clasificacion")
    reporte = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(reporte).transpose().round(4), use_container_width=True)

    # -------------------- 4. Prediccion individual --------------------
    st.header("4. Clasificar una imagen especifica")
    st.caption("Elige una imagen del conjunto de prueba y observa la clase predicha por el SVM.")

    indice = st.slider("Indice de la imagen de prueba", 0, len(X_test) - 1, 0)

    col_img, col_info = st.columns([1, 2])

    imagen = X_test[indice].reshape(28, 28)
    entrada_pca = pca.transform(X_test[indice].reshape(1, -1))
    prediccion = svm.predict(entrada_pca)[0]
    real = y_test[indice]

    with col_img:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(imagen, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)

    with col_info:
        st.metric("Digito real", int(real))
        st.metric("Digito predicho por SVM", int(prediccion))
        if prediccion == real:
            st.success("El modelo clasifico correctamente esta imagen.")
        else:
            st.error("El modelo se equivoco con esta imagen.")

    # -------------------- 5. Efecto de la reduccion de dimensionalidad --------------------
    st.header("5. Efecto del numero de componentes PCA sobre el SVM")
    st.caption("Se repite el entrenamiento de SVM con distintos numeros de componentes para comparar el desempeno.")

    if st.button("Ejecutar comparacion (puede tardar un poco)"):
        with st.spinner("Entrenando SVM con distintos numeros de componentes..."):
            lista_componentes = [5, 10, 20, 30, 50, 100]
            comparacion = comparar_componentes(
                X_train, X_test, y_train, y_test, kernel_actual, C_actual, lista_componentes
            )

        st.dataframe(comparacion.round(4), use_container_width=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(comparacion["n_componentes"], comparacion["varianza_explicada"], marker="o")
        axes[0].set_title("Varianza explicada vs numero de componentes")
        axes[0].set_xlabel("Numero de componentes")
        axes[0].set_ylabel("Varianza explicada")
        axes[0].grid(alpha=0.3)

        axes[1].plot(comparacion["n_componentes"], comparacion["accuracy"], marker="o", color="darkorange")
        axes[1].set_title("Accuracy del SVM vs numero de componentes")
        axes[1].set_xlabel("Numero de componentes")
        axes[1].set_ylabel("Accuracy")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown(
            """
            **Interpretacion:** con pocos componentes se pierde precision porque se
            descarta demasiada informacion de la imagen original. A partir de 20-30
            componentes el accuracy se estabiliza y agregar mas componentes ya no
            mejora mucho el resultado, aunque si aumenta el tiempo de entrenamiento.
            """
        )

st.markdown("---")

import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_iris
from pycaret.regression import setup as setup_reg, compare_models as compare_reg, pull as pull_reg
from pycaret.classification import setup as setup_cls, compare_models as compare_cls, pull as pull_cls

# --- Tytuł ---
st.markdown(
    "<h1 style='text-align: center;'>🔎 Znajdowanie najważniejszych cech w Twoim datasetcie</h1>",
    unsafe_allow_html=True
)

# --- Sidebar: wybór danych ---
data_source = st.sidebar.radio(
    "📊 Wybierz źródło danych:",
    ["Dane przykładowe (regresja)", "Dane przykładowe (klasyfikacja)", "Wczytaj własny plik CSV"]
)

df = pd.DataFrame()
dataset_name = ""

# --- Wczytanie danych ---
if data_source == "Dane przykładowe (regresja)":
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    dataset_name = "California Housing 🏠"
    st.write("Wybrano dane przykładowe: **California Housing**")

elif data_source == "Dane przykładowe (klasyfikacja)":
    iris = load_iris(as_frame=True)
    df = iris.frame
    df["target"] = iris.target  # kolumna numeryczna (0,1,2)
    dataset_name = "Iris 🌸"
    st.write("Wybrano dane przykładowe: **Iris**")

elif data_source == "Wczytaj własny plik CSV":
    uploaded_file = st.sidebar.file_uploader("📂 Wybierz plik CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        dataset_name = uploaded_file.name
        st.write(f"Wczytano plik: **{uploaded_file.name}** ✅")

# --- Wyświetlenie danych ---
if not df.empty:
    st.write(f"### 🔍 Podgląd danych ({dataset_name}) — pierwsze 5 wierszy")
    st.dataframe(df.head())

    # --- Wybór kolumny docelowej ---
    target_col = st.sidebar.selectbox(
        "🎯 Wybierz kolumnę docelową (target)",
        options=df.columns
    )
    st.write(f"**Wybrana kolumna docelowa:** `{target_col}`")

    # --- Rozpoznanie typu problemu ---
    target_dtype = df[target_col].dtype
    n_unique = df[target_col].nunique()

    if pd.api.types.is_numeric_dtype(target_dtype):
        if n_unique <= 10:
            problem_type = "klasyfikacja"
            problem_desc = "🧩 Problem klasyfikacji (wartości dyskretne)"
        else:
            problem_type = "regresja"
            problem_desc = "🧮 Problem regresji (wartości ciągłe)"
    else:
        problem_type = "klasyfikacja"
        problem_desc = "🧩 Problem klasyfikacji (wartości kategoryczne)"

    st.subheader("📘 Rozpoznanie problemu")
    st.info(f"Aplikacja rozpoznała, że to **{problem_desc}**.")
    st.write(f"🔢 Typ danych kolumny docelowej: `{target_dtype}`")
    st.write(f"🔹 Liczba unikalnych wartości: **{n_unique}**")

    # --- AUTOMATYCZNY WYBÓR MODELU ---
    if st.button("🚀 Uruchom automatyczny wybór najlepszego modelu"):
        with st.spinner("Trwa porównywanie modeli... ⏳"):
            if problem_type == "regresja":
                setup_reg(df, target=target_col, session_id=123, verbose=False)
                best_model = compare_reg()
                results = pull_reg()
            else:
                setup_cls(df, target=target_col, session_id=123, verbose=False)
                best_model = compare_cls()
                results = pull_cls()

        st.success("✅ Uczenie zakończone! Oto wyniki:")
        st.write("### 🏆 Ranking modeli:")
        st.dataframe(results)

        st.write("### 🌟 Najlepszy model:")
        st.write(best_model)

else:
    st.info("👉 Wybierz dane, aby kontynuować.")

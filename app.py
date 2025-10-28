import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_iris

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
        if n_unique <= 10:  # <=10 różnych wartości = klasyfikacja
            problem_type = "🧩 Problem klasyfikacji (wartości dyskretne)"
        else:
            problem_type = "🧮 Problem regresji (wartości ciągłe)"
    else:
        problem_type = "🧩 Problem klasyfikacji (wartości kategoryczne)"

    # --- Wyświetlenie informacji ---
    st.subheader("📘 Rozpoznanie problemu")
    st.info(f"Aplikacja rozpoznała, że to **{problem_type}**.")
    st.write(f"🔢 Typ danych kolumny docelowej: `{target_dtype}`")
    st.write(f"🔹 Liczba unikalnych wartości: **{n_unique}**")

else:
    st.info("👉 Wybierz dane, aby kontynuować.")

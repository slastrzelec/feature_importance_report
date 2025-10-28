import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing

# --- Tytuł wyjustowany ---
st.markdown(
    "<h1 style='text-align: center;'>Cześć, witaj w moim programie do znajdowania najważniejszych cech w twoim datasetcie :)</h1>",
    unsafe_allow_html=True
)

# --- Sidebar: wybór danych ---
data_source = st.sidebar.radio(
    "Wybierz źródło danych:",
    ["Dane przykładowe (regresja)", "Wczytaj własny plik CSV"]
)

if data_source == "Dane przykładowe (regresja)":
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    st.write("Wybrano dane przykładowe: California Housing")
elif data_source == "Wczytaj własny plik CSV":
    uploaded_file = st.sidebar.file_uploader("Wybierz plik CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"Wczytano plik: {uploaded_file.name}")
    else:
        df = pd.DataFrame()  # pusty DF, jeśli nie wybrano pliku

# --- Wyświetlenie danych ---
if not df.empty:
    st.write("### Podgląd danych (pierwsze 5 wierszy)")
    st.dataframe(df.head())

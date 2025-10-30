import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.datasets import fetch_california_housing, load_iris
from pycaret.regression import setup as setup_reg, compare_models as compare_reg, plot_model as plot_model_reg, pull as pull_reg
from pycaret.classification import setup as setup_cls, compare_models as compare_cls, plot_model as plot_model_cls, pull as pull_cls
import os
import datetime

# --- Lista ciekawostek ---
ai_facts = [
    "Sztuczna inteligencja jest używana w medycynie do diagnozowania chorób z obrazów medycznych.",
    "Pierwszy program szachowy został napisany już w latach 50-tych XX wieku.",
    "AI potrafi generować obrazy, muzykę i teksty w sposób niemal nieodróżnialny od ludzkich dzieł.",
    "Samouczenie maszynowe (Machine Learning) to poddziedzina AI, która pozwala komputerom uczyć się na danych.",
    "Sieci neuronowe inspirowane są strukturą i działaniem ludzkiego mózgu.",
    "AI jest wykorzystywana do prognozowania pogody i modelowania zmian klimatu."
]

# --- Tytuł ---
st.markdown(
    "<h1 style='text-align: center;'>🔎 Znajdowanie najważniejszych cech w Twoim datasetcie</h1>",
    unsafe_allow_html=True
)

# --- Sidebar: wybór danych ---
data_source = st.sidebar.radio(
    "📊 Wybierz źródło danych:",
    ["Dane przykładowe (regresja)", "Dane przykładowe (klasyfikacja)", "Wczytaj własny plik CSV"],
    key="data_source_radio"
)

# --- Session state ---
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'dataset_name' not in st.session_state:
    st.session_state.dataset_name = ""
if 'last_plot_path' not in st.session_state:
    st.session_state.last_plot_path = None

# --- Wczytanie danych ---
if data_source == "Dane przykładowe (regresja)":
    data = fetch_california_housing(as_frame=True)
    st.session_state.df = data.frame
    st.session_state.dataset_name = "California Housing 🏠"
    st.write("Wybrano dane przykładowe: **California Housing**")

elif data_source == "Dane przykładowe (klasyfikacja)":
    iris = load_iris(as_frame=True)
    st.session_state.df = iris.frame
    st.session_state.df["target"] = iris.target
    st.session_state.dataset_name = "Iris 🌸"
    st.write("Wybrano dane przykładowe: **Iris**")

elif data_source == "Wczytaj własny plik CSV":
    uploaded_file = st.sidebar.file_uploader("📂 Wybierz plik CSV", type="csv", key="file_uploader_key")
    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.dataset_name = uploaded_file.name
            st.write(f"Wczytano plik: **{uploaded_file.name}** ✅")
        except Exception as e:
            st.error(f"Błąd podczas wczytywania pliku: {e}")
            st.session_state.df = pd.DataFrame()
    else:
        st.session_state.df = pd.DataFrame()

# --- Podgląd danych ---
df = st.session_state.df
dataset_name = st.session_state.dataset_name

if not df.empty:
    st.write(f"### 🔍 Podgląd danych ({dataset_name}) — pierwsze 5 wierszy")
    st.dataframe(df.head())

    target_col = st.sidebar.selectbox(
        "🎯 Wybierz kolumnę docelową (target)",
        options=df.columns,
        key="target_selectbox"
    )
    st.write(f"**Wybrana kolumna docelowa:** `{target_col}`")

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

    # --- Automatyczny wybór modelu ---
    if st.button("🚀 Uruchom automatyczny wybór najlepszego modelu"):
        import time, random

        status_container = st.empty()
        progress_bar = st.progress(0, text="Rozpoczynanie procesu...")
        facts_to_show = random.sample(ai_facts, k=min(len(ai_facts), 4))
        for i, fact in enumerate(facts_to_show):
            progress_bar.progress(int((i + 1) * 100 / (len(facts_to_show) + 1)), text=f"⏳ {fact}")
            time.sleep(1.2)
        progress_bar.progress(100, text="Trwa uruchamianie modeli PyCaret...")

        with st.spinner("Porównywanie modeli..."):
            try:
                if problem_type == "regresja":
                    setup_reg(df, target=target_col, session_id=123, verbose=False, use_gpu=False)
                    best_model = compare_reg(n_select=1)
                    pull_reg()
                    plot_model_reg(best_model, plot="feature", save=True)
                else:
                    setup_cls(df, target=target_col, session_id=123, verbose=False, use_gpu=False)
                    best_model = compare_cls(n_select=1)
                    pull_cls()
                    plot_model_cls(best_model, plot="feature", save=True)

                # --- Unikalna nazwa pliku wykresu ---
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                new_path = f"feature_importance_{timestamp}.png"
                os.replace("Feature Importance.png", new_path)
                st.session_state.last_plot_path = new_path

                # --- Wyświetlenie ---
                img = Image.open(new_path)
                st.image(img, caption="🌟 Najważniejsze cechy modelu", use_column_width=True)
                st.success("✅ Wykres został wygenerowany pomyślnie!")

            except Exception as e:
                st.error(f"Wystąpił błąd: {e}")

    # --- Sekcja zapisu wykresu ---
    if st.session_state.last_plot_path:
        st.markdown("---")
        st.subheader("💾 Zapis wykresu")

        with open(st.session_state.last_plot_path, "rb") as file:
            st.download_button(
                label="📥 Pobierz wykres (PNG)",
                data=file,
                file_name=os.path.basename(st.session_state.last_plot_path),
                mime="image/png"
            )

else:
    st.info("👉 Wybierz dane, aby kontynuować.")

# --- Instrukcja ---
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Instrukcja:**
1. Wybierz dane (przykładowe lub własny plik CSV).  
2. Wybierz kolumnę docelową (target).  
3. Kliknij **Uruchom automatyczny wybór najlepszego modelu**.  
4. Po wygenerowaniu wykresu kliknij **📥 Pobierz wykres (PNG)**, aby go zapisać lokalnie.  
""")

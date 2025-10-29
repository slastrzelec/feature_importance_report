import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_iris
# Importowanie PyCaret
from pycaret.regression import setup as setup_reg, compare_models as compare_reg, pull as pull_reg
from pycaret.classification import setup as setup_cls, compare_models as compare_cls, pull as pull_cls

# Przywrócone importy, używane lokalnie w sekcji trenowania
# import random 
# import time 

# W Streamlit, użycie threading do aktualizacji UI nie jest wspierane i jest niebezpieczne.

# --- Lista ciekawostek o AI (teraz będą używane w innym miejscu) ---
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

# Użycie st.session_state do przechowywania danych i stanu
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'dataset_name' not in st.session_state:
    st.session_state.dataset_name = ""

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


# --- Wyświetlenie danych i wybór kolumny docelowej ---
df = st.session_state.df
dataset_name = st.session_state.dataset_name

if not df.empty:
    st.write(f"### 🔍 Podgląd danych ({dataset_name}) — pierwsze 5 wierszy")
    st.dataframe(df.head())

    # --- Wybór kolumny docelowej ---
    target_col = st.sidebar.selectbox(
        "🎯 Wybierz kolumnę docelową (target)",
        options=df.columns,
        key="target_selectbox"
    )
    st.write(f"**Wybrana kolumna docelowa:** `{target_col}`")

    # --- Rozpoznanie typu problemu ---
    target_dtype = df[target_col].dtype
    n_unique = df[target_col].nunique()

    # Logika PyCaret jest bardziej skomplikowana niż to, ale dla uproszczenia:
    # Używamy n_unique <= 10 jako heurystyki dla klasyfikacji numerycznej.
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
    if st.button("🚀 Uruchom automatyczny wybór najlepszego modelu", key="run_model_button"):
        
        # Importy niezbędne do symulacji dynamiki
        import random
        import time

        # 1. Symulacja dynamicznych ciekawostek i postępu (symulacja czasu oczekiwania)
        status_container = st.empty()
        progress_bar = st.progress(0, text="Rozpoczynanie procesu...")
        
        # Wybieramy kilka losowych faktów do rotacji
        facts_to_show = random.sample(ai_facts, k=min(len(ai_facts), 4))
        num_steps = len(facts_to_show)
        
        for i, fact in enumerate(facts_to_show):
            progress_value = int((i + 1) * 100 / (num_steps + 1)) # Ostatni krok będzie 100% po pętli
            progress_bar.progress(progress_value, text=f"Trwa przetwarzanie... ⏳ **Ciekawostka AI:** {fact}")
            time.sleep(1.5) # Krótka pauza, by użytkownik zobaczył zmianę
        
        progress_bar.progress(100, text="Wszystko gotowe! Uruchamianie algorytmów PyCaret...")
        
        # Oczyszczamy kontener statusu przed blokującym spinnerem
        progress_bar.empty()
        status_container.empty()
        
        # 2. PyCaret Execution (Blocking part)
        with st.spinner("Trwa porównywanie modeli PyCaret. Może to chwilę potrwać, dziękujemy za cierpliwość."):
            
            # W Streamlit musimy wywoływać funkcje PyCaret sekwencyjnie w głównym wątku
            if problem_type == "regresja":
                try:
                    # Setup musi być wykonany, aby skonfigurować środowisko PyCaret
                    setup_reg(df, target=target_col, session_id=123, 
                              # Ustawienie verbose=False jest kluczowe, 
                              # aby Streamlit nie był zalany konsolowym wyjściem PyCaret.
                              verbose=False, use_gpu=False, n_jobs=1)
                    
                    # compare_models trenuje i porównuje modele
                    # n_select=3 wybiera top 3 modeli
                    best_model = compare_reg(n_select=3)
                    
                    # pull pobiera wyniki porównania
                    results = pull_reg()
                    
                except Exception as e:
                    st.error(f"Wystąpił błąd podczas trenowania regresji: {e}")
                    results = pd.DataFrame()
                    best_model = None

            else: # Klasyfikacja
                try:
                    setup_cls(df, target=target_col, session_id=123, 
                              verbose=False, use_gpu=False, n_jobs=1)
                    
                    best_model = compare_cls(n_select=3)
                    results = pull_cls()
                    
                except Exception as e:
                    st.error(f"Wystąpił błąd podczas trenowania klasyfikacji: {e}")
                    results = pd.DataFrame()
                    best_model = None

        # --- Wyświetlenie wyników ---
        if not results.empty:
            st.success("✅ Uczenie zakończone! Oto wyniki:")
            st.write("### 🏆 Ranking modeli (Top 3):")
            # Dodajemy stylizację dla lepszego wyglądu tabeli
            st.dataframe(results.style.highlight_max(axis=0, color='yellow'))

            st.write("### 🌟 Najlepszy model (pierwszy z listy):")
            # Pokazanie najlepszego modelu PyCaret
            st.code(str(best_model[0]))
        else:
            st.warning("Nie udało się uzyskać wyników modeli. Sprawdź konsolę pod kątem błędów PyCaret.")

else:
    st.info("👉 Wybierz dane, aby kontynuować.")

# --- Instrukcja dla użytkownika ---
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Instrukcja:**
    1. Wybierz dane (przykładowe lub własny plik CSV).
    2. Wybierz kolumnę docelową (target) z listy po lewej.
    3. Kliknij **Uruchom automatyczny wybór najlepszego modelu**.
    
    Aplikacja automatycznie rozpozna, czy jest to problem regresji (wartości ciągłe) czy klasyfikacji (wartości dyskretne), a następnie użyje biblioteki PyCaret do porównania kilkunastu modeli ML.
    """
)

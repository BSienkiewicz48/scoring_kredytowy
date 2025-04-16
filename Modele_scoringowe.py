import streamlit as st
import pandas as pd
import numpy as np

# Funkcja do czyszczenia kolumn procentowych i kwotowych
def clean_data(df):
    # Usuwamy znaki % i przekształcamy do float
    percent_cols = ['oproc_refin', 'oproc_konkur', 'koszt_pieniadza', 'oproc_propon']
    for col in percent_cols:
        df[col] = df[col].str.replace('%', '').astype(float)

    # Kwota kredytu – usunięcie $ i przecinków, zamiana na float
    df['kwota_kredytu'] = df['kwota_kredytu'].replace('[\$,]', '', regex=True).astype(float)

    # Data – zamiana na datetime
    df['data_akceptacji'] = pd.to_datetime(df['data_akceptacji'], dayfirst=True)

    return df

# Tytuł aplikacji
st.title("📊 Aplikacja Scoringowa – Eksploracja Danych")

# Wczytywanie pliku
uploaded_file = st.file_uploader("Wgraj plik Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # Czyszczenie danych
    df = clean_data(df)

    # Wyświetlenie danych
    st.subheader("📌 Podgląd danych")
    st.dataframe(df.head())

    st.subheader("🔎 Informacje o danych")
    st.write(df.info())

    st.subheader("📈 Statystyki opisowe")
    st.write(df.describe(include='all'))

else:
    st.info("📂 Wgraj plik Excel, aby rozpocząć analizę.")




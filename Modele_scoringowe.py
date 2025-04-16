import streamlit as st
import pandas as pd

# Funkcja czyszcząca dane
def clean_data(df):
    df['kwota_kredytu'] = df['kwota_kredytu'].replace('[\$,]', '', regex=True).astype(float)
    for col in ['oproc_refin', 'oproc_konkur', 'koszt_pieniadza', 'oproc_propon']:
        # Ensure column is string, replace '%', then convert to numeric, coercing errors
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', '', regex=False), errors='coerce')
    df['data_akceptacji'] = pd.to_datetime(df['data_akceptacji'], dayfirst=True)
    return df

# Tytuł aplikacji
st.title("📊 Scoring kredytowy – eksploracja danych")

# Wczytanie danych z pliku w repo
df = pd.read_excel("kredyty_auto_Scoring2025s.xlsx")
df = clean_data(df)

# Wyświetlanie danych
st.subheader("📌 Podgląd danych")
st.dataframe(df.head())

st.subheader("🔍 Informacje o danych")
st.text(df.info())

st.subheader("📈 Statystyki opisowe")
st.write(df.describe())




import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Funkcja czyszcząca dane
def clean_data(df):
    df['kwota_kredytu'] = df['kwota_kredytu'].replace('[\$,]', '', regex=True).astype(float)
    for col in ['oproc_refin', 'oproc_konkur', 'koszt_pieniadza', 'oproc_propon']:
        # Ensure column is string, replace '%', then convert to numeric, coercing errors
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', '', regex=False), errors='coerce')
    df['data_akceptacji'] = pd.to_datetime(df['data_akceptacji'], dayfirst=True).dt.date
    return df

# Tytuł aplikacji
st.title("📊 Scoring kredytowy – eksploracja danych")

st.markdown("Celem tego narzędzia jest przewidywanie, czy dany klient zaakceptuje przedstawioną ofertę kredytową. Model zwraca wynik punktowy – im wyższy wynik, tym większe prawdopodobieństwo, że klient skorzysta z oferty. ")

# Wczytanie danych z pliku w repo
df = pd.read_excel("kredyty_auto_Scoring2025s.xlsx")
df = clean_data(df)

# Wyświetlanie danych
st.subheader("📌 Podgląd danych")
st.dataframe(df.drop(columns=['LP']), height=400, use_container_width=True)

st.subheader("🔍 Informacje o danych")
st.markdown("""
- **LP** – numer porządkowy wiersza.  
- **data_akceptacji** – data akceptacji wniosku kredytowego przez bank.  
- **grupa_ryzyka** – oznaczenie grupy ryzyka kredytowego klienta według klasyfikacji banku.  
- **kod_partnera** – identyfikator partnera biznesowego (sieci dealerów samochodowych).  
- **typ_umowy** – typ umowy:  
  - „N” – nowy samochód,  
  - „U” – samochód używany,  
  - „R” – refinansowanie kredytu (nowy kredyt na spłatę stałego).  
- **scoring_FICO** – ocena punktowa FICO (odpowiednik polskiego scoringu BIK).  
- **okres_kredytu** – długość okresu kredytowania w miesiącach (np. 48, 72, 60).  
- **kwota_kredytu** – kwota przyznanego kredytu (np. $26,500).  
- **oproc_refin** – oprocentowanie kredytu finansowego (dla typu umowy „R”).  
- **oproc_konkur** – oprocentowanie oferowane przez konkurencję (najlepsza stopa procentowa konkurenta, dane prawdopodobnie pochodzą od partnerów).  
- **koszt_pieniadza** – koszt pozyskania środków dla banku (np. 1.10%).  
- **oproc_propon** – oprocentowanie proponowane klientowi przez bank.  
- **akceptacja_klienta** – wynik akceptacji klienta (0 = brak akceptacji, 1 = akceptacja) - zmienna celu.  
""")

st.subheader("📈 Statystyki opisowe")
# Wybór kolumn numerycznych, które mają sens dla statystyk opisowych
numeric_columns = ['scoring_FICO', 'okres_kredytu', 'kwota_kredytu', 
                   'oproc_refin', 'oproc_konkur', 'koszt_pieniadza', 'oproc_propon']
st.write(df[numeric_columns].describe())


st.subheader("🎻 Wizualizacja wybranych zmiennych")

# Wykres violinowy dla scoring_FICO i okres_kredytu obok siebie
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Wykres dla scoring_FICO
sns.violinplot(data=df, y='scoring_FICO', ax=axes[0], color="blue")
axes[0].set_ylabel("scoring_FICO")
axes[0].set_title("Scoring_FICO")

# Wykres dla okres_kredytu
sns.violinplot(data=df, y='okres_kredytu', ax=axes[1], color="orange")
axes[1].set_ylabel("okres_kredytu")
axes[1].set_title("Okres_kredytu")

st.pyplot(fig)

# Wykres violinowy dla oproc_refin, oproc_konkur, koszt_pieniadza, oproc_propon z większą wysokością
fig2, ax2 = plt.subplots(figsize=(10, 10))
sns.violinplot(data=df[['oproc_konkur', 'koszt_pieniadza', 'oproc_propon']], ax=ax2)
ax2.set_title("Oprocentowania i koszt pieniądza")
ax2.set_ylim(0.01, 0.12)  # Ustawienie zakresu osi Y od 0 do 0.12
ax2.yaxis.set_major_locator(plt.MultipleLocator(0.005))  # Oznaczenia osi Y co 0.005
st.pyplot(fig2)

st.markdown("Wykresy violinowe okazują nie tylko minimum, maksimum i medianę, ale też jak „tłoczno” jest w różnych częściach rozkładu. Dzięki temu możemy szybko zorientować się, gdzie skupiają się obserwacje, a gdzie robi się luźniej.")

# Tworzenie zmiennych pochodnych
df['spread'] = df['oproc_propon'] - df['oproc_konkur']
df['margin'] = df['oproc_propon'] - df['koszt_pieniadza']
df['rata_miesieczna'] = df['kwota_kredytu'] / df['okres_kredytu']
df['intensity_rate'] = df['rata_miesieczna'] / df['scoring_FICO']

st.markdown("""
#### ✨ Nowe zmienne pochodne
Na podstawie danych pierwotnych stworzono dodatkowe cechy:
- `spread` – różnica między oprocentowaniem banku a konkurencją,
- `margin` – marża banku względem kosztu pozyskania środków,
- `rata_miesieczna` – szacunkowa wysokość miesięcznej raty,
- `intensity_rate` – relacja raty miesięcznej do scoringu FICO (im wyższa, tym większe „obciążenie” dla klienta).
""")

from category_encoders.woe import WOEEncoder
from sklearn.preprocessing import KBinsDiscretizer

# Funkcja obliczająca IV dla jednej zmiennej
def calculate_iv(df, feature, target, bins=10):
    # Tworzymy kopię danych
    df_temp = df[[feature, target]].dropna()
    
    # Jeżeli zmienna jest liczbowa – binujemy ją
    if pd.api.types.is_numeric_dtype(df_temp[feature]):
        df_temp['bin'] = pd.qcut(df_temp[feature], q=bins, duplicates='drop')
    else:
        df_temp['bin'] = df_temp[feature]
    
    # Obliczamy statystyki dla każdej grupy
    grouped = df_temp.groupby('bin')
    total_good = (df_temp[target] == 1).sum()
    total_bad = (df_temp[target] == 0).sum()
    
    iv = 0
    for name, group in grouped:
        good = (group[target] == 1).sum()
        bad = (group[target] == 0).sum()
        if good > 0 and bad > 0:
            dist_good = good / total_good
            dist_bad = bad / total_bad
            iv += (dist_good - dist_bad) * np.log(dist_good / dist_bad)
    
    return iv

# Lista zmiennych do oceny
features_to_check = ['scoring_FICO', 'okres_kredytu', 'kwota_kredytu',
                     'oproc_refin', 'oproc_konkur', 'koszt_pieniadza', 'oproc_propon',
                     'spread', 'margin', 'rata_miesieczna', 'intensity_rate']

# Obliczenie IV dla każdej zmiennej
iv_dict = {feature: calculate_iv(df, feature, 'akceptacja_klienta') for feature in features_to_check}
iv_series = pd.Series(iv_dict).sort_values(ascending=False)


st.subheader("📊 Siła predykcyjna zmiennych (IV)")
st.markdown("Wykres poniżej pokazuje, które zmienne najlepiej rozróżniają klientów, którzy zaakceptowali ofertę, od tych, którzy jej nie przyjęli.")
fig_iv, ax_iv = plt.subplots(figsize=(10, 6))
sns.barplot(x=iv_series.values, y=iv_series.index, palette="viridis", ax=ax_iv)
ax_iv.set_xlabel("Information Value (IV)")
ax_iv.set_ylabel("Zmienna")
ax_iv.set_title("Information value zmiennych")
st.pyplot(fig_iv)

st.markdown("""
**Interpretacja IV (Information Value):**
- IV < 0.02 – brak predykcyjnej mocy  
- 0.02–0.1 – słaba  
- 0.1–0.3 – średnia  
- 0.3–0.5 – silna  
- \>0.5 – bardzo silna
""")

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from category_encoders.woe import WOEEncoder
from sklearn.preprocessing import KBinsDiscretizer
from optbinning import OptimalBinning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.calibration import CalibratedClassifierCV
import joblib
import io

# Funkcja czyszcząca dane
def clean_data(df):
    df['kwota_kredytu'] = df['kwota_kredytu'].replace('[\$,]', '', regex=True).astype(float)
    for col in ['oproc_refin', 'oproc_konkur', 'koszt_pieniadza', 'oproc_propon']:
        # Ensure column is string, replace '%', then convert to numeric, coercing errors
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', '', regex=False), errors='coerce')
    df['data_akceptacji'] = pd.to_datetime(df['data_akceptacji'], dayfirst=True).dt.date
    return df

# Tytuł aplikacji
st.title("📊 Scoring prawdopodobieństwa przyjęcia kredytu")

st.markdown("Celem tego narzędzia jest przewidywanie, czy dany klient zaakceptuje przedstawioną ofertę kredytową. Model zwraca wynik punktowy – im wyższy wynik, tym większe prawdopodobieństwo, że klient skorzysta z oferty. ")

# Funkcja do wczytania i przygotowania danych z cache
@st.cache_data
def load_and_clean_data():
  df = pd.read_excel("kredyty_auto_Scoring2025s.xlsx")
  df = clean_data(df)
  return df

# Wczytanie danych z wykorzystaniem cache
df = load_and_clean_data()

# Wyświetlanie danych
st.subheader("📌 Podgląd danych")
st.dataframe(df.drop(columns=['LP']), height=400, use_container_width=True, hide_index=True)

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

features_to_check = ['scoring_FICO', 'okres_kredytu', 'kwota_kredytu',
                     'oproc_refin', 'oproc_konkur', 'koszt_pieniadza', 'oproc_propon',
                     'spread', 'margin', 'rata_miesieczna', 'intensity_rate']

iv_dict = {}
binning_tables = {}

@st.cache_data
def calculate_iv_binning(df, features_to_check):
    iv_dict = {}
    binning_tables = {}

    for feature in features_to_check:
        df_temp = df[[feature, 'akceptacja_klienta']].dropna()

        # Tworzenie binów kwantylowych
        try:
            df_temp['bin'] = pd.qcut(df_temp[feature], q=10, duplicates='drop')
        except ValueError:
            df_temp['bin'] = pd.qcut(df_temp[feature], q=5, duplicates='drop')

        # Liczenie total good/bad
        total_good = (df_temp['akceptacja_klienta'] == 1).sum()
        total_bad = (df_temp['akceptacja_klienta'] == 0).sum()

        iv = 0
        table_data = []

        for name, group in df_temp.groupby('bin'):
            good = (group['akceptacja_klienta'] == 1).sum()
            bad = (group['akceptacja_klienta'] == 0).sum()

            if good > 0 and bad > 0:
                dist_good = good / total_good
                dist_bad = bad / total_bad
                woe = np.log(dist_good / dist_bad)
                iv_bin = (dist_good - dist_bad) * woe
                iv += iv_bin
            else:
                woe = 0
                iv_bin = 0

            table_data.append({
                'Przedział': str(name),
                'Good': good,
                'Bad': bad,
                'WOE': round(woe, 4),
                'IV_bin': round(iv_bin, 4)
            })

        iv_dict[feature] = iv
        binning_tables[feature] = pd.DataFrame(table_data)

    return iv_dict, binning_tables

iv_dict, binning_tables = calculate_iv_binning(df, features_to_check)

# Posortuj zmienne wg IV malejąco
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
- < 0.02 – brak predykcyjnej mocy  
- 0.02–0.1 – słaba  
- 0.1–0.3 – średnia  
- 0.3–0.5 – silna  
- \\> 0.5 – bardzo silna  

---

**Zasady preselekcji zmiennych do modelu:**  
W celu wyboru zmiennych do modelu scoringowego zastosowano kryterium wartości IV.  
Do dalszego modelowania zakwalifikowano zmienne, dla których IV przekroczyło próg 0.02 – co oznacza, że posiadają co najmniej słabą zdolność do rozróżniania klas (akceptacja vs brak akceptacji).  
Dzięki temu model wykorzystuje tylko zmienne niosące istotną informację predykcyjną, co poprawia jego interpretowalność i stabilność.
""")

st.subheader("📄 Szczegóły binowania i WOE")

selected_var = st.selectbox("Wybierz zmienną, aby zobaczyć tabelę binów:", iv_series.index.tolist())
if selected_var:
    table = binning_tables[selected_var]
    st.dataframe(table, use_container_width=True, hide_index=True)

st.markdown("""
**Opis tabeli binowania:**

Poniższa tabela przedstawia statystyki dla każdego przedziału (binu), na które została podzielona zmienna.  
- **Przedział** – zakres wartości w danym binie (ustalony metodą kwantylową).  
- **Good / Bad** – liczba obserwacji z klasą 1 (zaakceptowana oferta) i 0 (odmowa) w tym przedziale.  
- **WOE (Weight of Evidence)** – miara siły rozróżnienia między klasami.  
  - Dodatnie WOE → przewaga „good”  
  - Ujemne WOE → przewaga „bad”  
  - Im dalej od zera, tym silniejsza różnicująca moc binu  
- **IV_bin** – wkład danego binu do całkowitego Information Value zmiennej.  
  - Im wyższy, tym większe znaczenie danego przedziału dla modelu.

WOE i IV są używane w modelach scoringowych opartych na regresji logistycznej, aby przekształcić dane w bardziej informatywny i stabilny sposób.
""")





@st.cache_data
def train_woe_model(df, target_col, features):
    # WOE transformacja
    encoder = WOEEncoder(cols=features)
    X = df[features]
    y = df[target_col]
    X_woe = encoder.fit_transform(X, y)

    # Podział na zbiór treningowy i testowy (90/10)
    X_train, X_test, y_train, y_test = train_test_split(
        X_woe, y, test_size=0.1, random_state=42, stratify=y
    )

    # Regresja logistyczna
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predykcja i ocena
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    gini = 2 * auc - 1
    error = np.mean(np.abs(y_test - y_pred_proba)) * 100  # średni błąd procentowy

    # Scorecard (czyli tabela z predykcjami i score)
    scorecard = pd.DataFrame({
        'Prawdziwa klasa': y_test.values,
        'Prawdopodobieństwo': np.round(y_pred_proba, 4),
        'Decyzja modelu': (y_pred_proba >= 0.5).astype(int)
    })

    return model, encoder, auc, gini, error, scorecard, X_test, y_test, y_pred_proba

# Wybierz zmienne do modelu (np. te z IV > 0.02)
features_for_model = [col for col in iv_series.index if iv_series[col] > 0.02]

st.subheader("🧪 Budowa modelu scoringowego WOE + regresja logistyczna")

st.markdown(f"""
Model został wytrenowany na zbiorze treningowym z losowym podziałem 90/10.  
Wykorzystano transformację WOE na zmiennych o wartości IV > 0.02.  
Wybrano {len(features_for_model)} zmiennych:  
**{', '.join(features_for_model)}**

W przypadku modelu WOE + regresja logistyczna wynik scoringowy uzyskano poprzez bezpośrednie przemnożenie prawdopodobieństwa przewidzianego przez model przez 100.  
Otrzymany wynik reprezentuje więc prawdopodobieństwo akceptacji oferty w skali 0–100 co jest jednocześnie scoringiem.
""")

model, encoder, auc, gini, error, scorecard, X_test, y_test, y_pred_proba = train_woe_model(
    df, "akceptacja_klienta", features_for_model
)

# Metryki modelu
st.subheader("📊 Ocena modelu")
st.markdown(f"""
- **AUC**: {round(auc, 4)}  
- **Gini**: {round(gini, 4)}  
- **Średni błąd predykcji**: {round(error, 2)}%
""")

# Wykres ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
ax_roc.plot([0, 1], [0, 1], 'k--')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('Krzywa ROC')
ax_roc.legend(loc='lower right')
st.pyplot(fig_roc)

# Tabela z wynikami
st.subheader("📋 Tabela predykcji")
st.markdown("Poniżej znajdują się przykładowe predykcje modelu na zbiorze testowym.")
# Przelicz prawdopodobieństwo na score (0-100)
scorecard['Score'] = np.round(scorecard['Prawdopodobieństwo'] * 100).astype(int)
# Recreate scorecard with index to easily merge with original features
scorecard_indexed = pd.DataFrame({
    'Prawdziwa klasa': y_test,
    'Prawdopodobieństwo': y_pred_proba
}, index=y_test.index)
scorecard_indexed['Decyzja modelu'] = (scorecard_indexed['Prawdopodobieństwo'] >= 0.5).astype(int)
scorecard_indexed['Score'] = np.round(scorecard_indexed['Prawdopodobieństwo'] * 100).astype(int)

# Get original features for the test set using the index
original_features_test = df.loc[y_test.index, features_for_model]

# Combine Score, original features, and other scorecard columns
# Ensure 'Score' is the first column
scorecard_display = pd.concat([
    scorecard_indexed[['Score']],
    original_features_test,
    scorecard_indexed[['Prawdziwa klasa', 'Decyzja modelu']]
], axis=1)

# Display the enhanced scorecard
st.dataframe(scorecard_display, height=400, use_container_width=True, hide_index=True)


st.subheader("🧮 Klasyczna karta scoringowa")


def build_scorecard(encoder, model, binning_tables):
    scorecard_rows = []

    # Słownik: zmienna → współczynnik regresji
    coefs = dict(zip(encoder.cols, model.coef_[0]))

    for feature in encoder.cols:
        coef = coefs[feature]
        table = binning_tables[feature]

        for i, row in table.iterrows():
            bin_label = row["Przedział"]
            woe = row["WOE"]
            waga = round(coef * woe, 4)
            scorecard_rows.append({
                "Zmienna": feature,
                "Przedział": bin_label,
                "WOE": woe,
                "Współczynnik RL": round(coef, 4),
                "Waga modelu (WOE × coef)": waga
            })

    scorecard_df = pd.DataFrame(scorecard_rows)
    return scorecard_df

scorecard_df = build_scorecard(encoder, model, binning_tables)

st.markdown("""
Tabela poniżej przedstawia klasyczną kartę scoringową:  
- Każda zmienna została podzielona na przedziały (biny)  
- Dla każdego przedziału obliczono wartość WOE  
- Następnie wyliczono „wagę” modelu: WOE × współczynnik regresji  
Im wyższa wartość – tym bardziej pozytywny wpływ danego przedziału na wynik modelu.
""")

st.dataframe(scorecard_df, use_container_width=True, hide_index=True)

@st.cache_data
def train_xgboost_model(df, target_col, features):
    X = df[features].copy()
    y = df[target_col]

    # Podział na trening/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    # Oblicz balans klas
    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    # Bazowy model XGBoost
    base_model = xgb.XGBClassifier(
        n_estimators=60,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=ratio,
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42
    )

    # Kalibracja przy użyciu Platt Scaling
    model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')

    # Najpierw fitujemy bazowy model
    base_model.fit(X_train, y_train)

    # Potem dopiero kalibrujemy (model wymaga wytrenowanego base_model)
    model.fit(X_train, y_train)

    # Predykcje skalibrowane
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    gini = 2 * auc - 1

    return model, auc, gini, y_pred_proba, y_test, X_test

# =======================
# 🔮 Trening modelu XGBoost
# =======================

st.subheader("🤖 Model scoringowy XGBoost")

# Usuń 'kwota_kredytu' z listy cech dla XGBoost
features_for_xgb_model = [f for f in features_for_model]

model_xgb, auc_xgb, gini_xgb, y_pred_proba_xgb, y_test_xgb, X_test_xgb = train_xgboost_model(
    df, "akceptacja_klienta", features_for_xgb_model
)

st.markdown(f"""
Model XGBoost został wytrenowany na tych samych zmiennych co model WOE + RL.  
Do kalibracji modelu zastosowano metodę **Platt Scaling** (dokładniej: **isotonic regression**) przy użyciu klasy `CalibratedClassifierCV`.  
Dzięki temu wyjściowe prawdopodobieństwa modelu zostały dopasowane do rozkładu obserwowanego na zbiorze treningowym,  
a wynik scoringowy reprezentuje już skalibrowane prawdopodobieństwo akceptacji oferty.  

**Wyniki modelu:**
- **AUC**: {round(auc_xgb, 4)}
- **Gini**: {round(gini_xgb, 4)}
""")


# ROC
fpr_xgb, tpr_xgb, _ = roc_curve(y_test_xgb, y_pred_proba_xgb)
fig_roc_xgb, ax_roc_xgb = plt.subplots()
ax_roc_xgb.plot(fpr_xgb, tpr_xgb, label=f'AUC = {auc_xgb:.3f}')
ax_roc_xgb.plot([0, 1], [0, 1], 'k--')
ax_roc_xgb.set_xlabel('False Positive Rate')
ax_roc_xgb.set_ylabel('True Positive Rate')
ax_roc_xgb.set_title('Krzywa ROC – XGBoost')
ax_roc_xgb.legend(loc='lower right')
st.pyplot(fig_roc_xgb)

# =======================
# 📋 Scorecard XGBoost – tabela predykcji
# =======================

st.subheader("📋 Tabela predykcji – XGBoost")
st.markdown("Poniżej znajdują się przykładowe predykcje modelu XGBoost na zbiorze testowym.")

# Stwórz DataFrame z wynikiem
scorecard_xgb_indexed = pd.DataFrame({
    'Prawdziwa klasa': y_test_xgb,
    'Prawdopodobieństwo': y_pred_proba_xgb
}, index=y_test_xgb.index)

scorecard_xgb_indexed['Decyzja modelu'] = (scorecard_xgb_indexed['Prawdopodobieństwo'] >= 0.5).astype(int)
scorecard_xgb_indexed['Score'] = np.round(scorecard_xgb_indexed['Prawdopodobieństwo'] * 100).astype(int)

# Pobierz cechy oryginalne dla testu
original_features_test_xgb = df.loc[y_test_xgb.index, features_for_model]

# Połącz Score, cechy i wyniki
scorecard_xgb_display = pd.concat([
    scorecard_xgb_indexed[['Score']],
    original_features_test_xgb,
    scorecard_xgb_indexed[['Prawdziwa klasa', 'Decyzja modelu']]
], axis=1)

# Wyświetl tabelę
st.dataframe(scorecard_xgb_display, height=400, use_container_width=True, hide_index=True)




# ✅ Nowy model XGBoost z dodatkowymi kolumnami WOE
@st.cache_data
def train_xgboost_model_with_woe(df, target_col, features, _encoder):
    # Przygotowanie danych surowych i WOE
    X_raw = df[features].copy()
    X_woe = _encoder.transform(df[features]) # Use _encoder here

    # Złącz dane: surowe + WOE (zmienne z sufiksem _woe)
    X_woe.columns = [f"{col}_woe" for col in X_woe.columns]
    X_combined = pd.concat([X_raw.reset_index(drop=True), X_woe.reset_index(drop=True)], axis=1)
    y = df[target_col].reset_index(drop=True)

    # Podział
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.1, random_state=42, stratify=y
    )

    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    # Bazowy model XGBoost
    base_model = xgb.XGBClassifier(
        n_estimators=60,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=ratio,
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42
    )

    # Kalibracja przy użyciu Platt Scaling
    model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')

    # Najpierw fitujemy bazowy model
    base_model.fit(X_train, y_train)

    # Potem dopiero kalibrujemy (model wymaga wytrenowanego base_model)
    model.fit(X_train, y_train)

    # Predykcje skalibrowane
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    gini = 2 * auc - 1

    return model, auc, gini, y_pred_proba, y_test, X_test, X_combined.columns.tolist()

# =======================
# 🤖 Model XGBoost (surowe + WOE)
# =======================

st.subheader("🤖 Model XGBoost z dodatkowymi zmiennymi WOE")

model_xgb_woe, auc_xgb_woe, gini_xgb_woe, y_pred_xgb_woe, y_test_xgb_woe, X_test_xgb_woe, full_feature_list = train_xgboost_model_with_woe(
    df, "akceptacja_klienta", features_for_model, encoder
)

st.markdown(f"""
Model XGBoost został wytrenowany na zestawie cech zawierającym zarówno surowe zmienne, jak i ich odpowiedniki po transformacji WOE.  
Do kalibracji modelu wykorzystano metodę **Platt Scaling (isotonic regression)** poprzez `CalibratedClassifierCV`.  
Dzięki temu predykcje modelu są dopasowane do rozkładu rzeczywistych odpowiedzi i można je traktować jako skalibrowane prawdopodobieństwo akceptacji oferty.  

**Wyniki modelu:**
- **AUC**: {round(auc_xgb_woe, 4)}
- **Gini**: {round(gini_xgb_woe, 4)}
""")


fpr_woe, tpr_woe, _ = roc_curve(y_test_xgb_woe, y_pred_xgb_woe)
fig_roc_woe, ax_roc_woe = plt.subplots()
ax_roc_woe.plot(fpr_woe, tpr_woe, label=f'AUC = {auc_xgb_woe:.3f}')
ax_roc_woe.plot([0, 1], [0, 1], 'k--')
ax_roc_woe.set_xlabel('False Positive Rate')
ax_roc_woe.set_ylabel('True Positive Rate')
ax_roc_woe.set_title('Krzywa ROC – XGBoost (z WOE)')
ax_roc_woe.legend(loc='lower right')
st.pyplot(fig_roc_woe)

# =======================
# 📋 Tabela predykcji – XGBoost z WOE
# =======================
st.subheader("📋 Tabela predykcji – XGBoost (z WOE)")
st.markdown("Poniżej znajdują się przykładowe predykcje modelu XGBoost na zbiorze testowym z dodatkowymi cechami WOE.")

scorecard_xgb_woe_indexed = pd.DataFrame({
    'Prawdziwa klasa': y_test_xgb_woe,
    'Prawdopodobieństwo': y_pred_xgb_woe
}, index=y_test_xgb_woe.index)

scorecard_xgb_woe_indexed['Decyzja modelu'] = (scorecard_xgb_woe_indexed['Prawdopodobieństwo'] >= 0.5).astype(int)
scorecard_xgb_woe_indexed['Score'] = np.round(scorecard_xgb_woe_indexed['Prawdopodobieństwo'] * 100).astype(int)

# Pobranie cech do scorecardu z X_test_xgb_woe, który zawiera już połączone cechy dla zbioru testowego
# Upewnij się, że X_test_xgb_woe jest DataFrame z odpowiednimi nazwami kolumn
if not isinstance(X_test_xgb_woe, pd.DataFrame):
    X_test_xgb_woe = pd.DataFrame(X_test_xgb_woe, columns=full_feature_list, index=y_test_xgb_woe.index)
else:
    # Ensure index matches y_test_xgb_woe if it was reset during split
    X_test_xgb_woe.index = y_test_xgb_woe.index


# Wybierz oryginalne cechy z X_test_xgb_woe
original_features_test_woe = X_test_xgb_woe[features_for_model].copy()

# Wybierz cechy WOE z X_test_xgb_woe
woe_feature_names = [f"{col}_woe" for col in features_for_model]
woe_features_test = X_test_xgb_woe[woe_feature_names].copy()

# Resetuj indeksy wszystkich części przed połączeniem, aby zapewnić wyrównanie
scorecard_xgb_woe_indexed.reset_index(drop=True, inplace=True)
original_features_test_woe.reset_index(drop=True, inplace=True)
woe_features_test.reset_index(drop=True, inplace=True)


# Łączenie
scorecard_xgb_woe_display = pd.concat([
    scorecard_xgb_woe_indexed[['Score', 'Prawdziwa klasa', 'Decyzja modelu']], # Przenieś Prawdziwa klasa i Decyzja modelu tutaj
    original_features_test_woe,
    woe_features_test
], axis=1)

# Opcjonalnie: Zmień kolejność kolumn, jeśli chcesz 'Score' na początku
cols_order = ['Score'] + features_for_model + woe_feature_names + ['Prawdziwa klasa', 'Decyzja modelu']
scorecard_xgb_woe_display = scorecard_xgb_woe_display[cols_order]


st.dataframe(scorecard_xgb_woe_display, height=400, use_container_width=True, hide_index=True)

# --------------------
# 🎯 Wprowadzenie danych wejściowych przez użytkownika
# --------------------
st.subheader("🔢 Wprowadź dane klienta do predykcji")

user_input = {}
base_features_for_input = []
derived_features = ['spread', 'margin', 'rata_miesieczna', 'intensity_rate']

# Identify base features needed for sliders
for feature in features_for_model:
    if feature not in derived_features:
        base_features_for_input.append(feature)

st.markdown("Ustaw wartości dla podstawowych cech:")

# Create sliders only for base features
for feature in base_features_for_input:
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    mean_val = float(df[feature].mean())
    # Adjust step for better usability, especially for percentages
    if "oproc" in feature or "koszt" in feature:
        step = 0.001 # Smaller step for percentages
    elif feature == 'okres_kredytu':
        step = 1.0 # Integer step for months
    else:
        step = (max_val - min_val) / 100 if max_val > min_val else 1.0 # Default step or 1 if min=max

    user_input[feature] = st.slider(
        label=f"{feature}",
        min_value=min_val,
        max_value=max_val,
        value=mean_val,
        step=step,
        # Format percentages nicely
        format="%.3f" if "oproc" in feature or "koszt" in feature else "%.2f"
    )

# Calculate derived features based on slider inputs
# Ensure necessary base features are present before calculation
calculated_derived = {}
if 'oproc_propon' in user_input and 'oproc_konkur' in user_input:
    calculated_derived['spread'] = user_input['oproc_propon'] - user_input['oproc_konkur']
if 'oproc_propon' in user_input and 'koszt_pieniadza' in user_input:
    calculated_derived['margin'] = user_input['oproc_propon'] - user_input['koszt_pieniadza']
if 'kwota_kredytu' in user_input and 'okres_kredytu' in user_input:
    # Avoid division by zero
    calculated_derived['rata_miesieczna'] = user_input['kwota_kredytu'] / user_input['okres_kredytu'] if user_input['okres_kredytu'] != 0 else 0
    if 'scoring_FICO' in user_input and 'rata_miesieczna' in calculated_derived:
         # Avoid division by zero
        calculated_derived['intensity_rate'] = calculated_derived['rata_miesieczna'] / user_input['scoring_FICO'] if user_input['scoring_FICO'] != 0 else 0

# Display calculated derived features
st.markdown("Obliczone cechy pochodne:")
cols_derived = st.columns(len(derived_features))
i = 0
for feature in derived_features:
    if feature in calculated_derived:
        with cols_derived[i]:
            st.metric(label=feature, value=f"{calculated_derived[feature]:.4f}")
        i += 1

# Combine base inputs and calculated derived features
final_user_input = user_input.copy()
final_user_input.update(calculated_derived)

# Ensure all features required by the model are present, even if not calculated (use default/mean if needed)
for feature in features_for_model:
    if feature not in final_user_input:
        # Provide a default value (e.g., mean) if a derived feature couldn't be calculated
        final_user_input[feature] = df[feature].mean()


# Create the DataFrame with the correct order of columns as expected by the models
user_input_df = pd.DataFrame([final_user_input])[features_for_model]

# Optionally display the final input DataFrame being used
# st.write("Dane wejściowe dla modeli:", user_input_df)

# --------------------
# 🔮 Predykcje z modeli
# --------------------
# WOE model
woe_transformed = encoder.transform(user_input_df[features_for_model])
pred_woe = model.predict_proba(woe_transformed)[0][1]

# XGBoost model (na surowych zmiennych)
pred_xgb = model_xgb.predict_proba(user_input_df[features_for_model])[0][1]

# XGBoost z WOE
woe_cols = encoder.transform(user_input_df[features_for_model])
woe_cols.columns = [f"{col}_woe" for col in woe_cols.columns]
combined_input = pd.concat([user_input_df[features_for_model], woe_cols], axis=1)
pred_xgb_woe = model_xgb_woe.predict_proba(combined_input)[0][1]

# --------------------
# 📊 Wykresy prędkości (gauge)
# --------------------
def draw_gauge(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 25], 'color': "#fab1a0"},
                {'range': [25, 60], 'color': "#ffeaa7"},
                {'range': [60, 100], 'color': "#55efc4"}
            ],
        }))
    fig.update_layout(margin=dict(t=5, b=5))  # Reduce top, bottom, left, and right margins
    return fig

st.subheader("📈 Predykcja scoringowa – wizualizacja")

col1, col2, col3 = st.columns(3)

with col1:
    st.plotly_chart(draw_gauge(pred_woe, "WOE + RL"), use_container_width=True)
with col2:
    st.plotly_chart(draw_gauge(pred_xgb, "XGBoost"), use_container_width=True)
with col3:
    st.plotly_chart(draw_gauge(pred_xgb_woe, "XGBoost + WOE"), use_container_width=True)  

st.markdown("""
Każdy wykres pokazuje przewidywaną szansę akceptacji oferty kredytowej przez klienta w skali 0–100. 
Im wyższy wynik – tym większe prawdopodobieństwo akceptacji według danego modelu.
""")

# --------------------
# 📤 Weryfikacja na danych testowych (.xlsx)
# --------------------
st.subheader("📂 Walidacja na zewnętrznym zbiorze testowym")

uploaded_file = st.file_uploader("Załaduj plik Excel zawierający dane testowe (format taki sam jak oryginalny):", type=["xlsx"])

if uploaded_file is not None:
    test_df = pd.read_excel(uploaded_file)
    test_df_original_for_display = test_df.copy() # Keep a copy for display
    test_df = clean_data(test_df)

    # Inżynieria zmiennych pochodnych
    test_df['spread'] = test_df['oproc_propon'] - test_df['oproc_konkur']
    test_df['margin'] = test_df['oproc_propon'] - test_df['koszt_pieniadza']
    test_df['rata_miesieczna'] = test_df['kwota_kredytu'] / test_df['okres_kredytu']
    test_df['intensity_rate'] = test_df['rata_miesieczna'] / test_df['scoring_FICO']

    # Sprawdź czy dane zawierają wszystkie wymagane kolumny
    missing = [col for col in features_for_model if col not in test_df.columns]
    if missing:
        st.error(f"Brakuje następujących kolumn w zbiorze testowym: {missing}")
    else:
        X_test_final = test_df[features_for_model]
        
        # Ensure 'akceptacja_klienta' exists for AUC calculation, if not, skip AUC
        y_true = None
        if 'akceptacja_klienta' in test_df.columns:
            y_true = test_df['akceptacja_klienta']
        else:
            st.warning("Kolumna 'akceptacja_klienta' nie została znaleziona w pliku. AUC nie zostanie obliczone.")


        # Model 1: WOE + RL
        X_test_woe = encoder.transform(X_test_final)
        y_pred_woe_proba = model.predict_proba(X_test_woe)[:, 1]
        test_df_original_for_display['Score_WOE_RL'] = np.round(y_pred_woe_proba * 100).astype(int)
        auc_woe = roc_auc_score(y_true, y_pred_woe_proba) if y_true is not None else "N/A"


        # Model 2: XGBoost
        y_pred_xgb_proba = model_xgb.predict_proba(X_test_final)[:, 1]
        test_df_original_for_display['Score_XGBoost'] = np.round(y_pred_xgb_proba * 100).astype(int)
        auc_xgb_test = roc_auc_score(y_true, y_pred_xgb_proba) if y_true is not None else "N/A"

        # Model 3: XGBoost + WOE
        X_test_woe_cols = encoder.transform(X_test_final)
        X_test_woe_cols.columns = [f"{col}_woe" for col in X_test_woe_cols.columns]
        X_test_combined = pd.concat([X_test_final.reset_index(drop=True), X_test_woe_cols.reset_index(drop=True)], axis=1)
        y_pred_xgb_woe_proba = model_xgb_woe.predict_proba(X_test_combined)[:, 1]
        test_df_original_for_display['Score_XGBoost_WOE'] = np.round(y_pred_xgb_woe_proba * 100).astype(int)
        auc_xgb_woe_test = roc_auc_score(y_true, y_pred_xgb_woe_proba) if y_true is not None else "N/A"

        st.success("✅ Pomyślnie przetworzono zbiór testowy!")

        auc_woe_display = f"{auc_woe:.4f}" if isinstance(auc_woe, float) else auc_woe
        auc_xgb_test_display = f"{auc_xgb_test:.4f}" if isinstance(auc_xgb_test, float) else auc_xgb_test
        auc_xgb_woe_test_display = f"{auc_xgb_woe_test:.4f}" if isinstance(auc_xgb_woe_test, float) else auc_xgb_woe_test

        st.markdown(f"""
        **Wyniki na zbiorze testowym:**
        - **WOE + RL** – AUC: `{auc_woe_display}`
        - **XGBoost** – AUC: `{auc_xgb_test_display}`
        - **XGBoost + WOE** – AUC: `{auc_xgb_woe_test_display}`
        """)

        st.markdown("#### Wyniki scoringu dla wgranego zbioru:")
        
        # Select columns for display: original columns + new score columns
        # Ensure score columns are at the end or in a specific order if desired
        cols_to_display = list(test_df_original_for_display.columns)
        # Move score columns to the end if they are not already
        score_cols = ['Score_WOE_RL', 'Score_XGBoost', 'Score_XGBoost_WOE']
        for sc in score_cols:
            if sc in cols_to_display:
                cols_to_display.remove(sc)
        cols_to_display.extend(score_cols)
        
        st.dataframe(test_df_original_for_display[cols_to_display], height=400, use_container_width=True, hide_index=True)


st.subheader("💾 Pobierz wytrenowane modele")

# Serializacja modelu WOE + RL
model_woe_rl_bytes = io.BytesIO()
joblib.dump(model, model_woe_rl_bytes)
model_woe_rl_bytes.seek(0)

# Serializacja modelu XGBoost
model_xgb_bytes = io.BytesIO()
joblib.dump(model_xgb, model_xgb_bytes)
model_xgb_bytes.seek(0)

# Serializacja modelu XGBoost + WOE
model_xgb_woe_bytes = io.BytesIO()
joblib.dump(model_xgb_woe, model_xgb_woe_bytes)
model_xgb_woe_bytes.seek(0)

col_dl1, col_dl2, col_dl3 = st.columns(3)

with col_dl1:
    st.download_button(
        label="Pobierz model WOE + RL (.pkl)",
        data=model_woe_rl_bytes,
        file_name="model_woe_rl.pkl",
        mime="application/octet-stream"
    )

with col_dl2:
    st.download_button(
        label="Pobierz model XGBoost (.pkl)",
        data=model_xgb_bytes,
        file_name="model_xgboost.pkl",
        mime="application/octet-stream"
    )

with col_dl3:
    st.download_button(
        label="Pobierz model XGBoost + WOE (.pkl)",
        data=model_xgb_woe_bytes,
        file_name="model_xgboost_woe.pkl",
        mime="application/octet-stream"
    )

    # Serializacja encodera WOE
    encoder_bytes = io.BytesIO()
    joblib.dump(encoder, encoder_bytes)
    encoder_bytes.seek(0)

# Serializacja listy wybranych cech (features_for_model)
features_bytes = io.BytesIO()
joblib.dump(features_for_model, features_bytes)
features_bytes.seek(0)


col_dl4, col_dl5, col_dl6 = st.columns(3)
with col_dl4:
    st.download_button(
        label="Pobierz encoder WOE (.pkl)",
        data=encoder_bytes,
        file_name="woe_encoder.pkl",
        mime="application/octet-stream"
    )
with col_dl5:
    st.download_button(
        label="Pobierz listę cech (.pkl)",
        data=features_bytes,
        file_name="selected_features.pkl",
        mime="application/octet-stream"
    )

with col_dl6:
    # Funkcja inżynierii cech (możesz dodać serializację jeśli chcesz)
    pass  # Możesz tu dodać np. pobieranie feature engineering jeśli chcesz

def scoring_function(df_input):
    import pandas as pd
    import numpy as np
    import joblib

    # Wczytaj modele i encoder
    model_woe_rl = joblib.load("model_woe_rl.pkl")
    model_xgb = joblib.load("model_xgboost.pkl")
    model_xgb_woe = joblib.load("model_xgboost_woe.pkl")
    encoder = joblib.load("woe_encoder.pkl")
    features_for_model = joblib.load("selected_features.pkl")

    # Feature engineering
    df = df_input.copy()
    df['kwota_kredytu'] = df['kwota_kredytu'].replace('[\$,]', '', regex=True).astype(float)
    for col in ['oproc_refin', 'oproc_konkur', 'koszt_pieniadza', 'oproc_propon']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', '', regex=False), errors='coerce')
    df['spread'] = df['oproc_propon'] - df['oproc_konkur']
    df['margin'] = df['oproc_propon'] - df['koszt_pieniadza']
    df['rata_miesieczna'] = df['kwota_kredytu'] / df['okres_kredytu']
    df['intensity_rate'] = df['rata_miesieczna'] / df['scoring_FICO']

    # Wyciągamy cechy
    X = df[features_for_model].copy()

    # Model 1 – WOE + RL
    X_woe = encoder.transform(X)
    df['Score_WOE_RL'] = np.round(model_woe_rl.predict_proba(X_woe)[:, 1] * 100).astype(int)

    # Model 2 – XGBoost
    df['Score_XGBoost'] = np.round(model_xgb.predict_proba(X)[:, 1] * 100).astype(int)

    # Model 3 – XGBoost + WOE
    X_woe.columns = [f"{col}_woe" for col in X_woe.columns]
    X_combined = pd.concat([X.reset_index(drop=True), X_woe.reset_index(drop=True)], axis=1)
    df['Score_XGBoost_WOE'] = np.round(model_xgb_woe.predict_proba(X_combined)[:, 1] * 100).astype(int)

    return df


scoring_pickle = io.BytesIO()
joblib.dump(scoring_function, scoring_pickle)
scoring_pickle.seek(0)

st.download_button(
    label="📥 Pobierz funkcję scorującą (.pkl)",
    data=scoring_pickle,
    file_name="scoring_function.pkl",
    mime="application/octet-stream"
)




st.markdown("""
Powyżej możesz pobrać wszystkie niezbędne pliki `.pkl`, które pozwolą na uruchomienie scoringu poza aplikacją Streamlit – np. w środowisku R z wykorzystaniem pakietu `reticulate` lub w czystym Pythonie.

W paczce znajdziesz:
- wytrenowane modele: `WOE + RL`, `XGBoost`, `XGBoost + WOE`,
- encoder WOE (`woe_encoder.pkl`),
- listę zmiennych użytych w modelu (`selected_features.pkl`),
- funkcję scorującą (`scoring_function.pkl`), która przyjmuje `DataFrame` i zwraca go z trzema kolumnami scoringowymi.

Wystarczy załadować dane w tym samym formacie co oryginalny zbiór i przekazać je do funkcji `scoring_function`.  
Zwrócony obiekt będzie zawierał wyniki scoringu dla każdego z modeli.
""")

import textwrap

@st.cache_data
def generate_rl_function_py():
    # Współczynniki regresji
    coefs = dict(zip(encoder.cols, model.coef_[0]))

    # Zakoduj biny i WOE z binning_tables
    binning_info = {}
    for feature in encoder.cols:
        table = binning_tables[feature]
        bins = []
        for _, row in table.iterrows():
            interval = row["Przedział"]
            woe = row["WOE"]
            bins.append((interval, woe))
        binning_info[feature] = bins

    # Funkcja jako kod Pythona
    lines = []
    lines.append("import pandas as pd")
    lines.append("import numpy as np")
    lines.append("")
    lines.append("def score_woe_rl(df):")
    lines.append("    df = df.copy()")
    lines.append("    # Zmienne pochodne")
    lines.append("    df['spread'] = df['oproc_propon'] - df['oproc_konkur']")
    lines.append("    df['margin'] = df['oproc_propon'] - df['koszt_pieniadza']")
    lines.append("    df['rata_miesieczna'] = df['kwota_kredytu'] / df['okres_kredytu']")
    lines.append("    df['intensity_rate'] = df['rata_miesieczna'] / df['scoring_FICO']")
    lines.append("")

    # Dla każdej zmiennej: generuj binning i scoring
    for feature in encoder.cols:
        lines.append(f"    # {feature} (WOE + coef)")
        lines.append(f"    def map_woe_{feature}(val):")
        for interval, woe in binning_info[feature]:
            interval_str = interval.replace('(', '').replace(']', '')
            low, high = map(float, interval_str.split(','))
            lines.append(f"        if {low} < val <= {high}: return {woe}")
        lines.append(f"        return 0")
        lines.append(f"    df['woe_{feature}'] = df['{feature}'].apply(map_woe_{feature})")
        coef = coefs[feature]
        lines.append(f"    df['score_{feature}'] = df['woe_{feature}'] * {coef}")
        lines.append("")

    # Sumuj score i przelicz na skale 0-100
    sum_expr = ' + '.join([f"df['score_{f}']" for f in encoder.cols])
    lines.append(f"    df['score_logit'] = 1 / (1 + np.exp(-({sum_expr})))")
    lines.append("    df['Score'] = np.round(df['score_logit'] * 100).astype(int)")
    lines.append("    return df")

    return "\n".join(lines)


# Generuj i udostępnij do pobrania
rl_code = generate_rl_function_py()
rl_bytes = rl_code.encode("utf-8")

st.download_button(
    label="📥 Pobierz funkcję scoringową (RL) jako .py",
    data=rl_bytes,
    file_name="score_woe_rl.py",
    mime="text/x-python"
)

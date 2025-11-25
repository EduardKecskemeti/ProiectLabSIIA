import streamlit as st
import pandas as pd

st.set_page_config(page_title="Proiect SIIA - Interfață ML")

# Sidebar menu
st.sidebar.title("Meniu")
pagina = st.sidebar.radio(
    "Navigare",
    ["Set de Date",
     "Logistic Regression",
     "Random Forest",
     "SVM",
     "Clustering (K-Means)"]
)

if pagina == "Set de Date":
    st.title("Set de Date – Heart Disease Prediction ")

    st.subheader("Descrierea setului de date")
    st.write("""
    Acest set de date conține informații clinice ale pacienților, folosite pentru
    a prezice probabilitatea existenței bolii cardiace.

    Variabila țintă este **HeartDisease**:
    - `1` = pacient cu boală cardiacă  
    - `0` = pacient fără boală cardiacă  

    Setul de date include variabile demografice, măsurători clinice și rezultate ECG.
    """)


    try:
        df = pd.read_csv("heart.csv")
        st.subheader("Tabelul complet al setului de date")
        st.dataframe(df)

        # Statistici
        st.subheader("Statistici descriptive")
        st.write(df.describe(include='all'))

        # Descriere coloane
        st.subheader("Descrierea coloanelor")

        descrieri = {
            "Age": "Vârsta pacientului.",
            "Sex": "Sexul biologic (M = masculin, F = feminin).",
            "ChestPainType": "Tipul durerii toracice (ATA, NAP, ASY, TA).",
            "RestingBP": "Tensiunea arterială în repaus (mm Hg).",
            "Cholesterol": "Colesterol seric (mg/dl).",
            "FastingBS": "Glicemie >120 mg/dl (1 = da).",
            "RestingECG": "Rezultatul electrocardiogramei în repaus (Normal, ST, LVH).",
            "MaxHR": "Ritmul cardiac maxim atins.",
            "ExerciseAngina": "Angină indusă de efort (Y/N).",
            "Oldpeak": "Depresia segmentului ST cauzată de efort.",
            "ST_Slope": "Panta segmentului ST la efort (Up, Flat, Down).",
            "HeartDisease": "Prezența bolii cardiace (0 = nu, 1 = da)."
        }

        for col, desc in descrieri.items():
            st.write(f"**{col}** – {desc}")

        # ---- Variabile independente/dependente ----
        st.subheader("Variabile independente și variabila dependentă")

        st.write("**Variabilă dependentă (target):**")
        st.code("HeartDisease")

        st.write("**Variabile independente:**")
        independent_vars = [col for col in df.columns if col != "HeartDisease"]
        st.code(", ".join(independent_vars))

    except FileNotFoundError:
        st.error("⚠ Fișierul `heart.csv` nu a fost găsit în folderul proiectului.")

elif pagina == "Logistic Regression":
    st.title("Model: Logistic Regression")
    st.write("Aici va fi implementat modelul Logistic Regression.")

elif pagina == "Random Forest":
    st.title("Model: Random Forest")
    st.write("Aici va fi implementat modelul Random Forest.")

elif pagina == "SVM":
    st.title("Model: SVM")
    st.write("Aici va fi implementat modelul SVM.")

elif pagina == "Clustering (K-Means)":
    st.title("Clustering – K-Means")
    st.write("Aici va fi implementat clustering-ul K-Means.")

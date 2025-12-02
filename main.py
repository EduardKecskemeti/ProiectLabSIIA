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
     "Clustering (K-Means)",
     "Comparare Modele"]
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

        st.subheader("Statistici descriptive")
        st.write(df.describe(include='all'))

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
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

    st.title("Model: Logistic Regression")
    st.write("Această pagină antrenează un model Logistic Regression pentru predicția bolilor cardiace.")

    try:
        df = pd.read_csv("heart.csv")

        st.subheader("Date brute")
        st.dataframe(df.head())


        st.subheader("Preprocesarea datelor")

        categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

        numeric_cols = [col for col in df.columns if col not in categorical_cols and col != "HeartDisease"]

        st.write("**Coloane numerice:**", numeric_cols)
        st.write("**Coloane categorice:**", categorical_cols)

        X = df.drop("HeartDisease", axis=1)
        y = df["HeartDisease"]

        ct = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(drop="first"), categorical_cols)
            ]
        )

        X_processed = ct.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )

        st.subheader("Antrenarea modelului Logistic Regression")

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        st.subheader("Metrici de performanță")

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        st.write(f"**Acuratețe:** {acc:.4f}")
        st.write(f"**Precizie:** {prec:.4f}")
        st.write(f"**Recall:** {rec:.4f}")
        st.write(f"**F1-score:** {f1:.4f}")

        st.subheader("Matricea de confuzie")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.subheader("ROC Curve")

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        ax2.plot([0, 1], [0, 1], linestyle="--")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve")
        ax2.legend()
        st.pyplot(fig2)

        st.subheader("Descărcare rezultate predicție")

        results = pd.DataFrame({
            "True_Label": y_test.values,
            "Predicted_Label": y_pred,
            "Predicted_Probability": y_pred_proba
        })

        csv = results.to_csv(index=False).encode()
        st.download_button(
            "Descarcă predicțiile ca CSV",
            data=csv,
            file_name="logistic_regression_predictions.csv",
            mime="text/csv"
        )

    except FileNotFoundError:
        st.error("⚠ Fișierul `heart.csv` nu a fost găsit în directorul aplicației.")


elif pagina == "Random Forest":
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, roc_curve, auc
    )

    st.title("Model: Random Forest")
    st.write("Această pagină antrenează un model Random Forest pentru predicția bolilor cardiace.")

    try:
        df = pd.read_csv("heart.csv")

        st.subheader("Date brute")
        st.dataframe(df.head())


        st.subheader("Preprocesarea datelor")

        categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

        numeric_cols = [col for col in df.columns if col not in categorical_cols and col != "HeartDisease"]

        st.write("**Coloane numerice:**", numeric_cols)
        st.write("**Coloane categorice:**", categorical_cols)

        X = df.drop("HeartDisease", axis=1)
        y = df["HeartDisease"]

        ct = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(drop="first"), categorical_cols)
            ]
        )

        X_processed = ct.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )

        # MODEL RANDOM FOREST

        st.subheader("Antrenarea modelului Random Forest")

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predicții
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        st.subheader("Metrici de performanță")

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        st.write(f"**Acuratețe:** {acc:.4f}")
        st.write(f"**Precizie:** {prec:.4f}")
        st.write(f"**Recall:** {rec:.4f}")
        st.write(f"**F1-score:** {f1:.4f}")

        st.subheader("Matricea de confuzie")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.subheader("Importanța caracteristicilor (Feature Importance)")

        # Numele coloanelor după OneHotEncoding
        ohe = ct.named_transformers_["cat"]
        encoded_cols = ohe.get_feature_names_out(categorical_cols)

        feature_names = numeric_cols + list(encoded_cols)
        importances = model.feature_importances_

        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(fi_df)

        # Grafic feature importance
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=fi_df, ax=ax3, palette="viridis")
        ax3.set_title("Importanța Caracteristicilor")
        st.pyplot(fig3)

        st.subheader("ROC Curve")

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        ax2.plot([0, 1], [0, 1], linestyle="--")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve")
        ax2.legend()
        st.pyplot(fig2)

        st.subheader("Descărcare rezultate predicție")

        results = pd.DataFrame({
            "True_Label": y_test.values,
            "Predicted_Label": y_pred,
            "Predicted_Probability": y_pred_proba
        })

        csv = results.to_csv(index=False).encode()
        st.download_button(
            "Descarcă predicțiile (CSV)",
            data=csv,
            file_name="random_forest_predictions.csv",
            mime="text/csv"
        )

    except FileNotFoundError:
        st.error("⚠ Fișierul heart.csv nu este în folderul proiectului.")


elif pagina == "SVM":
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.svm import SVC
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_curve, auc
    )

    st.title("Model: SVM (Support Vector Machine)")
    st.write("Această pagină antrenează un model SVM pentru predicția bolilor cardiace.")

    try:
        df = pd.read_csv("heart.csv")

        st.subheader("Date brute")
        st.dataframe(df.head())

        st.subheader("Preprocesarea datelor")

        categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
        numeric_cols = [col for col in df.columns if col not in categorical_cols and col != "HeartDisease"]

        st.write("**Coloane numerice:**", numeric_cols)
        st.write("**Coloane categorice:**", categorical_cols)

        X = df.drop("HeartDisease", axis=1)
        y = df["HeartDisease"]

        # OneHot + Scaling
        ct = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(drop="first"), categorical_cols)
            ]
        )

        X_processed = ct.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )

        # MODEL SVM

        st.subheader("Antrenarea modelului SVM")

        model = SVC(kernel="rbf", probability=True, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        st.subheader("Metrici de performanță")

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        st.write(f"**Acuratețe:** {acc:.4f}")
        st.write(f"**Precizie:** {prec:.4f}")
        st.write(f"**Recall:** {rec:.4f}")
        st.write(f"**F1-score:** {f1:.4f}")

        st.subheader("Matrice de confuzie")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.subheader("ROC Curve")

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        ax2.plot([0, 1], [0, 1], linestyle="--")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve")
        ax2.legend()
        st.pyplot(fig2)

        st.subheader("Descărcare rezultate predicție")

        results = pd.DataFrame({
            "True_Label": y_test.values,
            "Predicted_Label": y_pred,
            "Predicted_Probability": y_pred_proba
        })

        csv = results.to_csv(index=False).encode()
        st.download_button(
            "Descarcă predicțiile (CSV)",
            data=csv,
            file_name="svm_predictions.csv",
            mime="text/csv"
        )

    except FileNotFoundError:
        st.error("⚠ Fișierul `heart.csv` nu a fost găsit în folderul proiectului.")


elif pagina == "Clustering (K-Means)":
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    st.title("Clustering – K-Means")
    st.write("Această pagină aplică algoritmul K-Means pentru a grupa pacienții în funcție de caracteristici similare.")

    try:
        df = pd.read_csv("heart.csv")

        st.subheader("Date brute")
        st.dataframe(df.head())

        st.subheader("Preprocesarea datelor")

        categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
        numeric_cols = [col for col in df.columns if col not in categorical_cols and col != "HeartDisease"]

        st.write("**Coloane numerice:**", numeric_cols)
        st.write("**Coloane categorice:**", categorical_cols)

        # Folosim doar X (fără y)
        X = df.drop("HeartDisease", axis=1)

        ct = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(drop="first"), categorical_cols)
            ]
        )

        X_processed = ct.fit_transform(X)

        st.subheader("Reducerea dimensionalității cu PCA (2 componente)")

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_processed)

        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])

        # K-MEANS

        st.subheader("Aplicarea algoritmului K-Means")

        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(X_processed)

        pca_df["Cluster"] = clusters

        st.write("Primele linii cu cluster assignment:")
        st.dataframe(pca_df.head())

        st.subheader("Silhouette Score")
        silhouette = silhouette_score(X_processed, clusters)
        st.write(f"**Silhouette Score:** {silhouette:.4f}")

        st.subheader("Vizualizarea clusterelor în 2D (PCA)")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=pca_df["PC1"],
            y=pca_df["PC2"],
            hue=pca_df["Cluster"],
            palette="tab10",
            ax=ax
        )
        ax.set_title("Clustere K-Means (vizualizate cu PCA)")
        st.pyplot(fig)

        st.subheader("Descărcare rezultate clusterizare")

        results = pd.DataFrame(X)
        results["Cluster"] = clusters

        csv = results.to_csv(index=False).encode()
        st.download_button(
            "Descarcă clusterele (CSV)",
            data=csv,
            file_name="kmeans_clusters.csv",
            mime="text/csv"
        )

    except FileNotFoundError:
        st.error("⚠ Fișierul `heart.csv` nu a fost găsit în folderul proiectului.")
elif pagina == "Comparare Modele":
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import time

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    st.title("Comparare Modele de Clasificare")
    st.write("Această pagină compară Logistic Regression, Random Forest și SVM pe același set de date și aceleași condiții.")

    try:
        df = pd.read_csv("heart.csv")

        # ----------------------------
        # PREPROCESARE
        # ----------------------------
        categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
        numeric_cols = [col for col in df.columns if col not in categorical_cols and col != "HeartDisease"]

        X = df.drop("HeartDisease", axis=1)
        y = df["HeartDisease"]

        ct = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(drop="first"), categorical_cols)
            ]
        )

        X_processed = ct.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )

        # ----------------------------
        # FUNCȚIE UTILITARĂ
        # ----------------------------
        def eval_model(model):
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time

            y_pred = model.predict(X_test)

            return {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1-Score": f1_score(y_test, y_pred),
                "Train Time (s)": train_time
            }

        st.subheader("Evaluarea modelelor…")

        results = {}

        results["Logistic Regression"] = eval_model(
            LogisticRegression(max_iter=1000)
        )

        results["Random Forest"] = eval_model(
            RandomForestClassifier(n_estimators=200, random_state=42)
        )

        results["SVM (RBF)"] = eval_model(
            SVC(kernel="rbf", probability=True, random_state=42)
        )

        st.subheader("Tabel comparativ al performanțelor")

        results_df = pd.DataFrame(results).T
        st.dataframe(results_df)

        st.subheader("Comparație vizuală a metricilor")

        fig, ax = plt.subplots(figsize=(10, 6))
        results_df[["Accuracy", "Precision", "Recall", "F1-Score"]].plot(kind='bar', ax=ax)
        ax.set_title("Compararea modelelor")
        ax.set_ylabel("Scor")
        ax.set_ylim(0, 1)
        st.pyplot(fig)

        st.subheader("Cel mai bun model (după F1-Score)")

        best_model = results_df["F1-Score"].idxmax()
        best_score = results_df["F1-Score"].max()

        st.success(f"**Cel mai bun model este: {best_model} (F1 = {best_score:.4f})**")

    except FileNotFoundError:
        st.error("⚠ Fișierul `heart.csv` nu a fost găsit în folderul proiectului.")



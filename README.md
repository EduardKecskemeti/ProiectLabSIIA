#  Proiect – Sisteme Inteligente de Învățare Automată

Acest proiect reprezintă o interfață interactivă de Machine Learning realizată în **Python + Streamlit**, care permite explorarea setului de date *Heart Disease Prediction* și antrenarea mai multor modele de clasificare:

- Logistic Regression  
- Random Forest  
- Support Vector Machine (SVM)  
- Clustering (K-Means)  

Proiectul include vizualizări, metrici de performanță, preprocesare automată și comparația finală între modele.


## 📊 Set de date folosit

Dataset: **Heart Disease Prediction**  
Sursă: Kaggle  
Format: `heart.csv`  

Variabila dependentă:
- **HeartDisease** (0 = fără boală, 1 = boală cardiacă)

Variabile independente (feature-uri):
- Age  
- Sex  
- ChestPainType  
- RestingBP  
- Cholesterol  
- FastingBS  
- RestingECG  
- MaxHR  
- ExerciseAngina  
- Oldpeak  
- ST_Slope  

Setul de date este încărcat automat în aplicație și afișat în pagina „Set de Date”.

---

## 🚀 Funcționalități principale

###  1. Pagina Set de Date
- afișare dataset complet  
- statistici descriptive  
- descrierea fiecărei coloane  
- separarea variabilelor: dependente / independente  

---

###  2. Logistic Regression
- One-Hot Encoding + StandardScaler  
- metrici: accuracy, precision, recall, F1  
- matrice de confuzie (heatmap)  
- ROC Curve (AUC)  
- export CSV cu predicții  

---

### 3. Random Forest
- antrenare model RF  
- metrici complete  
- matrice de confuzie  
- ROC Curve  
- **Feature Importance** (tabel + bar chart)  
- export CSV  

---

###  4. SVM (Support Vector Machine)
- kernel RBF  
- metrici complete  
- matrice de confuzie  
- ROC Curve  
- export CSV  

---

###  5. Clustering (K-Means)
- preprocesare completă  
- reducere dimensionalitate cu PCA  
- clustering cu K=2  
- silhouette score  
- scatter plot în 2D  
- export CSV cu clustere  

---

###  6. Comparare Modele
Pagină care:
- reantrenează Logistic Regression, Random Forest și SVM  
- compară metricile:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-Score  
  - Timp de antrenare  
- afișează grafic comparativ  
- selectează automat modelul optim (după F1-score)

---

##  Tehnologii utilizate

- **Python 3.10+**  
- **Streamlit** – interfață web  
- **Pandas**, **NumPy** – manipulare date  
- **scikit-learn** – modele ML și preprocesare  
- **Matplotlib**, **Seaborn** – vizualizări  
- **PCA** (dimensionalitate redusă)  
- **KMeans** clustering  

---

##  Rulare aplicație

1. Instalează Streamlit:
```bash
pip install streamlit

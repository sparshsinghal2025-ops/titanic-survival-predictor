# 🚢 Titanic Passenger Survival Predictor

An end-to-end Machine Learning web application that predicts passenger survival probabilities on the Titanic using historical passenger manifest features. 

🔗 **Live Production App:** [Launch Web Application](https://streamlit.app)

---

## 📊 Project Architecture & Pipeline

1. **Exploratory Data Analysis**: Handled heavily shifted cells, malformed string categorical column attributes, and structural spreadsheet corruption issues.
2. **Feature Engineering & Imputation**: Automated missing value resolution algorithms utilizing median arrays for continuous numeric features (`Age`, `Fare`) and mode frequencies for categorical structures (`Sex`, `Embarked`).
3. **Data Transformation**: Engineered isolated numeric encoding via a custom `ColumnTransformer` scaling numeric features via `StandardScaler` and one-hot encoding categories with `OneHotEncoder`.
4. **Model Evaluation & Tuning**: Evaluated multiple pipelines across `Logistic Regression`, `Random Forest Classifiers`, and `XGBoost` frameworks optimized using `RandomizedSearchCV`.
5. **Serialization**: Bundled the final winning production pipeline configurations cleanly into a persistent object file (`TitanicSurvival.pkl`).
6. **Deployment**: Built a modern reactive UI dashboard engine using `Streamlit` deployed directly on cloud-native containerized architecture infrastructure.

---

## 🛠️ Technology Stack Used

* **Language**: Python
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn, XGBoost
* **Deployment & UI**: Streamlit Framework, Pickle Serialization
* **Version Control**: Git / GitHub Production Workflow

---

## 💻 Local Installation & Setup

If you want to run this application workflow on your local machine, execute these steps inside your terminal:

```bash
# 1. Clone this open-source repository
git clone https://github.com

# 2. Change directories into the project root path folder
cd titanic-survival-predictor

# 3. Install all mandatory ecosystem package dependencies
pip install -r requirements.txt

# 4. Spin up the reactive local interface web development server
streamlit run app_1.py
```

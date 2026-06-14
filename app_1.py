import pandas as pd
import pickle
import streamlit as st

# We parse string numbers here to prevent the markdown rendering block from hiding values
pclass_options = [int(x) for x in "1,2,3".split(",")]

# 1. Page Configuration Settings
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
st.title("🚢 Titanic Passenger Survival Predictor")
st.write("Enter details below to calculate the real-time survival probability.")


# 2. Safely Load Saved Modeling Pipeline Elements
@st.cache_resource
def load_pipeline():
    with open("TitanicSurvival.pkl", "rb") as f:
        return pickle.load(f)


artifacts = load_pipeline()
model = artifacts["model"]
preprocessor = artifacts["preprocessor"]
scaler = artifacts["scaler"]

# 3. Interactive Web Layout UI Form Elements
col1, col2 = st.columns(2)

with col1:
    # Using our safely generated option list
    pclass = st.selectbox("Ticket Class (Pclass)", pclass_options, index=2)
    sex = st.selectbox("Gender (Sex)", ["male", "female"])
    age = st.slider("Passenger Age", 0.0, 100.0, 25.0, 0.5)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

with col2:
    sibsp = st.number_input(
        "Count of Siblings/Spouses Aboard (SibSp)",
        min_value=0,
        max_value=10,
        value=0,
    )
    parch = st.number_input(
        "Count of Parents/Children Aboard (Parch)",
        min_value=0,
        max_value=10,
        value=0,
    )
    fare = st.number_input(
        "Ticket Fare Paid ($)", min_value=0.0, max_value=600.0, value=15.0
    )

# 4. Trigger Analysis on Button Activation
if st.button("Calculate Survival Odds", type="primary"):
    # Group inputs into expected structure
    input_data = pd.DataFrame(
        [
            {
                "Pclass": pclass,
                "Sex": sex,
                "Age": age,
                "SibSp": sibsp,
                "Parch": parch,
                "Fare": fare,
                "Embarked": embarked,
            }
        ]
    )

    # Apply saved feature scaling transformers
    processed_data = preprocessor.transform(input_data)
    scaled_data = scaler.transform(processed_data)

    # Extract target calculations
    prediction = model.predict(scaled_data)
    probabilities = model.predict_proba(scaled_data)[0]
    prob_died = probabilities[0]
    prob_survived = probabilities[1]


    st.markdown("---")
    # Display responses based on outputs
    if prediction == 1:
        st.success("### 🟢 Result: SURVIVED")
        st.metric(
            label="Survival Confidence Score", value=f"{prob_survived * 100:.2f}%"
        )
    else:
        st.error("### 🔴 Result: DIED")
        st.metric(label="Fatality Likelihood", value=f"{prob_died * 100:.2f}%")

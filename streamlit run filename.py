import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()
API_KEY = os.getenv("SPORTS_API_KEY")

st.set_page_config(page_title="Athlete Health & Performance Analysis", layout="wide")
st.title("🏋️ Athlete Health and Performance Dashboard")
st.markdown("This system helps analyze athlete fitness and reduce injury risk using KNN.")

# Load dataset
st.subheader("📊 Dataset Upload")
try:
    df = pd.read_csv("fitness_data.csv")
    st.success("Dataset loaded successfully!")
except:
    st.error("Failed to load dataset, using sample data.")
    data = {
        'heart_rate': [60, 72, 90, 110, 55, 75, 85, 95],
        'blood_pressure': [110, 120, 130, 150, 100, 118, 125, 140],
        'oxygen_level': [98, 97, 95, 92, 99, 96, 94, 91],
        'fatigue_level': [1, 2, 3, 4, 1, 2, 3, 4],
        'fit': [1, 1, 1, 0, 1, 1, 0, 0]
    }
    df = pd.DataFrame(data)

# Visualizations
st.subheader("🔍 Data Visualization")
col1, col2 = st.columns(2)

with col1:
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with col2:
    st.write("### Pair Plot")
    fig2 = sns.pairplot(df, hue="fit")
    st.pyplot(fig2)

# Split and Train Model
X = df.drop('fit', axis=1)
y = df['fit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Input Form
st.subheader("🧬 Enter Athlete Health Data")
with st.form("health_form"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        heart_rate = st.slider("Heart Rate", 50, 150, 70)
    with col2:
        blood_pressure = st.slider("Blood Pressure", 90, 160, 120)
    with col3:
        oxygen_level = st.slider("Oxygen Level", 85, 100, 95)
    with col4:
        fatigue_level = st.slider("Fatigue Level", 1, 5, 2)
    submitted = st.form_submit_button("Analyze")

    if submitted:
        new_data = pd.DataFrame([[heart_rate, blood_pressure, oxygen_level, fatigue_level]],
                                 columns=['heart_rate', 'blood_pressure', 'oxygen_level', 'fatigue_level'])
        prediction = knn.predict(new_data)[0]
        result = "✅ Fit to train" if prediction == 1 else "❌ Not fit for intense training"
        st.success(f"Athlete Health Status: {result}")

# Model Evaluation
st.subheader("📈 Model Evaluation")
st.metric("Accuracy", f"{accuracy * 100:.2f}%")
st.text("Classification Report:\n" + report)

# Live API Data Integration (SportsData.io - MMA)
st.subheader("🥊 Live MMA Fight Data")
API_URL = "https://api.sportsdata.io/v4/mma/scores/json/FightsByDate/2024-08-01"
headers = {
    'Ocp-Apim-Subscription-Key': API_KEY
}
response = requests.get(API_URL, headers=headers)
if response.status_code == 200:
    fights_data = response.json()
    if fights_data:
        df_fights = pd.DataFrame(fights_data)
        if not df_fights.empty:
            st.dataframe(df_fights[['FightID', 'Fighter1Name', 'Fighter2Name', 'WeightClass', 'DateTime', 'Result']])
        else:
            st.info("No fights available for this date.")
    else:
        st.info("No fight data returned from API.")
else:
    st.error(f"Failed to fetch MMA data: {response.status_code}")

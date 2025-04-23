import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Connect to SQLite database (creates one if not exists)
conn = sqlite3.connect('athletes.db')
cursor = conn.cursor()

# Create a table to store athlete data if not exists
cursor.execute('''
CREATE TABLE IF NOT EXISTS athletes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    heart_rate INTEGER,
    oxygen_level INTEGER,
    injury_score INTEGER,
    stress_level INTEGER,
    age INTEGER,
    bmi REAL,
    prediction TEXT
)
''')
conn.commit()

# Load and preprocess the dataset
data = pd.read_csv('athlete_health_data.csv')
data.dropna(inplace=True)
X = data.drop(['athlete_id', 'can_continue'], axis=1)
y = data['can_continue']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

def predict_health_status(input_data):
    prediction = knn.predict(input_data)
    return "Fit to continue training" if prediction[0] == 1 else "Recommended to stop training"

# Streamlit UI
st.title("Athlete Health Monitoring System")

st.sidebar.header("Enter Athlete Data")
heart_rate = st.sidebar.slider("Heart Rate", 40, 200, 72)
oxygen_level = st.sidebar.slider("Oxygen Level (%)", 80, 100, 98)
injury_score = st.sidebar.slider("Injury Score", 0, 10, 1)
stress_level = st.sidebar.slider("Stress Level", 0, 10, 3)
age = st.sidebar.slider("Age", 10, 60, 21)
bmi = st.sidebar.slider("BMI", 10.0, 40.0, 22.5)

if st.sidebar.button("Predict"):
    new_data = pd.DataFrame([{
        'heart_rate': heart_rate,
        'oxygen_level': oxygen_level,
        'injury_score': injury_score,
        'stress_level': stress_level,
        'age': age,
        'bmi': bmi
    }])
    result = predict_health_status(new_data)
    st.write("### Prediction Result:", result)

    # Insert into database
    cursor.execute('''INSERT INTO athletes (heart_rate, oxygen_level, injury_score, stress_level, age, bmi, prediction)
                      VALUES (?, ?, ?, ?, ?, ?, ?)''',
                   (heart_rate, oxygen_level, injury_score, stress_level, age, bmi, result))
    conn.commit()
    st.success("Athlete data saved to database.")

# Visualization Section
st.subheader("Data Visualizations")
if st.checkbox("Show Heatmap"):
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(data.drop('athlete_id', axis=1).corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

if st.checkbox("Show Class Distribution"):
    st.write("### Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='can_continue', data=data, ax=ax)
    st.pyplot(fig)

# Evaluation Metrics
st.subheader("Model Evaluation")
y_pred = knn.predict(X_test)
st.text("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Database Viewer
if st.checkbox("Show Saved Athlete Records"):
    st.subheader("Stored Athlete Predictions")
    athlete_df = pd.read_sql_query("SELECT * FROM athletes", conn)
    st.dataframe(athlete_df)

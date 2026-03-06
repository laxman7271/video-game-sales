import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error

st.title("🎮 Video Game Global Sales Prediction")
df = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv")

st.subheader("Dataset Preview")
st.write(df.head())

df = df.dropna()


le = LabelEncoder()

df["Platform"] = le.fit_transform(df["Platform"])
df["Genre"] = le.fit_transform(df["Genre"])
df["Publisher"] = le.fit_transform(df["Publisher"])

# Feature Selection

features = [
    "Platform",
    "Year_of_Release",
    "Genre",
    "Publisher",
    "Critic_Score",
    "User_Score"
]

X = df[features]
y = df["Global_Sales"]
# Train Model

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Model Performance

st.subheader("Model Performance")

st.write("R2 Score:", r2_score(y_test, y_pred))
st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Top 10 Games Chart

st.subheader("Top 10 Games by Global Sales")

top_games = df.sort_values("Global_Sales", ascending=False).head(10)

fig1, ax1 = plt.subplots()

ax1.barh(top_games["Name"], top_games["Global_Sales"])

ax1.set_xlabel("Global Sales")

st.pyplot(fig1)

# Sales by Platform

st.subheader("Sales by Platform")

platform_sales = df.groupby("Platform")["Global_Sales"].sum()

fig2, ax2 = plt.subplots()

platform_sales.plot(kind="bar", ax=ax2)

st.pyplot(fig2)


# Sales by Genre


st.subheader("Sales by Genre")

genre_sales = df.groupby("Genre")["Global_Sales"].sum()

fig3, ax3 = plt.subplots()

genre_sales.plot(kind="bar", ax=ax3)

st.pyplot(fig3)

# Correlation Heatmap

st.subheader("Correlation Heatmap")

fig4, ax4 = plt.subplots()

corr = df.select_dtypes(include=np.number).corr()

sns.heatmap(corr, cmap="coolwarm", ax=ax4)

st.pyplot(fig4)

# Actual vs Predicted


st.subheader("Actual vs Predicted Sales")

fig5, ax5 = plt.subplots()

ax5.scatter(y_test, y_pred)

ax5.set_xlabel("Actual Sales")

ax5.set_ylabel("Predicted Sales")

st.pyplot(fig5)

# Prediction Section

st.subheader("Predict Game Sales")

critic = st.number_input("Critic Score", 0, 100)

user = st.number_input("User Score", 0.0, 10.0)

year = st.number_input("Year of Release", 1980, 2025)

if st.button("Predict Sales"):

    input_data = np.zeros((1, len(features)))

    input_data[0][1] = year
    input_data[0][4] = critic
    input_data[0][5] = user

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Global Sales: {prediction:.2f} Million Copies")
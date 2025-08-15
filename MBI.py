import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import requests

st.set_page_config(page_title="Streamlit ML Demo", layout="centered")

st.title("MBI project for Cement Industry")

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in iris.target]

# Train model
model = RandomForestClassifier()
model.fit(df[iris.feature_names], iris.target)

# Sidebar inputs
st.sidebar.header("Input Features")
sl = st.sidebar.slider("Sepal length", float(df["sepal length (cm)"].min()), float(df["sepal length (cm)"].max()))
sw = st.sidebar.slider("Sepal width", float(df["sepal width (cm)"].min()), float(df["sepal width (cm)"].max()))
pl = st.sidebar.slider("Petal length", float(df["petal length (cm)"].min()), float(df["petal length (cm)"].max()))
pw = st.sidebar.slider("Petal width", float(df["petal width (cm)"].min()), float(df["petal width (cm)"].max()))

# Predict
pred = model.predict([[sl, sw, pl, pw]])[0]
st.subheader("Prediction")
st.write(f"Species: {iris.target_names[pred].title()}")

# Static plot with matplotlib & seaborn
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="sepal length (cm)", y="sepal width (cm)", hue="species", ax=ax)
st.pyplot(fig)

# Interactive plot with plotly
fig2 = px.scatter(df, x="petal length (cm)", y="petal width (cm)", color="species", title="Petal Dimensions")
st.plotly_chart(fig2)

# Show an image with Pillow
image = Image.new("RGB", (100, 50), color="lightblue")
st.image(image, caption="Example image with Pillow")

# Make an API request
r = requests.get("https://api.github.com")
st.write("GitHub API status code:", r.status_code)

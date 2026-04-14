import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris

# Page config
st.set_page_config(page_title="Iris Dashboard", layout="wide")

st.title("🌸 Iris Interactive Dashboard")

# Load data
iris = load_iris(as_frame=True)
df = iris.frame
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

# Sidebar
st.sidebar.header("Filters")

species = st.sidebar.multiselect(
    "Select Species",
    df["species"].unique(),
    default=df["species"].unique()
)

x_axis = st.sidebar.selectbox("X-axis", iris.feature_names)
y_axis = st.sidebar.selectbox("Y-axis", iris.feature_names)

# Filter data
filtered_df = df[df["species"].isin(species)]

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Rows", len(filtered_df))
col2.metric("Species", filtered_df["species"].nunique())
col3.metric("Avg Sepal Length", round(filtered_df["sepal length (cm)"].mean(), 2))

# Scatter plot
fig = px.scatter(
    filtered_df,
    x=x_axis,
    y=y_axis,
    color="species",
    title="Iris Scatter Plot"
)

st.plotly_chart(fig, use_container_width=True)

# Histogram
feature = st.selectbox("Feature Distribution", iris.feature_names)

hist = px.histogram(
    filtered_df,
    x=feature,
    color="species",
    barmode="overlay"
)

st.plotly_chart(hist, use_container_width=True)

# Data table
st.dataframe(filtered_df)
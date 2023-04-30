
import streamlit as st
import pandas as pd
from transformers import pipeline


model_name = "sshleifer/distilbart-cnn-12-6"
model_revision = "a4f8f3e"

@st.cache
def load_model():
    return pipeline("summarization", model=model_name, revision=model_revision)
summarizer = load_model()

# Define a function to generate summaries
def generate_summary(text):
    # Generate a summary of the text using the summarization model
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
    return summary

# Define the Streamlit app

# Set the title of the app
st.title("Medicine Summary Generator")

# Add a file uploader to allow the user to upload a CSV file
st.sidebar.title("Upload a CSV file")
file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# If a file was uploaded, display a preview of the data
if file is not None:
    df = pd.read_csv(file)
    st.write("Data preview:")
    st.write(df.head())

# Generate a summary for each medicine in the CSV file
    st.write("Summary:")
    for index, row in df.iterrows():
        summary = generate_summary(row["overview"])
        st.write(f"{row['drug names']}: {summary}")

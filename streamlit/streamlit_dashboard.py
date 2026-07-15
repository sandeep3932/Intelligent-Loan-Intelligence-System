import streamlit as st
import pandas as pd
import requests

API_URL = "http://localhost:8001"

st.set_page_config(page_title="Intelligent Loan Intelligence Platform", layout="wide")

st.title("Intelligent Loan Intelligence Platform")
st.markdown("An AI-powered NLP system for understanding customer applications, feedback, and queries.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Loan Analytics",
    "😃 Sentiment Dashboard",
    "🔍 Semantic Search",
    "🔮 BERT Prediction",
    "💬 QA Interface"
])

with tab1:
    st.header("Dataset Overview")
    try:
        df = pd.read_csv("../data/processed_dataset.csv")
        st.dataframe(df.head(10))
        st.subheader("Text Statistics")
        st.write(f"Total Applications: {len(df)}")
        st.write(f"Average Application Text Length: {df['Application_Text'].str.len().mean():.2f} characters")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

with tab2:
    st.header("Sentiment Dashboard")
    try:
        if 'df' in locals():
            st.subheader("Sentiment Distribution")
            sentiment_counts = df['Customer_Sentiment'].value_counts()
            st.bar_chart(sentiment_counts)
    except Exception as e:
        pass
        
    st.subheader("Sentiment Prediction Interface")
    feedback_input = st.text_area("Enter customer feedback:", "I am very happy with the service.")
    if st.button("Predict Sentiment"):
        with st.spinner("Analyzing..."):
            try:
                res = requests.post(f"{API_URL}/predict-sentiment", json={"text": feedback_input})
                if res.status_code == 200:
                    sentiment = res.json()["sentiment"]
                    if sentiment.lower() == "positive":
                        st.success(f"Predicted Sentiment: **{sentiment}**")
                    elif sentiment.lower() == "negative":
                        st.error(f"Predicted Sentiment: **{sentiment}**")
                    else:
                        st.info(f"Predicted Sentiment: **{sentiment}**")
                else:
                    st.error("Error connecting to API")
            except requests.exceptions.RequestException:
                st.error("API Server is not running. Please start the FastAPI server.")

with tab3:
    st.header("Semantic Search")
    query = st.text_input("Enter search query:")
    if st.button("Search"):
        with st.spinner("Searching..."):
            try:
                res = requests.post(f"{API_URL}/semantic-search", json={"query": query})
                if res.status_code == 200:
                    results = res.json()["results"]
                    for idx, result in enumerate(results):
                        st.markdown(f"**Match {idx+1} (Score: {result['score']})**")
                        st.write(result['text'])
                else:
                    st.error("Error connecting to API")
            except requests.exceptions.RequestException:
                st.error("API Server is not running. Please start the FastAPI server.")

with tab4:
    st.header("BERT Loan Status Prediction")
    app_input = st.text_area("Enter Loan Application text:")
    if st.button("Classify Loan"):
        with st.spinner("Analyzing..."):
            try:
                res = requests.post(f"{API_URL}/classify-loan", json={"text": app_input})
                if res.status_code == 200:
                    data = res.json()
                    st.write(f"**Predicted Loan Status:** {data['loan_status']}")
                    st.write(f"**Confidence Score:** {data['confidence']:.4f}")
                else:
                    st.error("Error connecting to API")
            except requests.exceptions.RequestException:
                st.error("API Server is not running. Please start the FastAPI server.")

with tab5:
    st.header("QA Interface")
    question = st.text_input("Ask a banking question:")
    if st.button("Ask"):
        with st.spinner("Generating answer..."):
            try:
                res = requests.post(f"{API_URL}/question-answer", json={"text": question})
                if res.status_code == 200:
                    st.success(f"**Answer:** {res.json()['answer']}")
                else:
                    st.error("Error connecting to API")
            except requests.exceptions.RequestException:
                st.error("API Server is not running. Please start the FastAPI server.")


import streamlit as st
import requests

API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Intelligent Loan Platform", page_icon="🏦", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Sentiment Dashboard", "Named Entity Recognition", "Semantic Search"])

if page == "Sentiment Dashboard":
    st.title("Sentiment Dashboard")
    st.markdown("Analyze the sentiment of customer feedback or agent notes.")
    
    with st.form("sentiment_form"):
        text_input = st.text_area("Enter Customer Feedback:")
        submit = st.form_submit_button("Analyze Sentiment")
        
    if submit and text_input:
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(f"{API_BASE_URL}/predict-sentiment", json={"text": text_input})
                if response.status_code == 200:
                    data = response.json()
                    sentiment = data.get("sentiment")
                    confidence = data.get("confidence")
                    
                    st.subheader("Result")
                    if sentiment == "Positive":
                        st.success(f"**Sentiment:** {sentiment} (Confidence: {confidence:.2f})")
                    elif sentiment == "Negative":
                        st.error(f"**Sentiment:** {sentiment} (Confidence: {confidence:.2f})")
                    else:
                        st.info(f"**Sentiment:** {sentiment} (Confidence: {confidence:.2f})")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")

elif page == "Named Entity Recognition":
    st.title("Named Entity Recognition")
    st.markdown("Extract key entities like Persons, Organizations, Locations, Dates, and Money from text.")
    
    with st.form("ner_form"):
        text_input = st.text_area("Enter Text (e.g., Agent Notes):")
        submit = st.form_submit_button("Extract Entities")
        
    if submit and text_input:
        with st.spinner("Extracting..."):
            try:
                response = requests.post(f"{API_BASE_URL}/named-entities", json={"text": text_input})
                if response.status_code == 200:
                    data = response.json()
                    entities = data.get("entities", [])
                    
                    st.subheader("Extracted Entities")
                    if not entities:
                        st.write("No matching entities found.")
                    else:
                        for ent in entities:
                            st.markdown(f"- **{ent['label']}**: {ent['text']}")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")

elif page == "Semantic Search":
    st.title("Semantic Search")
    st.markdown("Search for similar loan applications using natural language.")
    
    with st.form("search_form"):
        query_input = st.text_input("Enter Search Query:")
        top_k = st.slider("Number of Results", min_value=1, max_value=10, value=3)
        submit = st.form_submit_button("Search")
        
    if submit and query_input:
        with st.spinner("Searching..."):
            try:
                response = requests.post(f"{API_BASE_URL}/semantic-search", json={"query": query_input, "top_k": top_k})
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    st.subheader(f"Top {len(results)} Results")
                    for i, res in enumerate(results):
                        with st.expander(f"Rank {i+1} | Score: {res['score']:.4f} | Customer: {res['customer_name']} | Status: {res['loan_status']}"):
                            st.write(res['document'])
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")

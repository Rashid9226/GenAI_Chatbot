import streamlit as st
import requests

# Set FastAPI backend URL
API_URL = "http://localhost:8000/query"  # Change if hosted elsewhere

st.set_page_config(page_title="PDF-RAG Chatbot", page_icon="🩺")
st.title("🩺 Medical Assistant Chatbot")
st.write("Ask a medical question based on the documents loaded into the RAG system.")

# Input field
question = st.text_input("Enter your question:")

# Submit button
if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please enter a valid question.")
    else:
        try:
            with st.spinner("Getting answer..."):
                response = requests.post(API_URL, json={"question": question})
                response.raise_for_status()
                answer = response.json().get("answer", "No answer returned.")
                st.success("Answer:")
                st.write(answer)
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")

# # Health check (optional button)
# if st.button("Check API Health"):
#     try:
#         health = requests.get("http://localhost:8000/health")
#         status = health.json().get("status", "unknown")
#         st.info(f"API Health: {status}")
#     except:
#         st.error("Failed to connect to API.")

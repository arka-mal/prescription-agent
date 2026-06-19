import streamlit as st

USE_BACKEND_GROQ_KEY = True
USE_BACKEND_GCP_CREDS = True

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GCP_SERVICE_ACCOUNT_INFO = dict(st.secrets["gcp_service_account"])
GROQ_MODEL_DEFAULT = "llama-3.3-70b-versatile"
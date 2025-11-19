# test_app_fixed.py
import streamlit as st
import requests
import json
from urllib.parse import urlparse

st.set_page_config(page_title="Intent API Tester", layout="centered")
st.title("Intent API Tester")

st.markdown("This app POSTs to your FastAPI `/predict` endpoint and shows request/response details.")

api_url = st.text_input("Intent API URL (POST)", value="http://127.0.0.1:8080/predict")

# Build health URL robustly:
def build_health_url(api_url: str) -> str:
    u = api_url.rstrip("/")
    if u.endswith("/predict"):
        return u[: -len("/predict")] + "/health"
    # handle if user passed base URL already
    parsed = urlparse(u)
    # if api_url looks like http://host:port  (no path), append /health
    if parsed.path == "" or parsed.path == "/":
        return u + "/health"
    # otherwise default to root /health
    return f"{parsed.scheme}://{parsed.netloc}/health"

health_url = build_health_url(api_url)
st.write("Derived health URL:", health_url)

if st.button("Check API Health"):
    try:
        resp = requests.get(health_url, timeout=5)
        st.write("Health status code:", resp.status_code)
        try:
            st.json(resp.json())
        except Exception:
            st.text(resp.text)
    except Exception as e:
        st.error("Health check failed: " + str(e))

st.write("---")

default_text = "Show my 2nd year marks for Computer Networks"
text = st.text_area("User query", value=default_text, height=120)
req_id = st.text_input("Request ID", value="test-r1")

if st.button("Send POST /predict"):
    payload = {"request_id": req_id, "text": text}
    st.subheader("Request (sent)")
    st.code(json.dumps(payload, indent=2))
    try:
        with st.spinner("Sending POST..."):
            resp = requests.post(api_url, json=payload, timeout=30)
        st.write("HTTP status:", resp.status_code)
        st.write("Response headers:")
        st.write(dict(resp.headers))
        st.subheader("Response body")
        try:
            st.json(resp.json())
        except Exception:
            st.text(resp.text)
    except Exception as e:
        st.error("Request failed: " + str(e))
        st.code(f"curl -X POST {api_url} -H 'Content-Type: application/json' -d '{json.dumps(payload)}'")

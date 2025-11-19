# app.py (Streamlit)
import streamlit as st
import requests
import json

st.set_page_config(page_title="Nexus Intent UI", layout="centered")
st.title("Nexus — Intent Agent UI")

api_url = st.text_input("Intent API URL", value="http://127.0.0.1:8080/predict")
st.write("Using API:", api_url)

with st.form("q"):
    text = st.text_area("User query", height=120, value="Show my 2nd year marks for Computer Networks")
    request_id = st.text_input("Request ID (optional)", value="r1")
    submitted = st.form_submit_button("Send")

if submitted:
    if not text.strip():
        st.warning("Enter text")
    else:
        payload = {"request_id": request_id, "text": text}
        st.subheader("Sending payload")
        st.json(payload)
        try:
            with st.spinner("Calling intent API..."):
                res = requests.post(api_url, json=payload, timeout=20)
            st.write("HTTP status:", res.status_code)
            try:
                data = res.json()
                st.success("Received response")
                st.subheader("Response (raw)")
                st.json(data)
                st.subheader("Friendly view")
                st.write("Intent:", data.get("primary_intent"))
                st.write("Entities:", data.get("entities"))
                st.write("Query descriptor:", data.get("query_descriptor"))
            except Exception as e:
                st.error("Response not JSON: " + str(e))
                st.text(res.text)
        except Exception as e:
            st.error("Request failed: " + str(e))
            st.code(f"curl -X POST {api_url} -H 'Content-Type: application/json' -d '{{\"request_id\":\"{request_id}\",\"text\":\"{text}\"}}'")

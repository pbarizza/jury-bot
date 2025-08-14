# streamlit_app/app.py
import streamlit as st
import requests
import pandas as pd
import json

# FastAPI backend URL (on Render)
FASTAPI_URL = "https://your-fastapi.onrender.com/generate-startup"

st.set_page_config(page_title="Startup Generator", layout="wide")
st.title("🚀 Register a New Synthetic Startup")

st.markdown("""
This app generates a **privacy-safe synthetic startup** using AI and stores it in DuckDB.
""")

if st.button("Register Startup", type="primary"):
    with st.spinner("Generating synthetic startup..."):
        try:
            response = requests.post(FASTAPI_URL)
            if response.status_code == 200:
                data = response.json()
                startup = data["startup"]
                startup_id = data["id"]

                st.success(f"✅ Startup #{startup_id} registered!")

                # Convert to DataFrame for display
                df = pd.DataFrame([startup])
                st.dataframe(df, use_container_width=True)

                # Optional: Show raw JSON
                with st.expander("View Raw JSON"):
                    st.json(startup)

            else:
                st.error(f"❌ Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")
else:
    st.info("Click the button above to generate a new synthetic startup.")

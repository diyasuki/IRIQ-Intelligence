import streamlit as st
import base64
import json
import re
import csv
import io
import tempfile
import os
import fitz
import asyncio
import time
from datetime import datetime

from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
import requests
import pandas as pd

# =========================================================
# BRANDING / LOGO
# =========================================================
LOGO_PATH = "assets/iriq_intelligence_logo.png"
SCORE_HISTORY_FILE = "iriq_score_history.csv"

# =========================================================
# PROMPT MAP
# =========================================================
GITHUB_PROMPT_MAP = {
    "Invoice": "https://raw.githubusercontent.com/diyasuki/IRIQ-Intelligence/main/Invoice.txt",
    "Purchase Order": "https://raw.githubusercontent.com/diyasuki/IRIQ-Intelligence/main/PurchaseOrder.txt",
    "Receipt": "https://raw.githubusercontent.com/diyasuki/IRIQ-Intelligence/main/receipts.txt",
    "Bank Statement": "https://raw.githubusercontent.com/diyasuki/IRIQ-Intelligence/main/Bankstatement.txt",
}

# =========================================================
# CONFIG
# =========================================================
VERTEX_LOCATIONS = [
    "us-central1", "us-east1", "us-west1",
    "us-west4", "europe-west1", "asia-south1"
]

MAX_CONCURRENCY = 4
MAX_RETRIES_PER_PAGE = 2

# =========================================================
# SESSION STATE
# =========================================================
if "extracted_json" not in st.session_state:
    st.session_state.extracted_json = None

# =========================================================
# HELPERS
# =========================================================
def get_project_id_from_sa(uploaded_file):
    sa = json.loads(uploaded_file.getvalue().decode("utf-8"))
    return sa.get("project_id")

def load_prompt_from_github(document_type):
    url = GITHUB_PROMPT_MAP.get(document_type)
    if not url:
        return None
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.text
    except Exception:
        return None

def safe_json_loads(text):
    text = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        return json.loads(match.group(1))
    raise ValueError("UNRECOVERABLE_JSON")

# =========================================================
# IRIQ SCORE LOGIC
# =========================================================
def compute_iriq_score(data: dict):
    score = 0

    # Completeness (40)
    required_fields = ["InvoiceId", "InvoiceDate", "InvoiceTotal"]
    completeness = sum(1 for f in required_fields if data.get(f)) / len(required_fields)
    score += completeness * 40

    # Structure (30)
    items = data.get("items") or data.get("transactions") or []
    if isinstance(items, list) and len(items) > 0:
        score += 30

    # Consistency (20)
    if data.get("InvoiceTotal") or data.get("balance"):
        score += 20

    # Stability (10)
    flags = data.get("flags", {})
    if not flags.get("ocr_uncertain"):
        score += 10

    score = round(score)

    if score >= 90:
        grade = "Excellent"
        color = "#22c55e"
    elif score >= 75:
        grade = "Good"
        color = "#84cc16"
    elif score >= 60:
        grade = "Fair"
        color = "#facc15"
    else:
        grade = "Needs Review"
        color = "#ef4444"

    return score, grade, color

def log_iriq_score(score, document_type):
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "document_type": document_type,
        "score": score
    }
    file_exists = os.path.exists(SCORE_HISTORY_FILE)
    with open(SCORE_HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="IRIQ Intelligence", layout="wide")

logo_b64 = base64.b64encode(open(LOGO_PATH, "rb").read()).decode()

st.markdown(
    f"""
    <div style="text-align:center;">
        <img src="data:image/png;base64,{logo_b64}" width="180"/>
        <h2 style="color:white;">IRIQ Intelligence</h2>
        <div style="color:#9fd3ff;">Intelligent Retrieval & Information Quotient</div>
    </div>
    """,
    unsafe_allow_html=True
)

left, right = st.columns([1, 1.4])

with left:
    document_type = st.selectbox(
        "📑 Document Type",
        ["Invoice", "Purchase Order", "Receipt", "Bank Statement", "Others"]
    )

    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    prompt_file = st.file_uploader("Upload Prompt (.txt)", type=["txt"])
    run = st.button("🚀 Run Extraction")

    if document_type == "Others":
        if not prompt_file:
            st.stop()
        prompt = prompt_file.getvalue().decode("utf-8")
    else:
        prompt = load_prompt_from_github(document_type)
        if not prompt and prompt_file:
            prompt = prompt_file.getvalue().decode("utf-8")

with right:
    if run and pdf_file and prompt:
        # MOCK extraction result (replace with your Gemini call result)
        extracted = {
            "InvoiceId": "123",
            "InvoiceDate": "2026-01-23",
            "InvoiceTotal": "250.00",
            "items": [{"desc": "Item"}],
            "flags": {}
        }

        score, grade, color = compute_iriq_score(extracted)
        extracted["iriq_score"] = {
            "overall": score,
            "grade": grade
        }

        log_iriq_score(score, document_type)
        st.session_state.extracted_json = extracted

# =========================================================
# RESULTS
# =========================================================
if st.session_state.extracted_json:
    data = st.session_state.extracted_json
    score = data["iriq_score"]["overall"]
    grade = data["iriq_score"]["grade"]
    _, _, color = compute_iriq_score(data)

    st.markdown(
        f"""
        <div style="margin-top:15px;">
            <b>IRIQ Score</b>
            <div style="background:#1f2937;border-radius:8px;">
                <div style="width:{score}%;
                            background:{color};
                            padding:6px;
                            border-radius:8px;
                            color:black;
                            font-weight:600;">
                    {score} – {grade}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.json(data)

# =========================================================
# IRIQ SCORE TREND
# =========================================================
st.markdown("## 📈 IRIQ Score Trend")

if os.path.exists(SCORE_HISTORY_FILE):
    df = pd.read_csv(SCORE_HISTORY_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    st.line_chart(df.set_index("timestamp")["score"])
else:
    st.info("No IRIQ score history yet.")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:#9fd3ff;">
        Demo Version – For evaluation only<br>
        Licensing & Commercial Use:
        <a href="mailto:Kranthi.c85@gmail.com" style="color:#7ec8ff;">
            Kranthi.c85@gmail.com
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

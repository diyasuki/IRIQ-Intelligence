import streamlit as st
import base64
import json
import re
import csv
import io
import tempfile
import os
import fitz  # PyMuPDF
import asyncio
import time

from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
import requests
import base64

with st.sidebar:
    logo_sidebar = base64.b64encode(
        open("assets/iriq_intelligence_logo.png", "rb").read()
    ).decode()

    st.markdown(
        f"""
        <div style="text-align:center; margin-bottom:15px;">
            <img src="data:image/png;base64,{logo_sidebar}" width="110"/>
            <div style="font-size:12px; color:#b9ddff; margin-top:6px;">
                IRIQ Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
GITHUB_PROMPT_MAP = {
    "Invoice": "https://github.com/diyasuki/IRIQ-Intelligence/blob/main/Invoice.txt",
    "Purchase Order": "https://github.com/diyasuki/IRIQ-Intelligence/blob/main/PurchaseOrder.txt",
    "Receipt": "https://github.com/diyasuki/IRIQ-Intelligence/blob/main/receipts.txt",
    "Bank Statement": "https://github.com/diyasuki/IRIQ-Intelligence/blob/main/Bankstatement.txt",
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

def load_prompt_from_github(document_type: str) -> str | None:
    url = GITHUB_PROMPT_MAP.get(document_type)
    if not url:
        return None

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.text
    except Exception as e:
        st.warning(f"⚠️ Unable to load prompt from GitHub for {document_type}")
        return None
def safe_json_loads(text: str):
    text = re.sub(r"```json|```", "", text).strip()

    # Strategy 1: normal JSON block
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # Strategy 2: progressive trim
    for i in range(len(text), 0, -1):
        snippet = text[:i]
        try:
            if snippet.count("{") == snippet.count("}"):
                return json.loads(snippet)
        except Exception:
            continue

    # Strategy 3: force close braces
    open_b = text.count("{")
    close_b = text.count("}")
    if open_b > close_b:
        fixed = text + ("}" * (open_b - close_b))
        try:
            return json.loads(fixed)
        except Exception:
            pass

    raise ValueError("UNRECOVERABLE_JSON")


# =========================================================
# PDF PAGE SPLIT
# =========================================================
def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        single = fitz.open()
        single.insert_pdf(doc, from_page=i, to_page=i)
        pages.append((i, single.write()))
    return pages


# =========================================================
# MERGE PAGE RESULTS (DICT / LIST SAFE)
# =========================================================
def merge_page_results(results):
    merged = {}
    all_txns = []

    for _, page_data in sorted(results, key=lambda x: x[0]):
        if isinstance(page_data, list):
            all_txns.extend(page_data)
            continue

        if isinstance(page_data, dict):
            for k, v in page_data.items():
                if isinstance(v, list):
                    if k == "transactions":
                        all_txns.extend(v)
                    else:
                        merged.setdefault(k, []).extend(v)
                else:
                    if k not in merged or merged[k] in (None, "", {}):
                        merged[k] = v

    if all_txns:
        merged["transactions"] = all_txns

    return merged


# =========================================================
# GEMINI STREAMING (SYNC, RETRY SAFE)
# =========================================================
def call_gemini_stream_sync(
    pdf_bytes,
    prompt,
    model,
    page_index,
    progress_queue,
    max_retries=MAX_RETRIES_PER_PAGE
):
    for attempt in range(max_retries + 1):
        try:
            part = Part.from_dict({
                "inline_data": {
                    "mime_type": "application/pdf",
                    "data": base64.b64encode(pdf_bytes).decode()
                }
            })

            stream = model.generate_content(
                contents=[part, prompt],
                generation_config=GenerationConfig(
                    temperature=0.0,
                    response_mime_type="application/json"
                ),
                stream=True
            )

            text = ""
            for chunk in stream:
                if hasattr(chunk, "text") and chunk.text:
                    text += chunk.text
                    progress_queue.put_nowait(page_index)

            return page_index, safe_json_loads(text)

        except Exception:
            if attempt == max_retries:
                # FINAL FALLBACK — DO NOT CRASH PIPELINE
                return page_index, {
                    "transactions": [],
                    "flags": {"ocr_uncertain": True}
                }
            time.sleep(1.5)


# =========================================================
# ASYNC WRAPPER
# =========================================================
async def call_gemini_stream_async(
    pdf_bytes,
    prompt,
    model,
    semaphore,
    page_index,
    progress_queue
):
    async with semaphore:
        return await asyncio.to_thread(
            call_gemini_stream_sync,
            pdf_bytes,
            prompt,
            model,
            page_index,
            progress_queue
        )


# =========================================================
# ASYNC PARALLEL EXTRACTION (STREAMLIT SAFE)
# =========================================================
async def extract_parallel_pages_streaming_async(
    pdf_path,
    prompt,
    creds,
    project_id,
    location,
    progress_bar,
    status_box
):
    vertexai.init(project=project_id, location=location, credentials=creds)
    model = GenerativeModel("gemini-2.5-flash")

    pages = extract_pages(pdf_path)
    total_pages = len(pages)

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    progress_queue = asyncio.Queue()
    completed = set()

    async def progress_watcher():
        while len(completed) < total_pages:
            idx = await progress_queue.get()
            completed.add(idx)
            progress_bar.progress(len(completed) / total_pages)
            status_box.info(f"📄 Processing page {idx + 1}/{total_pages}")

    watcher = asyncio.create_task(progress_watcher())

    tasks = [
        call_gemini_stream_async(b, prompt, model, semaphore, i, progress_queue)
        for i, b in pages
    ]

    results = await asyncio.gather(*tasks)
    await watcher

    return merge_page_results(results)


# =========================================================
# TRANSACTIONS → CSV (ONE ROW PER TXN)
# =========================================================
def transactions_to_csv(data: dict) -> str:
    if not isinstance(data, dict):
        return ""

    txns = data.get("transactions", [])
    if not txns:
        return ""

    base = {
        "bank_name": data.get("bank_name"),
        "account_holder_name": data.get("account_holder_name"),
        "account_number": data.get("account_number"),
        "currency": data.get("currency"),
        "statement_from": (data.get("statement_period") or {}).get("from"),
        "statement_to": (data.get("statement_period") or {}).get("to"),
    }

    output = io.StringIO()
    rows = []

    for idx, t in enumerate(txns, start=1):
        row = base.copy()
        row.update({
            "txn_index": idx,
            "date": t.get("date"),
            "description": t.get("description"),
            "amount": t.get("amount"),
            "balance": t.get("balance"),
            "type": t.get("type"),
            "category": t.get("category"),
        })
        rows.append(row)

    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    return output.getvalue()


# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="IRIQ Intelligent Extractor", layout="wide")
st.markdown(
    """
    <style>
        /* ================================
           IRIQ Intelligence – Logo Background
        ================================= */

        .stApp {
            background: radial-gradient(
                circle at top,
                #0f243d 0%,
                #0b1a2a 45%,
                #070f1a 100%
            );
            color: #ffffff;
        }

        /* Sidebar background */
        section[data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #0b1a2a 0%,
                #070f1a 100%
            );
        }

        /* Cards / containers */
        div[data-testid="stVerticalBlock"] > div {
            background: transparent;
        }

        /* File uploader & selectbox containers */
        div[data-testid="stFileUploader"],
        div[data-testid="stSelectbox"] {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 10px;
        }
        

        /* Buttons */
        .stButton > button {
            background-color: #7ec8ff;
            color: #000000;
            font-weight: 600;
            border-radius: 8px;
            border: none;
        }

        .stButton > button:hover {
            background-color: #5bb6ff;
            color: #000000;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
        div[data-testid="stFileUploader"] label,
        div[data-testid="stSelectbox"] label {
            color:#ffffff !important;
        }
        div[data-testid="stSidebar"] label,
        div[data-testid="stSelectbox"] label {
            color:#ffffff !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
        /* ================================
           File Uploader – White Text
        ================================= */

        /* Main uploader text */
        div[data-testid="stFileUploader"] * {
            color: #ffffff !important;
        }

        /* "Drag and drop file here" text */
        div[data-testid="stFileUploader"] span {
            color: #000000 !important;
        }

        div[data-testid="stSelectbox"] > label {
            color: #000000 !important;
        }
        /* Uploaded filename */
        div[data-testid="stFileUploader"] small {
            color: #000000 !important;
        }

        /* Browse files button text */
        div[data-testid="stFileUploader"] button {
            color: #000000 !important;   /* keep readable on blue button */
            font-weight: 600;
        }

        /* Border */
        div[data-testid="stFileUploader"] {
            border: 1px solid rgba(255,255,255,0.25);
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
        /* ================================
           BUTTON TEXT → BLACK
        ================================= */
        .stButton > button {
            color: #000000 !important;
            font-weight: 600;
        }

        .stButton > button:hover {
            color: #000000 !important;
        }

        /* ================================
           SELECTBOX (DROPDOWN) TEXT → BLACK
           Covers:
           - Selected value
           - Placeholder
           - Dropdown options
        ================================= */

        /* Selected value */
        div[data-testid="stSelectbox"] div[role="combobox"] {
            color: #000000 !important;
        }

        /* Placeholder text */
        div[data-testid="stSelectbox"] span {
            color: #000000 !important;
        }

        /* Dropdown menu items */
        ul[role="listbox"] li {
            color: #000000 !important;
            background-color: #ffffff !important;
        }

        /* Selected item highlight */
        ul[role="listbox"] li[aria-selected="true"] {
            background-color: #dbeafe !important;
            color: #000000 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="text-align:center; padding: 10px 0 5px 0;">
        <img src="data:image/png;base64,{logo}" width="180"/>
        <h2 style="color:#ffffff; margin:10px 0 2px 0;">
            IRIQ Intelligence
        </h2>
        <div style="color:#9fd3ff; font-size:15px;">
            Intelligent Document Processing & AI Extraction Platform
        </div>
        <div style="color:#7ec8ff; font-size:13px; margin-top:4px;">
            Powered by <b>IRIQ Intelligence</b>
        </div>
    </div>
    """.format(
        logo=base64.b64encode(
            open("assets/iriq_intelligence_logo.png", "rb").read()
        ).decode()
    ),
    unsafe_allow_html=True
)
#st.markdown(
#    """
#    <div style="
#        text-align:center;
#        padding:12px;
#        margin-bottom:15px;
#        border-radius:10px;
#        background: linear-gradient(135deg, #3a1b66, #2b124c);
 #       border: 1px solid #7ec8ff;
  #  ">
  #      <h3 style="margin:0; color:#ffffff;">IRIQ Intelligence</h3>
  #      <span style="font-size:12px; color:#b9ddff;">
  #          AI Document Automation
  #      </span>
  #  </div>
  #  """,
  #  unsafe_allow_html=True
#)
with st.sidebar:
    service_account_file = st.file_uploader(
        "Service Account JSON",
        type=["json"],
        key="service_account_json"
    )

    project_id = get_project_id_from_sa(service_account_file) if service_account_file else None
    if project_id:
        st.text_input("Project ID", project_id, disabled=True, key="project_id_display")

    location = st.selectbox(
        "Vertex AI Location",
        VERTEX_LOCATIONS,
        key="vertex_location"
    )
left, right = st.columns([1, 1.4])

with left:
    pdf_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key="upload_pdf"
    )

    prompt_file = st.file_uploader(
        "Upload Prompt (.txt)",
        type=["txt"],
        key="upload_prompt"
    )

    run = st.button(
        "🚀 Run Extraction",
        use_container_width=True,
        key="run_extraction"
    )

    status_box = st.empty()
    progress_bar = st.progress(0.0)

with left:
    document_type = st.selectbox(
        "📑 Document Type",
        ["Invoice", "Purchase Order", "Receipt", "Bank Statement", "Others"]
    )

    auto_prompt = None
    prompt = None

    if document_type == "Others":
        if not prompt_file:
            st.error("For 'Others', uploading a prompt text file is mandatory.")
            st.stop()

        prompt = prompt_file.getvalue().decode("utf-8")

    else:
        auto_prompt = load_prompt_from_github(document_type)

        if auto_prompt:
            st.success(f"✅ Prompt auto-loaded for {document_type}")
            prompt = auto_prompt
        else:
            st.warning("⚠️ Auto prompt not available. Please upload a prompt file.")

            if not prompt_file:
                st.stop()

            prompt = prompt_file.getvalue().decode("utf-8")

    status_box = st.empty()
    progress_bar = st.progress(0.0)


with right:
    json_out = st.empty()
    dl_json = st.empty()
    dl_csv = st.empty()

# =========================================================
# ACTION
# =========================================================
if run:
    if not all([pdf_file, service_account_file, prompt]):
        st.error("Upload PDF, Service Account JSON, and ensure a Prompt is loaded")
        st.stop()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as sa_tmp:
        sa_tmp.write(service_account_file.getvalue())
        sa_path = sa_tmp.name

    creds = service_account.Credentials.from_service_account_file(
        sa_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = os.path.join(tmp, pdf_file.name)
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.read())

        
        st.session_state.extracted_json = asyncio.run(
            extract_parallel_pages_streaming_async(
                pdf_path,
                prompt,
                creds,
                project_id,
                location,
                progress_bar,
                status_box
            )
        )

        status_box.success("✅ Extraction completed")

# =========================================================
# RENDER + DOWNLOAD
# =========================================================
if st.session_state.extracted_json:
    json_out.json(st.session_state.extracted_json)

    dl_json.download_button(
        "⬇️ Download JSON",
        json.dumps(st.session_state.extracted_json, indent=2,ensure_ascii = False),
        "output.json",
        mime="application/json; charset=utf-8"
    )

    dl_csv.download_button(
        "⬇️ Download Transactions CSV",
        transactions_to_csv(st.session_state.extracted_json),
        "transactions.csv",
        mime="text/csv"
    )


st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        padding:12px;
        margin-bottom:15px;
        border-radius:10px;
        background: linear-gradient(135deg, #3a1b66, #2b124c);
        border: 1px solid #7ec8ff;
    ">
        <h3 style="margin:0; color:#ffffff;">This application is for demonstration and evaluation purposes only</h3>
        <span style="font-size:12px; color:#b9ddff;">
            Licensing & Commercial Use
        </span>
        <span style="font-size:12px; color:#b9ddff;">
            For licensing, customization, or enterprise deployment, contact:
        </span>
        <span style="font-size:12px; color:#b9ddff;">
            <a href="mailto:Kranthi.c85@gmail.com"
                 style="color:#7ec8ff; text-decoration:none; font-weight:500;">
                Kranthi.c85@gmail.com
            </a>
        </span>
    </div>
    """,
    unsafe_allow_html=True
)



















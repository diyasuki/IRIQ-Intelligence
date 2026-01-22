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
st.set_page_config(page_title="Async Gemini Bank Statement Extractor", layout="wide")
st.title("📄 Async Gemini Bank Statement Extractor")

with st.sidebar:
    service_account_file = st.file_uploader("Service Account JSON", type=["json"])
    project_id = get_project_id_from_sa(service_account_file) if service_account_file else None
    if project_id:
        st.text_input("Project ID", project_id, disabled=True)

    location = st.selectbox("Vertex AI Location", VERTEX_LOCATIONS)

left, right = st.columns([1, 1.4])

with left:
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    prompt_file = st.file_uploader("Upload Prompt (.txt)", type=["txt"])
    run = st.button("🚀 Run Extraction", use_container_width=True)
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
    if not all([pdf_file, prompt_file, service_account_file]):
        st.error("Upload PDF, Prompt, and Service Account JSON")
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

        prompt = prompt_file.read().decode()

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
        json.dumps(st.session_state.extracted_json, indent=2),
        "output.json",
        mime="application/json"
    )

    dl_csv.download_button(
        "⬇️ Download Transactions CSV",
        transactions_to_csv(st.session_state.extracted_json),
        "transactions.csv",
        mime="text/csv"
    )

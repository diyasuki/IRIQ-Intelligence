#from img2table.ocr import PaddleOCR
#from img2table.document import PDF
import pandas as pd
import fitz
import os
from img2table.document import Image
import tempfile
from img2table.ocr import EasyOCR
import numpy as np
import streamlit as st
import cv2
from datetime import datetime
from skimage.filters import threshold_sauvola
import io
import img2table.tables.processing as proc
import sys
st.write("Python:", sys.version)
os.environ["EASYOCR_MODULE_PATH"] = os.path.join(os.getcwd(), ".easyocr")

@st.cache_resource
def get_ocr():
    import traceback
    try:
        return EasyOCR(lang=["en"])
    except Exception as e:
        st.error("EasyOCR failed to initialize. Full error below:")
        st.text(traceback.format_exc())
        raise

ocr = get_ocr()
def render_page_to_png_bytes(page, dpi=300):
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")

def extract_tables_from_png_bytes(img_bytes, ocr):
    with tempfile.TemporaryDirectory() as tmp:
        img_path = os.path.join(tmp, "page.png")
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        doc_img = Image(src=img_path)  # ✅ must be a file path
        tables = doc_img.extract_tables(
            ocr=ocr,
            implicit_rows=True,
            implicit_columns=True,
            borderless_tables=True
        )

        dfs = []
        for t in tables:
            df = t.df
            if df is not None and len(df) > 0:
                dfs.append(pd.DataFrame(df))
        return dfs

def pdf_to_full_df(uploaded_pdf_bytes, ocr, dpi=300):
    page_set = fitz.open(stream=uploaded_pdf_bytes, filetype="pdf")

    full_df = pd.DataFrame()
    for pg, page in enumerate(page_set):
        img_bytes = render_page_to_png_bytes(page, dpi=dpi)
        page_dfs = extract_tables_from_png_bytes(img_bytes, ocr)

        for ext_df in page_dfs:
            # remove header-like rows
            col0 = ext_df.columns[0]
            ext_df = ext_df[~ext_df[col0].astype(str).str.contains("Date", na=False)]
            full_df = pd.concat([full_df, ext_df], ignore_index=True)

    if not full_df.empty:
        col0 = full_df.columns[0]
        full_df[col0] = full_df[col0].ffill()

    return full_df

def detect_template(df):
    txt = " ".join(df.astype(str).fillna("").values.ravel()).lower()
    if "hsbcn" in txt:
        return "HSBC"
    if "to contact u.s. bank" in txt or "uslbank" in txt or "800-us banks" in txt:
        return "U.S. BANK"
    return ""

def build_final_df(df, template, years=2025):
    column_names = ['Date', 'Description', 'Debit or Credit','Amount','Balance']
    final_df = pd.DataFrame(columns=column_names)

    if df.empty:
        return final_df

    # Your column mapping assumptions (keep as-is)
    dates = df.columns[0]
    decname = df.columns[1]
    debitname = df.columns[2]
    creditname = df.columns[3]
    balance = df.columns[4]

    df[decname] = df[decname].astype(str)
    df[dates] = df[dates].astype(str)

    word_list = ["apr","may","jan","feb","mar","jun","jul","aug","sep","oct","nov","dec","0ct"]

    if template == "HSBC":
        for _, row in df.iterrows():
            my_string = str(row[dates])
            found_any = any(w in my_string.lower() for w in word_list)

            if "forward" in str(row[decname]).lower():
                continue

            if found_any:
                if pd.notna(row[debitname]) and str(row[debitname]).strip() != "":
                    final_df.loc[len(final_df)] = [row[dates], row[decname], 'Debit', row[debitname], row[balance]]
                elif pd.notna(row[creditname]) and str(row[creditname]).strip() != "":
                    final_df.loc[len(final_df)] = [row[dates], row[decname], 'Credit', row[creditname], row[balance]]

    if template == "U.S. BANK":
        def is_number(s):
            try:
                float(str(s).replace(",","").replace("—","").replace("-",""))
                return True
            except:
                return False

        df[creditname] = df[creditname].astype(str)

        for _, row in df.iterrows():
            my_string = str(row[dates])
            combo_desc = str(row[decname] + " " + row[creditname])
            found_any = any(w in my_string.lower() for w in word_list)

            if "total" in combo_desc.lower():
                continue

            if found_any:
                s1 = str(row[dates]).replace('Mobile Banking Transfer','').replace('\n','').replace('Electronic Deposit','')
                amt = row[balance]

                if is_number(amt) and pd.notna(amt) and str(amt).strip() != "":
                    if "—" in str(amt) or "-" in str(amt):
                        final_df.loc[len(final_df)] = [f"{s1} {years}", combo_desc, "Debit", str(amt).replace("—","").replace("-",""), ""]
                    else:
                        final_df.loc[len(final_df)] = [f"{s1} {years}", combo_desc, "Credit", amt, ""]

    return final_df


def threshold_dark_areas_no_ximgproc(img, char_length=11):
    """
    Replacement for img2table.tables.threshold_dark_areas
    Avoids cv2.ximgproc.niBlackThreshold (missing on Streamlit Cloud with EasyOCR)
    """
    # img is BGR uint8
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    # window size must be odd and >= 3
    win = int(char_length)
    if win < 3:
        win = 3
    if win % 2 == 0:
        win += 1

    # Sauvola threshold (similar purpose to niBlackThreshold Sauvola)
    th = threshold_sauvola(gray, window_size=win, k=0.2)

    # Equivalent to THRESH_BINARY_INV (text dark -> white foreground in mask)
    bin_img = (gray < th).astype(np.uint8) * 255
    return bin_img
proc.threshold_dark_areas = threshold_dark_areas_no_ximgproc
templates=""
#PADDLE_PDX_DISABLE_DEV_MODEL_WL=1
#ocr = PaddleOCR(lang="en",device="cpu") # Speciy the language
st.title("Data Extractor")
st.subheader("Application to Extract Statement from PDF's to Downloadable DataFrames")

############################# Sidebar ########################
st.sidebar.title("Guide")
st.sidebar.markdown("> Quality screenshot images taken from phone, laptop, etc is preferred")
st.sidebar.markdown("> Images captured with phone camera yields incomplete data")
st.sidebar.markdown("""> Images affected by artifacts including partial occulsion, distorted perspective, 
                    and complex background yields incomplete tables""")
st.sidebar.markdown(""" > Handwriting recognition on images containing tables will be significantly harder
                       due to infinite variations of handwriting styles and limitations of optical character recognition""")

print(cv2.__version__)
print("ximgproc:", hasattr(cv2, "ximgproc"))
print("niBlackThreshold:", hasattr(cv2.ximgproc, "niBlackThreshold") if hasattr(cv2, "ximgproc") else None)
###################### loading images #######################
uploaded_file = st.file_uploader("Choose an image | Accepted formats: only PDF", type=("pdf"))
if uploaded_file is not None:


    pdf_bytes = uploaded_file.getvalue()

    # Build table DF from PDF
    df = pdf_to_full_df(pdf_bytes, ocr, dpi=300)
    df = df.replace('', np.nan)
    
    st.subheader("DataFrame")
    st.dataframe(df)
    
    template = detect_template(df)
    st.write(f"Found Template: {template}")
    
    final_df = build_final_df(df, template, years=2025)
    
    st.subheader("Extracted Transactions")
    st.dataframe(final_df)
    















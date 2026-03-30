import streamlit as st
import numpy as np
from scipy.optimize import linprog
from PIL import Image
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cv2
import io
from fpdf import FPDF
import pytesseract
import shutil

# ── Tesseract path ──────────────────────────────────────────
import os, platform
tess_path = shutil.which("tesseract")
if tess_path:
    pytesseract.pytesseract.tesseract_cmd = tess_path
elif platform.system() == "Windows":
    for _p in [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"D:\Program Files\Tesseract-OCR\tesseract.exe",
    ]:
        if os.path.exists(_p):
            pytesseract.pytesseract.tesseract_cmd = _p
            break

st.set_page_config(page_title="Duality LP Solver", layout="wide",
                   initial_sidebar_state="expanded")

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"
theme = st.sidebar.radio("Theme", ["Light", "Dark"],
                         index=0 if st.session_state.theme == "Light" else 1)
st.session_state.theme = theme

if theme == "Light":
    BG="#F7F9FC"; SURFACE="#FFFFFF"; SURFACE2="#EEF3FA"; BORDER="#D1DCF0"
    ACCENT="#2563EB"; ACCENT2="#7C3AED"; ACCENT3="#059669"; WARN="#DC2626"
    TEXT="#0F172A"; TEXT_MED="#334155"; TEXT_MUTED="#64748B"
    TAG_BG="#EEF2FF"; TAG_TEXT="#3730A3"; SUCCESS_BG="#ECFDF5"; ERROR_BG="#FEF2F2"
    BTN_BG="#2563EB"; BTN_TEXT="#FFFFFF"
    CHART_COLORS=["#38BDF8","#A78BFA","#34D399","#FBBF24","#F87171","#FB923C"]
else:
    BG="#080D14"; SURFACE="#0F1923"; SURFACE2="#152030"; BORDER="#1E3A5F"
    ACCENT="#38BDF8"; ACCENT2="#A78BFA"; ACCENT3="#34D399"; WARN="#F87171"
    TEXT="#F0F6FF"; TEXT_MED="#CBD5E1"; TEXT_MUTED="#94A3B8"
    TAG_BG="#1E3A5F"; TAG_TEXT="#93C5FD"; SUCCESS_BG="#052E16"; ERROR_BG="#450A0A"
    BTN_BG="#38BDF8"; BTN_TEXT="#080D14"
    CHART_COLORS=["#38BDF8","#A78BFA","#34D399","#FBBF24","#F87171","#FB923C"]

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]{{background-color:{BG}!important;color:{TEXT}!important;font-family:'Space Grotesk',sans-serif!important;}}
[data-testid="stSidebar"]{{background-color:{SURFACE}!important;border-right:1px solid {BORDER};}}
[data-testid="stSidebar"] *{{color:{TEXT}!important;}}
h1,h2,h3,h4,h5,h6{{font-family:'Space Grotesk',sans-serif!important;color:{TEXT}!important;letter-spacing:-0.02em;}}
p,span,li,div,label{{color:{TEXT_MED}!important;}}
code{{background:{SURFACE2}!important;color:{ACCENT}!important;font-family:'JetBrains Mono',monospace!important;padding:2px 6px;border-radius:4px;}}
.lp-card{{background:{SURFACE};border:1px solid {BORDER};border-radius:16px;padding:24px 28px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,0.15);}}
input,textarea,select{{background-color:{SURFACE2}!important;color:{TEXT}!important;border:1px solid {BORDER}!important;border-radius:10px!important;}}
[data-baseweb="input"] input,[data-baseweb="textarea"] textarea,.stNumberInput input,.stTextArea textarea{{color:{TEXT}!important;background:{SURFACE2}!important;}}
[data-baseweb="select"] div{{background:{SURFACE2}!important;color:{TEXT}!important;border-color:{BORDER}!important;}}
[data-baseweb="popover"],[data-baseweb="menu"]{{background:{SURFACE}!important;border:1px solid {BORDER}!important;}}
[data-baseweb="option"]{{color:{TEXT}!important;background:{SURFACE}!important;}}
[data-baseweb="option"]:hover{{background:{SURFACE2}!important;}}
.stButton>button{{background:{BTN_BG}!important;color:{BTN_TEXT}!important;border:none!important;border-radius:10px!important;font-weight:700!important;font-size:0.92rem!important;padding:11px 28px!important;transition:all 0.2s ease!important;}}
.stButton>button:hover{{filter:brightness(1.1);transform:translateY(-1px);box-shadow:0 6px 20px {ACCENT}44!important;}}
.stDownloadButton>button{{background:{ACCENT3}!important;color:#080D14!important;border:none!important;border-radius:10px!important;font-weight:700!important;width:100%;}}
[data-testid="stMetric"]{{background:{SURFACE}!important;border:1px solid {BORDER};border-top:3px solid {ACCENT};border-radius:14px;padding:14px 18px;}}
[data-testid="stMetricLabel"] p{{color:{TEXT_MUTED}!important;font-size:0.78rem!important;font-weight:600!important;text-transform:uppercase;letter-spacing:0.06em;}}
[data-testid="stMetricValue"]{{color:{TEXT}!important;font-size:1.6rem!important;font-weight:700!important;font-family:'JetBrains Mono',monospace!important;}}
.stTabs [data-baseweb="tab-list"]{{background:{SURFACE2}!important;border-radius:12px;padding:4px;gap:2px;border:1px solid {BORDER};}}
.stTabs [data-baseweb="tab"]{{background:transparent!important;color:{TEXT_MUTED}!important;border-radius:8px!important;border:none!important;font-weight:600!important;font-size:0.85rem!important;padding:8px 16px!important;}}
.stTabs [aria-selected="true"]{{background:{ACCENT}!important;color:{BTN_TEXT}!important;}}
[data-testid="stDataFrame"]{{border:1px solid {BORDER};border-radius:12px;overflow:hidden;}}
.stSuccess{{background:{SUCCESS_BG}!important;border:1px solid {ACCENT3};border-radius:10px;}}
.stError{{background:{ERROR_BG}!important;border:1px solid {WARN};border-radius:10px;}}
.stInfo{{background:{SURFACE2}!important;border:1px solid {ACCENT};border-radius:10px;}}
.stSuccess p,.stError p,.stInfo p{{color:{TEXT}!important;}}
[data-testid="stExpander"]{{border:1px solid {BORDER}!important;border-radius:12px!important;background:{SURFACE}!important;}}
[data-testid="stExpander"] summary{{color:{TEXT}!important;font-weight:600!important;}}
[data-testid="stFileUploader"]{{background:{SURFACE2}!important;border:2px dashed {BORDER}!important;border-radius:14px!important;}}
.stSelectbox label,.stNumberInput label,.stTextArea label,.stFileUploader label{{color:{TEXT}!important;font-weight:600!important;font-size:0.85rem!important;}}
hr{{border-color:{BORDER}!important;}}
::-webkit-scrollbar{{width:6px;height:6px;}}
::-webkit-scrollbar-track{{background:{SURFACE};}}
::-webkit-scrollbar-thumb{{background:{BORDER};border-radius:3px;}}
</style>""", unsafe_allow_html=True)

st.markdown(f"""
<div style="background:linear-gradient(135deg,{SURFACE} 0%,{SURFACE2} 100%);border:1px solid {BORDER};
border-radius:20px;padding:32px 36px 28px;margin-bottom:28px;position:relative;overflow:hidden;">
<div style="position:absolute;top:-40px;right:-40px;width:180px;height:180px;border-radius:50%;
background:radial-gradient(circle,{ACCENT}22,transparent 70%);"></div>
<span style="font-size:0.75rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:{ACCENT};">Linear Programming</span>
<h1 style="font-size:2.4rem;font-weight:800;margin:6px 0 8px;color:{TEXT}!important;letter-spacing:-0.03em;">&#x2202; Duality LP Solver</h1>
<p style="color:{TEXT_MUTED}!important;font-size:1rem;margin:0;">
Primal &rarr; Dual &nbsp;&middot;&nbsp; Simplex &amp; Big-M &nbsp;&middot;&nbsp; Feasible Region &nbsp;&middot;&nbsp; PDF &amp; CSV Export</p>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"<hr style='border-color:{BORDER};margin:8px 0 16px;'>", unsafe_allow_html=True)
    st.markdown(f"""<div style='color:{TEXT_MUTED}!important;font-size:0.82rem;line-height:1.8;'>
&#x2022; <b style='color:{TEXT_MED}!important;'>Primal &rarr; Dual</b> conversion<br>
&#x2022; <b style='color:{TEXT_MED}!important;'>Simplex</b> (&le; constraints)<br>
&#x2022; <b style='color:{TEXT_MED}!important;'>Big-M</b> (&ge; / = constraints)<br>
&#x2022; <b style='color:{TEXT_MED}!important;'>Image OCR</b> (Tesseract)<br>
&#x2022; <b style='color:{TEXT_MED}!important;'>Feasible Region</b> (2-var)<br>
&#x2022; <b style='color:{TEXT_MED}!important;'>PDF &amp; CSV</b> export
</div>""", unsafe_allow_html=True)

BIG_M = 1e4

# ============================================================
# UNICODE NORMALISER  ← THE KEY FIX FOR SUBSCRIPT VARIABLES
# ============================================================
_UNICODE_MAP = {
    # Subscript digits  ₀₁₂₃₄₅₆₇₈₉  (these appear when users paste from Word/PDF)
    '\u2080':'0','\u2081':'1','\u2082':'2','\u2083':'3','\u2084':'4',
    '\u2085':'5','\u2086':'6','\u2087':'7','\u2088':'8','\u2089':'9',
    # Superscript digits
    '\u00b9':'1','\u00b2':'2','\u00b3':'3',
    '\u2070':'0','\u2074':'4','\u2075':'5','\u2076':'6',
    '\u2077':'7','\u2078':'8','\u2079':'9',
    # Math inequality symbols  ← FIXES ≤ and ≥ not being recognized
    '\u2264':'<=',  # ≤
    '\u2265':'>=',  # ≥
    '\u2266':'<=',
    '\u2267':'>=',
    '\u2272':'<=',
    '\u2273':'>=',
    '\u226a':'<<',
    '\u226b':'>>',
    # Math operators
    '\u2212':'-',   # minus sign (different from hyphen!)
    '\u00d7':'*',   # ×
    '\u00f7':'/',   # ÷
    '\u2260':'!=',
    '\u00b1':'+',
    '\u2192':'->',
    '\u221e':'inf',
    '\u2248':'~=',
    '\u00a0':' ',   # non-breaking space  ← VERY COMMON CAUSE OF PARSE FAILURES
    '\u2009':' ',   # thin space
    '\u200b':'',    # zero-width space
    '\u200c':'',    # zero-width non-joiner
    '\u200d':'',    # zero-width joiner
    '\u2022':'',    # bullet
    '\u25cf':'',    # black circle
    # Greek
    '\u03b1':'a','\u03b2':'b','\u03b3':'c',
    '\u03bb':'lambda','\u03bc':'u','\u03c3':'s',
}

def normalize_unicode(text: str) -> str:
    """Replace every known unicode symbol with its ASCII equivalent."""
    for char, repl in _UNICODE_MAP.items():
        text = text.replace(char, repl)
    return text

# ============================================================
# PDF SANITISER
# ============================================================
def _s(text):
    """Sanitize any text for FPDF (latin-1 safe). Catches ALL unicode."""
    text = str(text)
    text = normalize_unicode(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    result = ''
    for ch in text:
        try:    ch.encode('latin-1'); result += ch
        except: result += '?'
    return result

# ============================================================
# OCR
# ============================================================
@st.cache_resource
def get_tesseract_available():
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

def preprocess_for_ocr(image_np):
    gray     = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords   = np.column_stack(np.where(thresh > 128))
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90+angle) if angle < -45 else -angle
        if abs(angle) < 15:
            h, w = enhanced.shape
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            enhanced = cv2.warpAffine(enhanced, M, (w, h),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    large = cv2.resize(binary, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(large, -1, kernel)

def ocr_extract(image_np):
    preprocessed = preprocess_for_ocr(image_np)
    from PIL import Image as PILImg
    pil_img = PILImg.fromarray(preprocessed)
    lp_kw = re.compile(
        r'(maximiz|minimiz|subject|s\.t\.|x\s*\d|z\s*=|<=|>=|[≤≥]|\d+\s*x)',
        re.IGNORECASE)
    configs = ["--psm 6 --oem 3", "--psm 4 --oem 3", "--psm 11 --oem 3"]
    best_text, best_score, results_all = "", -1, {}
    for cfg in configs:
        try:
            text  = pytesseract.image_to_string(pil_img, config=cfg)
            score = len(lp_kw.findall(text))
            results_all[cfg] = text
            if score > best_score:
                best_score, best_text = score, text
        except Exception:
            pass
    return best_text or results_all.get(configs[0], ""), results_all

# ============================================================
# LP CLEANER  ← FIXED: handles subscripts, ≤/≥, and "Minimize Z=..."
# ============================================================
def clean_lp(text: str) -> str:
    # Step 1: Normalise ALL unicode first (subscripts, ≤, ≥, minus signs, spaces)
    text = normalize_unicode(text)

    # Step 2: Remaining symbol aliases
    text = text.replace('≤', '<=').replace('≥', '>=')
    text = text.replace('×', '*').replace('✕', '*')

    # Step 3: OCR digit confusions
    text = re.sub(r'(?<![a-zA-Z])O(?=\d)', '0', text)
    text = re.sub(r'(?<=\d)l(?!\d)',        '1', text)

    # Step 4: Keyword markers — must come BEFORE collapsing spaces
    text = re.sub(r'\b(subject\s*to|s\.t\.)\b', '\nST\n',       text, flags=re.IGNORECASE)
    text = re.sub(r'\b(maximize|maximise|max)\b', '\nMAXIMIZE\n', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(minimize|minimise|min)\b', '\nMINIMIZE\n', text, flags=re.IGNORECASE)

    # Step 5: Remove commas that are list separators
    text = re.sub(r',\s*(?=[a-zA-Z])', ' ', text)
    text = re.sub(r',\s*(?=\d)',        ' ', text)

    # Step 6: Collapse spaces inside variable tokens: "x 1" → "x1", "x  2" → "x2"
    text = re.sub(r'\bx\s+(\d{1,2})\b', r'x\1', text, flags=re.IGNORECASE)

    # Step 7: Attach coefficient to variable: "2 x1" → "2x1"
    text = re.sub(r'(\d)\s+(x\d{1,2})\b', r'\1\2', text, flags=re.IGNORECASE)

    # Step 8: Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()

# ============================================================
# LP PARSER  ← FIXED: correct handling of "Z = 3x1 + 5x2" without Maximize/Minimize keyword on same line
# ============================================================
_TERM_RE = re.compile(
    r'([+-]?)\s*(\d+(?:\.\d+)?)?\s*\*?\s*x(\d{1,2})',
    re.IGNORECASE
)

def _parse_expr(expr: str) -> dict:
    """Return {0-based var index: coefficient}."""
    expr = re.sub(r'([+-])', r' \1 ', expr)
    expr = re.sub(r'\s+', ' ', expr).strip()
    d = {}
    for sign, coef, idx in _TERM_RE.findall(expr):
        i  = int(idx) - 1
        c  = float(coef) if coef else 1.0
        sv = -1.0 if sign == '-' else 1.0
        d[i] = d.get(i, 0.0) + sv * c
    return d

def _is_nonnegativity(line: str) -> bool:
    """Return True for lines like 'x1, x2 >= 0' — skip these."""
    stripped = re.sub(r'\bx\d{1,2}\b', '', line, flags=re.IGNORECASE)
    stripped = stripped.replace(',', '').strip()
    return bool(re.fullmatch(r'\s*(>=|>|<=|<|=)\s*0\.?\s*', stripped))

def parse_lp(text: str):
    lines  = [l.strip() for l in text.split('\n') if l.strip()]
    c      = []
    A      = []
    b      = []
    ineq   = []
    opt    = "Maximize"
    n_vars = 0

    def _ensure(idx):
        nonlocal n_vars
        while n_vars <= idx:
            c.append(0.0)
            n_vars += 1

    obj_found = False

    for line in lines:
        ll = line.lower()

        # Direction keywords
        if 'maximize' in ll:
            opt = 'Maximize'
        if 'minimize' in ll:
            opt = 'Minimize'

        # Skip non-negativity declarations
        if _is_nonnegativity(line):
            continue

        # ── Objective: matches "Z = ...", "Maximize Z = ...", "Z: ..." ──
        # Also matches bare "Maximize 3x1 + 5x2" without Z=
        is_obj_line = bool(re.search(r'\bz\s*[=:]', ll))
        is_obj_line = is_obj_line or (
            ('maximize' in ll or 'minimize' in ll) and _TERM_RE.search(line)
        )

        if is_obj_line and not obj_found:
            obj_found = True
            # Take everything after the first '=' or ':', or after max/min keyword
            if re.search(r'[=:]', line):
                rhs_part = re.split(r'[=:]', line, maxsplit=1)[-1]
            else:
                # "Maximize 3x1 + 5x2" — strip the keyword
                rhs_part = re.sub(r'\b(maximize|maximise|minimize|minimise|max|min)\b', '',
                                  line, flags=re.IGNORECASE)
            d = _parse_expr(rhs_part)
            for idx, val in d.items():
                _ensure(idx)
                c[idx] = val
            continue

        # ── Constraint lines ──
        m = re.search(r'(<=|>=|(?<![<>])=(?![>=]))', line)
        if m:
            sign  = m.group(1)
            parts = re.split(r'<=|>=|(?<![<>])=(?![>=])', line, maxsplit=1)
            if len(parts) != 2:
                continue

            lhs_d = _parse_expr(parts[0])
            if not lhs_d:
                continue   # no variables on LHS → skip (e.g. non-negativity)

            nums = re.findall(r'[+-]?\s*\d+(?:\.\d+)?', parts[1])
            if not nums:
                continue
            try:
                rhs_v = float(nums[0].replace(' ', ''))
            except ValueError:
                continue

            for idx in lhs_d:
                _ensure(idx)

            A.append(lhs_d)
            b.append(rhs_v)
            ineq.append(sign)

    if not obj_found and not A:
        return [], [], [], [], opt

    # Build dense A
    tv = max(n_vars, len(c), 1)
    while len(c) < tv:
        c.append(0.0)

    A_dense = []
    for row_d in A:
        row = [0.0] * tv
        for idx, val in row_d.items():
            if idx < tv:
                row[idx] = val
        A_dense.append(row)

    return c, A_dense, b, ineq, opt

def validate_lp(c, A, b, ineq):
    if not c:
        return False, "No objective coefficients found."
    if not A:
        return False, "No constraints found."
    n = len(c)
    for i, row in enumerate(A):
        if len(row) != n:
            return False, f"Constraint {i+1} has {len(row)} vars but objective has {n}."
    if len(A) != len(b) or len(A) != len(ineq):
        return False, "Mismatch between A, b, and inequality signs."
    if all(abs(v) < 1e-12 for v in c):
        return False, "All objective coefficients are zero — check your input."
    return True, ""

# ============================================================
# PREPROCESSING — positive RHS
# ============================================================
def normalize_constraints(A, b, ineq):
    A_out, b_out, iq_out = [], [], []
    flip = {"<=":">=", ">=":"<=", "=":"="}
    for i in range(len(b)):
        if b[i] < 0:
            A_out.append([-v for v in A[i]])
            b_out.append(-b[i])
            iq_out.append(flip[ineq[i]])
        else:
            A_out.append(list(A[i]))
            b_out.append(b[i])
            iq_out.append(ineq[i])
    return A_out, b_out, iq_out

# ============================================================
# DUAL
# ============================================================
def generate_dual(c, A, b, ineq, opt):
    At = np.array(A, dtype=float).T.tolist()
    d_bounds = []
    for s in ineq:
        if   s == "<=": d_bounds.append((0, None))
        elif s == ">=": d_bounds.append((None, 0))
        else:           d_bounds.append((None, None))
    dineq = [">=" if opt == "Maximize" else "<=" for _ in c]
    dtype = "Minimize" if opt == "Maximize" else "Maximize"
    return list(b), At, list(c), dineq, dtype, d_bounds

def solve_dual(dc, dA, db, dineq, dtype, d_bounds):
    sign_d = 1 if dtype == "Minimize" else -1
    dc_arr = [sign_d * v for v in dc]
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    n = min(len(dA), len(dineq), len(db))
    for i in range(n):
        if   dineq[i] == "<=": A_ub.append(dA[i]);              b_ub.append(db[i])
        elif dineq[i] == ">=": A_ub.append([-v for v in dA[i]]); b_ub.append(-db[i])
        else:                   A_eq.append(dA[i]);              b_eq.append(db[i])
    return linprog(dc_arr,
                   A_ub=A_ub or None, b_ub=b_ub or None,
                   A_eq=A_eq or None, b_eq=b_eq or None,
                   bounds=d_bounds, method="highs")

# ============================================================
# SIMPLEX
# ============================================================
def simplex_iterations(c, A, b, max_iter=50):
    n_con, n_var = len(A), len(c)
    tableau = []
    for i, row in enumerate(A):
        slack = [1.0 if j==i else 0.0 for j in range(n_con)]
        tableau.append(list(row) + slack + [b[i]])
    tableau.append([-ci for ci in c] + [0.0]*n_con + [0.0])
    basis = [f"s{i+1}" for i in range(n_con)]
    cols  = [f"x{i+1}" for i in range(n_var)] + [f"s{i+1}" for i in range(n_con)] + ["RHS"]
    T     = np.array(tableau, dtype=float)
    records = []
    for it in range(max_iter):
        obj = T[-1, :-1]
        pc  = int(np.argmin(obj))
        if obj[pc] >= -1e-9: break
        ratios = np.where(T[:-1,pc] > 1e-9, T[:-1,-1]/T[:-1,pc], np.inf)
        pr = int(np.argmin(ratios))
        if ratios[pr] == np.inf: break
        entering, leaving = cols[pc], basis[pr]
        T[pr] /= T[pr, pc]
        for i in range(len(T)):
            if i != pr: T[i] -= T[i,pc] * T[pr]
        basis[pr] = entering
        df = pd.DataFrame(T, columns=cols,
                          index=[f"R{i+1}" for i in range(n_con)]+["Z"]).round(4)
        records.append({"iteration":it+1,"entering":entering,"leaving":leaving,
                        "basis":list(basis),"tableau":df,"method":"Simplex"})
    sol = [0.0]*n_var
    for i, bv in enumerate(basis):
        m = re.match(r'x(\d+)', bv)
        if m:
            xi = int(m.group(1))-1
            if xi < n_var: sol[xi] = round(T[i,-1], 6)
    return records, sol, round(T[-1,-1], 6)

# ============================================================
# BIG-M
# ============================================================
def bigm_iterations(c, A, b, ineq, opt, max_iter=80):
    n_v, n_c = len(c), len(A)
    sign  = -1.0 if opt == "Maximize" else 1.0
    c_arr = [sign*ci for ci in c]
    n_slack = sum(1 for iq in ineq if iq in ("<=",">="))
    n_artif = sum(1 for iq in ineq if iq in (">=","="))
    total   = n_v + n_slack + n_artif + 1
    cols = ([f"x{i+1}" for i in range(n_v)] +
            [f"s{i+1}" for i in range(n_slack)] +
            [f"a{i+1}" for i in range(n_artif)] + ["RHS"])
    T = np.zeros((n_c+1, total))
    slack_map, artif_map, ks, ka = {}, {}, 0, 0
    for i, iq in enumerate(ineq):
        T[i, :n_v] = A[i]; T[i, -1] = b[i]
        if iq == "<=":
            T[i, n_v+ks] = 1.0;  slack_map[i] = n_v+ks;  ks+=1
        elif iq == ">=":
            T[i, n_v+ks] = -1.0; slack_map[i] = n_v+ks;  ks+=1
            T[i, n_v+n_slack+ka] = 1.0; artif_map[i] = n_v+n_slack+ka; ka+=1
        else:
            T[i, n_v+n_slack+ka] = 1.0; artif_map[i] = n_v+n_slack+ka; ka+=1
    basis = [cols[slack_map[i]] if ineq[i]=="<=" else cols[artif_map[i]] for i in range(n_c)]
    for j in range(n_v): T[-1, j] = c_arr[j]
    for ac in artif_map.values(): T[-1, ac] = BIG_M
    for ri, iq in enumerate(ineq):
        if iq in (">=","="): T[-1] -= T[-1, artif_map[ri]] * T[ri]
    records = []
    for it in range(max_iter):
        obj = T[-1, :-1]
        pc  = int(np.argmin(obj))
        if obj[pc] >= -1e-9: break
        ratios = np.where(T[:-1,pc]>1e-9, T[:-1,-1]/T[:-1,pc], np.inf)
        pr = int(np.argmin(ratios))
        if ratios[pr] == np.inf: break
        entering, leaving = cols[pc], basis[pr]
        T[pr] /= T[pr, pc]
        for i in range(len(T)):
            if i != pr: T[i] -= T[i,pc] * T[pr]
        basis[pr] = entering
        df = pd.DataFrame(T, columns=cols,
                          index=[f"R{i+1}" for i in range(n_c)]+["Z"]).round(4)
        records.append({"iteration":it+1,"entering":entering,"leaving":leaving,
                        "basis":list(basis),"tableau":df,"method":"Big-M"})
    feasible = all(not (bv and bv.startswith('a') and abs(T[i,-1])>1e-6)
                   for i, bv in enumerate(basis))
    sol = [0.0]*n_v
    for i, bv in enumerate(basis):
        m = re.match(r'x(\d+)', bv)
        if m:
            xi = int(m.group(1))-1
            if xi < n_v: sol[xi] = round(T[i,-1], 6)
    raw = round(T[-1,-1], 6)
    return records, sol, (-raw if opt=="Maximize" else raw), feasible

# ============================================================
# SMART SOLVER
# ============================================================
def solve_primal(c, A, b, ineq, opt):
    A, b, ineq = normalize_constraints(A, b, ineq)
    needs_bigm = any(iq in (">=","=") for iq in ineq)
    if needs_bigm:
        recs, sol, obj, feas = bigm_iterations(c, A, b, ineq, opt)
        return recs, sol, obj, "Big-M", feas
    else:
        try:
            recs, sol, obj = simplex_iterations(c, [list(r) for r in A], list(b))
            return recs, sol, (obj if opt=="Maximize" else -obj), "Simplex", True
        except Exception:
            return [], [], 0.0, "Simplex", False

# ============================================================
# ANALYSIS
# ============================================================
def generate_analysis(c, A, b, ineq, opt, primal_sol, primal_obj,
                       dual_sol, dual_obj, method_used, feasible,
                       dc, dA, db, dineq, dtype):
    n, m = len(c), len(A)
    lines = []
    if method_used == "Big-M":
        ge = sum(1 for iq in ineq if iq==">=")
        eq = sum(1 for iq in ineq if iq=="=")
        lines.append(f"**Why Big-M?** Your problem has {ge} '>=' and {eq} '=' constraint(s). "
            "Standard Simplex needs a ready identity basis. Big-M adds artificial variables "
            f"with penalty M={BIG_M:.0e} to force them out, giving a valid starting BFS.")
    else:
        lines.append("**Why Simplex?** All constraints are '<=' so slack variables form "
            "a natural identity basis. The Simplex method pivots toward the optimal corner "
            "of the feasible polyhedron by choosing the most negative reduced cost each step.")

    active = [(f"x{i+1}", round(primal_sol[i],4)) for i in range(n) if abs(primal_sol[i])>1e-6]
    zeros  = [f"x{i+1}" for i in range(n) if abs(primal_sol[i])<=1e-6]
    if active:
        lines.append(f"**Optimal Primal:** {', '.join(f'{v}={val}' for v,val in active)}. "
            + (f"Variables {', '.join(zeros)} are non-basic (=0) at optimality." if zeros else
               "All variables are active at the optimal vertex."))
    else:
        lines.append("**Primal Solution:** All variables are zero — may be trivial or degenerate.")

    gap = abs(primal_obj - dual_obj)
    if gap < 1e-3 and feasible:
        lines.append(f"**Strong Duality holds:** Primal Z = {round(primal_obj,4)}, "
            f"Dual W = {round(dual_obj,4)}. Both objectives match - confirming global optimality.")
    elif not feasible:
        lines.append("**WARNING - Feasibility issue:** Artificial variables remain in the basis. "
            "The problem may be infeasible or unbounded. Review constraint RHS values.")
    else:
        lines.append(f"**Duality gap = {round(gap,6)}** - small numerical discrepancy (floating-point).")

    binding = []
    for i in range(m):
        lhs = sum(A[i][j]*primal_sol[j] for j in range(n))
        if abs(lhs - b[i]) < 1e-4:
            binding.append(f"C{i+1}")
    if binding:
        lines.append(f"**Binding Constraints:** {', '.join(binding)} are active (LHS=RHS). "
            "These define the optimal vertex. Relaxing them would allow a better objective.")

    nz_dual = [(f"y{i+1}", round(dual_sol[i],4)) for i in range(len(dual_sol)) if abs(dual_sol[i])>1e-6]
    if nz_dual:
        lines.append(f"**Shadow Prices:** {', '.join(f'{v}={val}' for v,val in nz_dual)}. "
            "Each yi = marginal value of relaxing constraint i by 1 unit.")

    lines.append(f"**Objective ({opt}):** Z = {' + '.join(f'{c[i]}*x{i+1}' for i in range(n))}. "
        f"Optimal Z* = {round(primal_obj,4)} - best achievable within all constraints.")
    return lines

# ============================================================
# VISUALIZATION
# ============================================================
def plot_results(c, A, b, ineq, opt, primal_sol, primal_obj, dual_sol, dual_obj):
    n_vars = len(c)
    ncols  = 3 if n_vars == 2 else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 6))
    if ncols == 1:
        axes = [axes]
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=TEXT_MED, labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

    ax1   = axes[0]
    mx    = max(primal_sol) if any(v>0 for v in primal_sol) else 1
    bars  = ax1.bar([f"x{i+1}" for i in range(n_vars)], primal_sol,
                    color=CHART_COLORS[:n_vars], edgecolor='none', zorder=3, width=0.5, alpha=0.9)
    for bar, val in zip(bars, primal_sol):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+mx*0.02,
                 f'{val:.3f}', ha='center', va='bottom', color=TEXT_MED, fontsize=9, fontweight='bold')
    ax1.set_title("Primal Solution", color=TEXT_MED, fontsize=11, fontweight='bold', pad=10)
    ax1.set_ylabel("Value", color=TEXT_MED, fontsize=9)
    ax1.yaxis.grid(True, color=BORDER, linestyle='--', alpha=0.5, zorder=0)
    ax1.set_axisbelow(True); ax1.axhline(0, color=BORDER, linewidth=0.8)

    ax2  = axes[1]
    vals = np.array(dual_sol)
    if vals.sum() > 1e-9:
        _, _, ats = ax2.pie(np.abs(vals)+1e-9,
            labels=[f"y{i+1}={v:.3f}" for i,v in enumerate(dual_sol)],
            colors=CHART_COLORS[:len(dual_sol)], autopct="%1.1f%%",
            textprops={"color":TEXT_MED,"fontsize":8},
            wedgeprops={"linewidth":2,"edgecolor":SURFACE}, startangle=90)
        for at in ats: at.set_color(SURFACE); at.set_fontweight('bold')
    else:
        ax2.text(0.5,0.5,"All dual\nvariables = 0", ha='center', va='center',
                 color=TEXT_MED, fontsize=11, transform=ax2.transAxes)
    ax2.set_title("Dual Variables (Shadow Prices)", color=TEXT_MED, fontsize=11, fontweight='bold', pad=10)

    if n_vars == 2:
        ax3 = axes[2]
        ax3.set_facecolor(SURFACE)
        ax3.tick_params(colors=TEXT_MED, labelsize=9)
        for sp in ax3.spines.values(): sp.set_edgecolor(BORDER)
        max_b  = max(b) if b else 10
        x_max  = max(max_b*1.5, primal_sol[0]*2+1, 5)
        x      = np.linspace(0, x_max, 500)
        c_cols = [ACCENT3, WARN, ACCENT2, ACCENT, "#FBBF24", "#FB923C"]
        for i, (row, rhs, sign) in enumerate(zip(A, b, ineq)):
            a1, a2 = row[0], row[1]
            col    = c_cols[i % len(c_cols)]
            lbl    = f"C{i+1}: {a1}x1+{a2}x2 {sign} {rhs}"
            if abs(a2) > 1e-9:
                y    = (rhs - a1*x) / a2
                mask = (y >= -0.5) & (y <= x_max*1.2)
                ax3.plot(x[mask], y[mask], color=col, linewidth=2, label=lbl, alpha=0.9)
            elif abs(a1) > 1e-9:
                ax3.axvline(rhs/a1, color=col, linewidth=2, label=lbl, alpha=0.9)
        try:
            from scipy.spatial import ConvexHull
            pts = []
            gx  = np.linspace(0, x_max, 80)
            gy  = np.linspace(0, x_max, 80)
            for xi in gx:
                for yi in gy:
                    ok = True
                    for row, rhs, sign in zip(A, b, ineq):
                        v = row[0]*xi + row[1]*yi
                        if sign=="<=" and v > rhs+1e-6: ok=False; break
                        if sign==">=" and v < rhs-1e-6: ok=False; break
                        if sign=="="  and abs(v-rhs)>1e-4: ok=False; break
                    if ok: pts.append([xi, yi])
            if len(pts) >= 3:
                pts_arr = np.array(pts)
                hull    = ConvexHull(pts_arr)
                poly    = Polygon(pts_arr[hull.vertices], closed=True,
                                  facecolor=ACCENT+"28", edgecolor=ACCENT,
                                  linewidth=1.5, linestyle='--', alpha=0.8)
                ax3.add_patch(poly)
        except Exception:
            pass
        ox, oy = primal_sol[0], primal_sol[1]
        ax3.scatter([ox],[oy], color=ACCENT, s=200, zorder=10,
                    edgecolors='white', linewidth=2, label=f"Optimal ({ox:.2f},{oy:.2f})")
        ax3.annotate(f"Z*={round(primal_obj,2)}",
                     xy=(ox,oy), xytext=(ox+x_max*0.05, oy+x_max*0.06),
                     color=ACCENT, fontsize=9, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1.5))
        ax3.set_xlim(-x_max*0.04, x_max); ax3.set_ylim(-x_max*0.04, x_max)
        ax3.set_xlabel("x1", color=TEXT_MED, fontsize=10)
        ax3.set_ylabel("x2", color=TEXT_MED, fontsize=10)
        ax3.set_title("Feasible Region & Optimal Point", color=TEXT_MED,
                      fontsize=11, fontweight='bold', pad=10)
        ax3.legend(fontsize=7, framealpha=0.3, facecolor=SURFACE2,
                   edgecolor=BORDER, labelcolor=TEXT_MED, loc='upper right')
        ax3.yaxis.grid(True, color=BORDER, linestyle='--', alpha=0.3)
        ax3.xaxis.grid(True, color=BORDER, linestyle='--', alpha=0.3)
        ax3.set_axisbelow(True)

    plt.tight_layout(pad=2.5)
    return fig

# ============================================================
# PDF BUILDER
# ============================================================
def build_pdf(primal_c, primal_A, primal_b, primal_ineq, primal_opt,
              dual_c, dual_A, dual_b, dual_ineq, dual_opt,
              sol_x, sol_obj, pivot_records, primal_sol, primal_obj,
              method_used, analysis_lines):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_fill_color(37,99,235); pdf.rect(0,0,210,44,'F')
    pdf.set_font("Arial","B",20); pdf.set_text_color(255,255,255)
    pdf.cell(0,15,"",ln=True)
    pdf.cell(0,14,"  Duality LP Solver - Solution Report",ln=True)
    pdf.set_font("Arial","",10)
    pdf.cell(0,8,_s(f"  Method: {method_used}  |  Primal Z={round(primal_obj,4)}  |  Dual W={round(sol_obj,4)}"),ln=True)
    pdf.set_text_color(0,0,0); pdf.ln(8)

    def sec(title, r=37, g=99, b_=235):
        pdf.set_fill_color(r,g,b_); pdf.set_text_color(255,255,255)
        pdf.set_font("Arial","B",12)
        pdf.cell(0,9,_s(f"  {title}"),ln=True,fill=True)
        pdf.set_text_color(0,0,0); pdf.ln(2)

    n = len(primal_c)
    sec("1. Primal Problem")
    pdf.set_font("Courier","",10)
    pdf.cell(0,7,_s(f"  {primal_opt} Z = " + " + ".join(f"{primal_c[i]}*x{i+1}" for i in range(n))),ln=True)
    pdf.cell(0,7,"  Subject to:",ln=True)
    for i in range(len(primal_A)):
        pdf.cell(0,7,_s("    " + " + ".join(f"{primal_A[i][j]}*x{j+1}" for j in range(n)) +
                         f"  {primal_ineq[i]}  {primal_b[i]}"),ln=True)
    pdf.cell(0,7,"  All variables >= 0",ln=True)
    pdf.set_font("Arial","B",10)
    pdf.cell(0,7,_s(f"Optimal Z = {round(primal_obj,6)}  x = {[round(v,4) for v in primal_sol]}"),ln=True)
    pdf.ln(4)

    m = len(dual_c)
    sec("2. Dual Problem", 124,58,237)
    pdf.set_font("Courier","",10)
    pdf.cell(0,7,_s(f"  {dual_opt} W = " + " + ".join(f"{dual_c[i]}*y{i+1}" for i in range(m))),ln=True)
    pdf.cell(0,7,"  Subject to:",ln=True)
    for i in range(len(dual_A)):
        pdf.cell(0,7,_s("    " + " + ".join(f"{dual_A[i][j]}*y{j+1}" for j in range(len(dual_A[i]))) +
                         f"  {dual_ineq[i]}  {dual_b[i]}"),ln=True)
    pdf.cell(0,7,"  All variables >= 0",ln=True)
    pdf.set_font("Arial","B",10)
    pdf.cell(0,7,_s(f"Optimal W = {round(sol_obj,6)}  y = {[round(v,4) for v in sol_x]}"),ln=True)
    pdf.ln(4)

    sec("3. Analysis & Interpretation", 5,46,22)
    pdf.set_font("Arial","",9)
    for line in analysis_lines:
        pdf.multi_cell(0,6,_s(f"  {line}")); pdf.ln(1)
    pdf.ln(4)

    if pivot_records:
        sec(f"4. {method_used} Pivot Table", 15,23,42)
        pdf.set_font("Arial","B",9)
        cw = [16,26,26,100,20]
        for h_,w_ in zip(["Iter","Entering","Leaving","Basis","Method"],cw):
            pdf.cell(w_,7,h_,border=1,align="C")
        pdf.ln()
        pdf.set_font("Arial","",8)
        for rec in pivot_records:
            pdf.cell(cw[0],6,str(rec["iteration"]),border=1,align="C")
            pdf.cell(cw[1],6,_s(rec["entering"]),border=1,align="C")
            pdf.cell(cw[2],6,_s(rec["leaving"]),border=1,align="C")
            pdf.cell(cw[3],6,_s(", ".join(rec["basis"])[:50]),border=1)
            pdf.cell(cw[4],6,_s(rec["method"]),border=1,align="C")
            pdf.ln()
        pdf.ln(4)
        pdf.set_font("Arial","B",10)
        pdf.cell(0,8,"Detailed Tableaux:",ln=True); pdf.ln(2)
        for rec in pivot_records:
            if pdf.get_y() > 240: pdf.add_page()
            pdf.set_font("Arial","B",9); pdf.set_text_color(37,99,235)
            pdf.cell(0,7,_s(f"  Iter {rec['iteration']}  Enter:{rec['entering']}  "
                            f"Leave:{rec['leaving']}  Basis:{', '.join(rec['basis'])}"),ln=True)
            pdf.set_text_color(0,0,0)
            df  = rec["tableau"]; nc = len(df.columns)
            cw2 = min(20, int(190//(nc+1)))
            pdf.set_font("Arial","B",7); pdf.set_fill_color(220,235,255)
            pdf.cell(cw2,5,"Row",border=1,align="C",fill=True)
            for col in df.columns:
                pdf.cell(cw2,5,_s(str(col)[:6]),border=1,align="C",fill=True)
            pdf.ln(); pdf.set_font("Courier","",6)
            for ridx, rrow in df.iterrows():
                pdf.cell(cw2,5,_s(str(ridx)[:5]),border=1,align="C")
                for val in rrow: pdf.cell(cw2,5,f"{val:.2f}",border=1,align="R")
                pdf.ln()
            pdf.ln(3)

    # ← FIXED: use bytes(pdf.output()) instead of encode('latin-1')
    buf = io.BytesIO()
    buf.write(bytes(pdf.output()))
    buf.seek(0)
    return buf

def build_csv(primal_sol, primal_obj, dual_sol, dual_obj):
    rows = []
    for i,v in enumerate(primal_sol): rows.append({"Type":"Primal","Variable":f"x{i+1}","Value":round(v,6)})
    rows.append({"Type":"Primal","Variable":"Z","Value":round(primal_obj,6)})
    for i,v in enumerate(dual_sol):   rows.append({"Type":"Dual","Variable":f"y{i+1}","Value":round(v,6)})
    rows.append({"Type":"Dual","Variable":"W","Value":round(dual_obj,6)})
    return pd.DataFrame(rows).to_csv(index=False)

# ============================================================
# FORMULA RENDERER  ← FIXED: safe int conversion, handles empty lists
# ============================================================
def render_lp_formula(c, A, b, ineq, opt, is_dual=False):
    # Guard against empty inputs
    if not c or not A:
        return f"<div style='color:{WARN};padding:12px;'>No valid LP data to display.</div>"

    vs, os_ = ("y","W") if is_dual else ("x","Z")
    n = len(c)

    def term(coef, var):
        try:
            coef_f = float(coef)
            coef_d = int(coef_f) if coef_f == int(coef_f) else coef_f
        except (TypeError, ValueError):
            coef_d = coef
        return (f"<span style='color:{ACCENT};font-weight:700;"
                f"font-family:JetBrains Mono,monospace;'>{coef_d}</span>{var}")

    obj_html = (f"<span style='color:{ACCENT2};font-weight:700;'>{opt}</span> &nbsp;"
                f"<span style='color:{TEXT};font-weight:700;font-family:JetBrains Mono,monospace;'>{os_} = </span>"
                + " + ".join(term(c[i], f"{vs}<sub>{i+1}</sub>") for i in range(n)))

    con_rows = ""
    for i, row in enumerate(A):
        if i >= len(b) or i >= len(ineq):
            break
        lhs = " + ".join(term(row[j], f"{vs}<sub>{j+1}</sub>") for j in range(len(row)))
        # ← FIXED: safe conversion — handles None, non-numeric, large floats
        try:
            bv    = float(b[i])
            rv    = int(bv) if bv == int(bv) else round(bv, 4)
        except (TypeError, ValueError):
            rv = b[i]
        ic = ACCENT3 if ineq[i]=="<=" else (WARN if ineq[i]==">=" else ACCENT2)
        con_rows += (f"<div style='margin:4px 0;padding:6px 12px;background:{SURFACE2};"
                     f"border-radius:8px;border-left:3px solid {ic};'>"
                     f"<span style='color:{TEXT_MUTED};font-size:0.75rem;'>C{i+1}&nbsp;</span>"
                     f"{lhs}&nbsp;<span style='color:{ic};font-weight:700;'>{ineq[i]}</span>&nbsp;"
                     f"<span style='color:{ACCENT};font-weight:700;"
                     f"font-family:JetBrains Mono,monospace;'>{rv}</span></div>")

    nn = ", ".join(f"{vs}<sub>{i+1}</sub>" for i in range(n))
    return (f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:14px;padding:20px 24px;'>"
            f"<div style='margin-bottom:12px;font-size:1rem;'>{obj_html}</div>"
            f"<div style='color:{TEXT_MUTED};font-size:0.8rem;margin-bottom:8px;font-weight:600;"
            f"text-transform:uppercase;letter-spacing:0.06em;'>Subject to</div>{con_rows}"
            f"<div style='margin-top:10px;color:{TEXT_MUTED};font-size:0.8rem;font-style:italic;'>"
            f"{nn} &ge; 0</div></div>")

# ============================================================
# INPUT SECTION
# ============================================================
st.markdown("<div class='lp-card' style='padding-bottom:12px;'>", unsafe_allow_html=True)
st.markdown(f"<span style='font-size:1.1rem;font-weight:700;color:{TEXT}!important;'>&#x1F4E5; Input Mode</span>",
            unsafe_allow_html=True)
mode = st.selectbox("", ["✏️ Manual Entry","📝 Text / Equation","🖼️ Image OCR","📊 CSV Upload"],
                    label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div class='lp-card'>", unsafe_allow_html=True)

# ── Manual Entry ────────────────────────────────────────────
if mode == "✏️ Manual Entry":
    st.markdown(f"<h3 style='margin-top:0;color:{TEXT}!important;'>Manual Entry</h3>", unsafe_allow_html=True)
    col1,col2,col3 = st.columns(3)
    num_vars = col1.number_input("Variables",1,10,2)
    num_cons = col2.number_input("Constraints",1,10,2)
    opt      = col3.selectbox("Optimisation",["Maximize","Minimize"])
    st.markdown(f"<p style='font-weight:700;color:{TEXT}!important;margin-top:12px;'>Objective Coefficients</p>",
                unsafe_allow_html=True)
    c_cols = st.columns(int(num_vars))
    c = [c_cols[i].number_input(f"c(x{i+1})",value=0.0,key=f"c{i}",format="%.2f") for i in range(int(num_vars))]
    st.markdown(f"<p style='font-weight:700;color:{TEXT}!important;margin-top:12px;'>Constraint Matrix</p>",
                unsafe_allow_html=True)
    A, b, ineq = [], [], []
    for i in range(int(num_cons)):
        cols = st.columns(int(num_vars)+2)
        row  = [cols[j].number_input(f"a{i+1},{j+1}",value=0.0,key=f"a{i}_{j}",format="%.2f")
                for j in range(int(num_vars))]
        sign = cols[-2].selectbox("",["<=",">=","="],key=f"s{i}")
        rhs  = cols[-1].number_input("RHS",value=0.0,key=f"b{i}",format="%.2f")
        A.append(row); b.append(rhs); ineq.append(sign)
    if st.button("💾 Save Problem", use_container_width=True):
        ok, err = validate_lp(c, A, b, ineq)
        if ok:
            st.session_state.data = (c, A, b, ineq, opt)
            st.success("✓ Problem saved — click Solve below.")
        else:
            st.error(f"Validation error: {err}")

# ── Text / Equation ─────────────────────────────────────────
elif mode == "📝 Text / Equation":
    st.markdown(f"<h3 style='margin-top:0;color:{TEXT}!important;'>Text / Equation Input</h3>", unsafe_allow_html=True)
    st.markdown(f"""<div style='background:{SURFACE2};border:1px solid {BORDER};border-radius:10px;
    padding:14px 18px;margin-bottom:14px;font-family:JetBrains Mono,monospace;font-size:0.85rem;color:{TEXT_MED}!important;'>
    <b style='color:{ACCENT};'>Supported formats (subscripts x&#x2081; and plain x1 both work):</b><br><br>
    <span style='color:{ACCENT2};'>Maximize</span> Z = 5x1 + 4x2<br>
    6x1 + 4x2 &lt;= 24<br>x1 + 2x2 &lt;= 6<br><br>
    <span style='color:{ACCENT2};'>Minimize</span> Z = 2x1 + 3x2<br>
    x1 + x2 &gt;= 4<br>x1 - x2 = 1<br><br>
    <span style='color:{TEXT_MUTED};font-size:0.78rem;'>
    Both &lt;= and &#x2264; are accepted. Subscript x&#x2081; is auto-converted to x1.</span>
    </div>""", unsafe_allow_html=True)
    text_input = st.text_area("LP Problem", height=180, placeholder="Type your LP problem here…")
    if st.button("🔍 Parse & Save", use_container_width=True):
        cleaned = clean_lp(text_input)
        c, A, b, ineq, opt = parse_lp(cleaned)
        ok, err = validate_lp(c, A, b, ineq)
        if ok:
            st.session_state.data = (c, A, b, ineq, opt)
            nb = any(iq in (">=","=") for iq in ineq)
            st.success(f"✓ Parsed: {len(c)} vars, {len(A)} constraints — **{'Big-M' if nb else 'Simplex'}** method")
            with st.expander("🔎 Parsed Values (verify these are correct)"):
                p1,p2 = st.columns(2)
                p1.markdown(f"**Objective c:** `{c}`")
                p1.markdown(f"**RHS b:** `{b}`")
                p2.markdown(f"**Matrix A:** `{A}`")
                p2.markdown(f"**Signs:** `{ineq}`  **Opt:** `{opt}`")
        else:
            st.error(f"Parse error: {err}")
            st.info("💡 Tip: Write variables as x1, x2. Include 'Maximize Z = ...' on the first line. "
                    "Subscripts like x₁ are automatically handled.")

# ── Image OCR ───────────────────────────────────────────────
elif mode == "🖼️ Image OCR":
    st.markdown(f"<h3 style='margin-top:0;color:{TEXT}!important;'>Image OCR (Tesseract)</h3>", unsafe_allow_html=True)
    tess_ok = get_tesseract_available()
    if not tess_ok:
        st.warning("⚠️ Tesseract not found. OCR unavailable.")
    file = st.file_uploader("Upload image", type=["png","jpg","jpeg","bmp","webp"])
    if file and tess_ok:
        image   = Image.open(file).convert("RGB")
        img_np  = np.array(image)
        ca, cb  = st.columns(2)
        ca.image(image, caption="Original", use_container_width=True)
        pre = preprocess_for_ocr(img_np)
        cb.image(pre, caption="Preprocessed", use_container_width=True, clamp=True)
        with st.spinner("Running OCR…"):
            best_text, all_results = ocr_extract(img_np)
            best_text = normalize_unicode(best_text)
        extracted = st.text_area("Extracted Text (editable)", best_text, height=200, key="ocr_out")
        with st.expander("🔍 Raw OCR configs"):
            for cfg, txt in all_results.items():
                st.markdown(f"**`{cfg}`:**"); st.code(txt)
        co, cb2 = st.columns([2,3])
        opt_ocr = co.selectbox("Optimisation", ["Maximize","Minimize"], key="ocr_opt")
        if cb2.button("🔍 Parse OCR Output", use_container_width=True):
            cleaned = clean_lp(extracted)
            c, A, b, ineq, opt = parse_lp(cleaned)
            opt = opt_ocr
            ok, err = validate_lp(c, A, b, ineq)
            if ok:
                st.session_state.data = (c, A, b, ineq, opt)
                nb = any(iq in (">=","=") for iq in ineq)
                st.success(f"✓ Parsed: {len(c)} vars, {len(A)} constraints — **{'Big-M' if nb else 'Simplex'}**")
            else:
                st.error(f"Parser error: {err} — edit the text above and retry.")

# ── CSV Upload ──────────────────────────────────────────────
elif mode == "📊 CSV Upload":
    st.markdown(f"<h3 style='margin-top:0;color:{TEXT}!important;'>CSV Upload</h3>", unsafe_allow_html=True)
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df    = pd.read_csv(file); st.dataframe(df, use_container_width=True)
        n_var = df.shape[1]-1
        cc2   = st.columns(min(n_var,5))
        c_csv = [cc2[i%min(n_var,5)].number_input(f"Obj x{i+1}",value=1.0,key=f"ccsv{i}") for i in range(n_var)]
        co, cb2 = st.columns(2)
        opt_csv  = co.selectbox("Optimisation",["Maximize","Minimize"],key="csv_opt")
        sign_csv = cb2.selectbox("Constraint type",["<=",">=","="],key="csv_sign")
        if st.button("📥 Load CSV", use_container_width=True):
            A=df.iloc[:,:-1].values.tolist(); b=df.iloc[:,-1].values.tolist()
            ineq=[sign_csv]*len(A)
            ok,err = validate_lp(c_csv,A,b,ineq)
            if ok:
                st.session_state.data=(c_csv,A,b,ineq,opt_csv); st.success("✓ CSV loaded.")
            else:
                st.error(f"Validation: {err}")

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PROBLEM PREVIEW
# ============================================================
if "data" in st.session_state:
    c_p,A_p,b_p,iq_p,op_p = st.session_state.data
    nb     = any(iq in (">=","=") for iq in iq_p)
    ml     = "Big-M Method" if nb else "Simplex Method"
    tc_tag = ACCENT2 if nb else ACCENT3
    st.markdown(f"""<div style='display:flex;align-items:center;gap:8px;margin:14px 0 8px;'>
    <span style='font-size:1.1rem;font-weight:700;color:{TEXT}!important;'>Problem Preview</span>
    <span style='background:{TAG_BG};color:{TAG_TEXT}!important;padding:3px 10px;border-radius:999px;
    font-size:0.72rem;font-weight:700;text-transform:uppercase;'>Primal</span>
    <span style='background:{tc_tag}22;color:{tc_tag}!important;padding:3px 10px;border-radius:999px;
    font-size:0.72rem;font-weight:700;text-transform:uppercase;'>{ml}</span>
    </div>""", unsafe_allow_html=True)
    st.markdown(render_lp_formula(c_p,A_p,b_p,iq_p,op_p), unsafe_allow_html=True)

# ============================================================
# SOLVE
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
_, sc, _ = st.columns([1,2,1])
with sc:
    solve_clicked = st.button("🚀 Solve Problem", use_container_width=True)

if solve_clicked:
    if "data" not in st.session_state:
        st.error("Please enter/parse a problem first."); st.stop()
    c, A, b, ineq, opt = st.session_state.data
    ok, err = validate_lp(c, A, b, ineq)
    if not ok:
        st.error(f"Invalid problem: {err}"); st.stop()

    with st.spinner("Solving primal…"):
        pivot_records, primal_sol, primal_obj, method_used, feasible = solve_primal(c,A,b,ineq,opt)

    dc,dA,db,dineq,dtype,d_bounds = generate_dual(c,A,b,ineq,opt)
    with st.spinner("Solving dual…"):
        res = solve_dual(dc,dA,db,dineq,dtype,d_bounds)
    if not res.success:
        st.error(f"Dual solver failed: {res.message}"); st.stop()
    sol_x   = res.x
    sol_obj = -res.fun if dtype=="Maximize" else res.fun

    st.markdown(f"""<div style='margin:28px 0 18px;'>
    <span style='font-size:0.75rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:{ACCENT};'>Results</span>
    <h2 style='margin:4px 0 0;color:{TEXT}!important;font-size:1.8rem;font-weight:800;'>Solution Report</h2>
    </div>""", unsafe_allow_html=True)

    sc_col = ACCENT3 if feasible else WARN
    st.markdown(f"""<div style='background:{sc_col}22;border:1px solid {sc_col};border-radius:12px;
    padding:12px 20px;margin-bottom:20px;color:{sc_col}!important;font-weight:600;'>
    {"✓ Optimal solution found using " + method_used + " Method" if feasible else
     "⚠ Feasibility issue — artificial variables remain in basis. Review constraints."}
    </div>""", unsafe_allow_html=True)

    st.markdown(f"<p style='font-weight:700;color:{TEXT_MUTED}!important;font-size:0.78rem;"
                f"text-transform:uppercase;letter-spacing:0.08em;'>Primal Solution</p>", unsafe_allow_html=True)
    pcols = st.columns(len(primal_sol)+1)
    for i,v in enumerate(primal_sol): pcols[i].metric(f"x{i+1}", f"{v:.4f}")
    pcols[-1].metric("Z (Primal)", f"{primal_obj:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-weight:700;color:{TEXT_MUTED}!important;font-size:0.78rem;"
                f"text-transform:uppercase;letter-spacing:0.08em;'>Dual Solution / Shadow Prices</p>",
                unsafe_allow_html=True)
    dcols = st.columns(len(sol_x)+1)
    for i,v in enumerate(sol_x): dcols[i].metric(f"y{i+1}", f"{v:.4f}")
    dcols[-1].metric("W (Dual)", f"{sol_obj:.4f}")
    st.markdown("<br>", unsafe_allow_html=True)

    analysis_lines = generate_analysis(c,A,b,ineq,opt,primal_sol,primal_obj,
                                        sol_x,sol_obj,method_used,feasible,
                                        dc,dA,db,dineq,dtype)

    tabs = st.tabs(["📐 Primal LP","🔀 Dual LP",
                    f"📋 {method_used} Iterations","📈 Charts","🧠 Analysis"])

    with tabs[0]:
        st.markdown(render_lp_formula(c,A,b,ineq,opt,is_dual=False), unsafe_allow_html=True)

    with tabs[1]:
        st.markdown(render_lp_formula(db,dA,dc,dineq,dtype,is_dual=True), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Variable":[f"y{i+1}" for i in range(len(sol_x))],
            "Value":np.round(sol_x,6),
            "Meaning":[f"Shadow price - Constraint {i+1}" for i in range(len(sol_x))]
        }), use_container_width=True)

    with tabs[2]:
        if pivot_records:
            mtc = ACCENT2 if method_used=="Big-M" else ACCENT3
            st.markdown(f"""<div style='margin-bottom:16px;'>
            <span style='background:{mtc}22;color:{mtc}!important;padding:4px 12px;border-radius:999px;
            font-size:0.78rem;font-weight:700;'>{method_used} - {len(pivot_records)} iteration(s)</span></div>""",
                        unsafe_allow_html=True)
            summary = pd.DataFrame([{"Iter":r["iteration"],"Method":r["method"],
                "Entering":r["entering"],"Leaving":r["leaving"],"Basis":", ".join(r["basis"])}
                for r in pivot_records])
            st.dataframe(summary, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📐 Full Tableau at Each Iteration", expanded=False):
                for rec in pivot_records:
                    st.markdown(f"""<div style='background:{SURFACE2};border-left:3px solid {ACCENT};
                    border-radius:0 8px 8px 0;padding:8px 14px;margin:8px 0 4px;'>
                    <span style='font-weight:700;color:{TEXT}!important;'>Iter {rec["iteration"]}</span>
                    &nbsp; Enter: <code>{rec["entering"]}</code>
                    &nbsp; Leave: <code>{rec["leaving"]}</code>
                    &nbsp; Basis: <code>{", ".join(rec["basis"])}</code></div>""", unsafe_allow_html=True)
                    st.dataframe(rec["tableau"], use_container_width=True)
        else:
            st.info("No pivot iterations — initial basis was already optimal or all RHS = 0.")

    with tabs[3]:
        fig = plot_results(c,A,b,ineq,opt,primal_sol,primal_obj,sol_x,sol_obj)
        st.pyplot(fig); plt.close(fig)
        if len(c) != 2:
            st.info("💡 Feasible region chart available for 2-variable problems only.")

    with tabs[4]:
        st.markdown(f"""<div style='background:{SURFACE};border:1px solid {BORDER};
        border-radius:14px;padding:24px 28px;'>
        <h3 style='margin-top:0;color:{TEXT}!important;'>🧠 Solution Analysis & Interpretation</h3>""",
                    unsafe_allow_html=True)
        pal = [ACCENT3, ACCENT, ACCENT2, WARN, ACCENT3, ACCENT]
        for i, line in enumerate(analysis_lines):
            display = re.sub(r'\*\*(.+?)\*\*', f"<b style='color:{TEXT}!important;'>\\1</b>", line)
            st.markdown(
                f"<div style='margin:10px 0;padding:12px 16px;background:{SURFACE2};"
                f"border-radius:10px;border-left:4px solid {pal[i%len(pal)]};'>"
                f"<p style='margin:0;color:{TEXT_MED}!important;font-size:0.9rem;line-height:1.7;'>"
                f"{display}</p></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    with st.spinner("Building PDF…"):
        pdf_buf = build_pdf(c,A,b,ineq,opt,dc,dA,db,dineq,dtype,
                            sol_x,sol_obj,pivot_records,primal_sol,primal_obj,
                            method_used,analysis_lines)
    csv_data = build_csv(primal_sol,primal_obj,sol_x,sol_obj)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("📄 Download PDF", data=pdf_buf,
                           file_name="duality_solution.pdf", mime="application/pdf",
                           use_container_width=True)
    with d2:
        st.download_button("📊 Download CSV", data=csv_data,
                           file_name="lp_results.csv", mime="text/csv",
                           use_container_width=True)

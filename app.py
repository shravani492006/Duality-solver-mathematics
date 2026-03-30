import streamlit as st
import numpy as np
from scipy.optimize import linprog
from PIL import Image, ImageEnhance, ImageFilter
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import cv2
import io
from fpdf import FPDF
import pytesseract

import shutil
tess_path = shutil.which("tesseract")
if tess_path:
    pytesseract.pytesseract.tesseract_cmd = tess_path

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Duality LP Solver", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# THEME
# ============================================================
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
    BTN_BG="#6FABFF"; BTN_TEXT="#FFFFFF"; BTN2_BG="#059669"
    CHART_COLORS=["#38BDF8","#A78BFA","#34D399","#FBBF24","#F87171","#FB923C"]
else:
    BG="#080D14"; SURFACE="#0F1923"; SURFACE2="#152030"; BORDER="#1E3A5F"
    ACCENT="#38BDF8"; ACCENT2="#A78BFA"; ACCENT3="#34D399"; WARN="#F87171"
    TEXT="#F0F6FF"; TEXT_MED="#CBD5E1"; TEXT_MUTED="#94A3B8"
    TAG_BG="#1E3A5F"; TAG_TEXT="#93C5FD"; SUCCESS_BG="#052E16"; ERROR_BG="#450A0A"
    BTN_BG="#38BDF8"; BTN_TEXT="#080D14"; BTN2_BG="#34D399"
    CHART_COLORS=["#38BDF8","#A78BFA","#34D399","#FBBF24","#F87171","#FB923C"]

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]{{background-color:{BG}!important;color:{TEXT}!important;font-family:'Space Grotesk',sans-serif!important;}}
[data-testid="stSidebar"]{{background-color:{SURFACE}!important;border-right:1px solid {BORDER};}}
[data-testid="stSidebar"] *{{color:{TEXT}!important;}}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p{{color:{TEXT_MED}!important;}}
h1,h2,h3,h4,h5,h6{{font-family:'Space Grotesk',sans-serif!important;color:{TEXT}!important;letter-spacing:-0.02em;}}
p,span,li,div,label{{color:{TEXT_MED}!important;}}
.stMarkdown p,.stMarkdown span,.stMarkdown li{{color:{TEXT_MED}!important;}}
code{{background:{SURFACE2}!important;color:{ACCENT}!important;font-family:'JetBrains Mono',monospace!important;padding:2px 6px;border-radius:4px;}}
.lp-card{{background:{SURFACE};border:1px solid {BORDER};border-radius:16px;padding:24px 28px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,0.15);}}
input,textarea,select{{background-color:{SURFACE2}!important;color:{TEXT}!important;border:1px solid {BORDER}!important;border-radius:10px!important;font-family:'Space Grotesk',sans-serif!important;}}
[data-baseweb="input"] input,[data-baseweb="textarea"] textarea,.stNumberInput input,.stTextArea textarea{{color:{TEXT}!important;background:{SURFACE2}!important;}}
[data-baseweb="select"] div{{background:{SURFACE2}!important;color:{TEXT}!important;border-color:{BORDER}!important;}}
[data-baseweb="popover"],[data-baseweb="menu"]{{background:{SURFACE}!important;border:1px solid {BORDER}!important;}}
[data-baseweb="option"]{{color:{TEXT}!important;background:{SURFACE}!important;}}
[data-baseweb="option"]:hover{{background:{SURFACE2}!important;}}
.stButton>button{{background:{BTN_BG}!important;color:{BTN_TEXT}!important;border:none!important;border-radius:10px!important;font-family:'Space Grotesk',sans-serif!important;font-weight:700!important;font-size:0.92rem!important;padding:11px 28px!important;transition:all 0.2s ease!important;}}
.stButton>button:hover{{filter:brightness(1.1);transform:translateY(-1px);box-shadow:0 6px 20px {ACCENT}44!important;}}
.stDownloadButton>button{{background:{ACCENT3}!important;color:#080D14!important;border:none!important;border-radius:10px!important;font-weight:700!important;width:100%;}}
[data-testid="stMetric"]{{background:{SURFACE}!important;border:1px solid {BORDER};border-top:3px solid {ACCENT};border-radius:14px;padding:14px 18px;}}
[data-testid="stMetricLabel"] p{{color:{TEXT_MUTED}!important;font-size:0.78rem!important;font-weight:600!important;text-transform:uppercase;letter-spacing:0.06em;}}
[data-testid="stMetricValue"]{{color:{TEXT}!important;font-size:1.6rem!important;font-weight:700!important;font-family:'JetBrains Mono',monospace!important;}}
.stTabs [data-baseweb="tab-list"]{{background:{SURFACE2}!important;border-radius:12px;padding:4px;gap:2px;border:1px solid {BORDER};}}
.stTabs [data-baseweb="tab"]{{background:transparent!important;color:{TEXT_MUTED}!important;border-radius:8px!important;border:none!important;font-weight:600!important;font-size:0.85rem!important;padding:8px 16px!important;transition:all 0.15s ease;}}
.stTabs [aria-selected="true"]{{background:{ACCENT}!important;color:{BTN_TEXT}!important;}}
[data-testid="stDataFrame"]{{border:1px solid {BORDER};border-radius:12px;overflow:hidden;}}
[data-testid="stDataFrame"] th{{background:{SURFACE2}!important;color:{TEXT}!important;font-weight:700!important;font-size:0.8rem!important;text-transform:uppercase;letter-spacing:0.05em;}}
[data-testid="stDataFrame"] td{{color:{TEXT_MED}!important;}}
.stSuccess{{background:{SUCCESS_BG}!important;border:1px solid {ACCENT3};border-radius:10px;}}
.stError{{background:{ERROR_BG}!important;border:1px solid {WARN};border-radius:10px;}}
.stInfo{{background:{SURFACE2}!important;border:1px solid {ACCENT};border-radius:10px;}}
.stSuccess p,.stError p,.stInfo p{{color:{TEXT}!important;}}
[data-testid="stExpander"]{{border:1px solid {BORDER}!important;border-radius:12px!important;background:{SURFACE}!important;}}
[data-testid="stExpander"] summary{{color:{TEXT}!important;font-weight:600!important;}}
[data-testid="stFileUploader"]{{background:{SURFACE2}!important;border:2px dashed {BORDER}!important;border-radius:14px!important;}}
[data-testid="stFileUploader"] span{{color:{TEXT_MED}!important;}}
.stSelectbox label,.stNumberInput label,.stTextArea label,.stFileUploader label{{color:{TEXT}!important;font-weight:600!important;font-size:0.85rem!important;}}
[data-testid="stRadio"] label{{color:{TEXT_MED}!important;}}
hr{{border-color:{BORDER}!important;}}
::-webkit-scrollbar{{width:6px;height:6px;}}
::-webkit-scrollbar-track{{background:{SURFACE};}}
::-webkit-scrollbar-thumb{{background:{BORDER};border-radius:3px;}}
::-webkit-scrollbar-thumb:hover{{background:{ACCENT};}}
</style>""", unsafe_allow_html=True)

# ---- Header ----
st.markdown(f"""
<div style="background:linear-gradient(135deg,{SURFACE} 0%,{SURFACE2} 100%);border:1px solid {BORDER};
border-radius:20px;padding:32px 36px 28px;margin-bottom:28px;position:relative;overflow:hidden;">
<div style="position:absolute;top:-40px;right:-40px;width:180px;height:180px;border-radius:50%;
background:radial-gradient(circle,{ACCENT}22,transparent 70%);"></div>
<span style="font-size:0.75rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
color:{ACCENT};font-family:'Space Grotesk',sans-serif;">Linear Programming</span>
<h1 style="font-size:2.4rem;font-weight:800;margin:6px 0 8px;color:{TEXT}!important;
letter-spacing:-0.03em;font-family:'Space Grotesk',sans-serif;">&#x2202; Duality LP Solver</h1>
<p style="color:{TEXT_MUTED}!important;font-size:1rem;margin:0;">
Primal &rarr; Dual &nbsp;&middot;&nbsp; Simplex &amp; Big-M Pivot Tables &nbsp;&middot;&nbsp; Feasible Region &nbsp;&middot;&nbsp; PDF Export</p>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"<hr style='border-color:{BORDER};margin:8px 0 16px;'>", unsafe_allow_html=True)
    st.markdown(f"""<div style='color:{TEXT_MUTED}!important;font-size:0.82rem;line-height:1.8;'>
This solver handles:<br>
&#x2022; <b style='color:{TEXT_MED}!important;'>Primal &rarr; Dual</b> conversion<br>
&#x2022; <b style='color:{TEXT_MED}!important;'>Simplex Method</b> (&le; constraints)<br>
&#x2022; <b style='color:{TEXT_MED}!important;'>Big-M Method</b> (&ge; / = constraints)<br>
&#x2022; <b style='color:{TEXT_MED}!important;'>Image OCR</b> (Tesseract)<br>
&#x2022; <b style='color:{TEXT_MED}!important;'>Feasible Region</b> visualization<br>
&#x2022; <b style='color:{TEXT_MED}!important;'>PDF &amp; CSV</b> export<br>
&#x2022; <b style='color:{TEXT_MED}!important;'>Solution Analysis</b>
</div>""", unsafe_allow_html=True)

# ============================================================
# BIG-M CONSTANT
# ============================================================
BIG_M = 1e4

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
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 128))
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) < 15:
            h, w = enhanced.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            enhanced = cv2.warpAffine(enhanced, M, (w, h),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    large = cv2.resize(binary, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(large, -1, kernel)
    return sharpened

def ocr_extract(image_np):
    preprocessed = preprocess_for_ocr(image_np)
    pil_img = Image.fromarray(preprocessed)
    configs = [
        "--psm 6 --oem 3",
        "--psm 4 --oem 3",
        "--psm 11 --oem 3",
    ]
    lp_kw = re.compile(
        r'(maximiz|minimiz|subject|s\.t\.|x\s*\d|y\s*\d|z\s*=|w\s*=|<=|>=|[≤≥]|\d+\s*[xy])',
        re.IGNORECASE
    )
    best_text = ""
    best_score = -1
    results_all = {}
    for cfg in configs:
        try:
            text = pytesseract.image_to_string(pil_img, config=cfg)
            score = len(lp_kw.findall(text))
            results_all[cfg] = text
            if score > best_score:
                best_score = score
                best_text = text
        except Exception:
            pass
    if not best_text:
        best_text = results_all.get(configs[0], "")
    return best_text, results_all

def normalize_ocr_lp(text):
    text = text.replace('|', '1').replace('O', '0')
    text = re.sub(r'\bI\b', '1', text)
    text = re.sub(r'(?<=[0-9])l(?=[^a-z])', '1', text)
    text = re.sub(r'[≤⩽]', '<=', text)
    text = re.sub(r'[≥⩾]', '>=', text)
    text = re.sub(r'[×✕]', 'x', text)
    text = re.sub(r'\bZ\s*[=:]\s*', 'Z = ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b([xy])\s+(\d)', r'\1\2', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d)\s+([xy]\d)', r'\1\2', text, flags=re.IGNORECASE)
    return text

# ============================================================
# LP PARSER  — FIX: supports both x and y variable names
# ============================================================
def clean_lp(text):
    text = text.lower()
    text = text.replace("≤","<=").replace("≥",">=").replace("⩽","<=").replace("⩾",">=")
    text = text.replace("×","x").replace("✕","x")
    text = re.sub(r'(?<![a-z])o(?=\d)','0',text)
    text = re.sub(r'(?<=\d)l(?!\d)','1',text)
    text = re.sub(r'\b(subject\s*to|s\.t\.)\b','\nST\n',text)
    text = re.sub(r'\b(maximize|maximise|max)\b','MAXIMIZE\n',text)
    text = re.sub(r'\b(minimize|minimise|min)\b','MINIMIZE\n',text)
    text = re.sub(r',(?!\d)',' ',text)
    # normalise spacing for both x and y variables
    text = re.sub(r'\b([xy])\s+(\d)', r'\1\2', text)
    text = re.sub(r'[ \t]+',' ',text)
    return text.strip()

def parse_lp(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    c, A, b, ineq, opt = [], [], [], [], "Maximize"
    n_vars, var_map = 0, {}

    def get_idx(v):
        nonlocal n_vars
        v = v.lower().replace(' ', '')
        if v not in var_map:
            var_map[v] = n_vars
            n_vars += 1
        return var_map[v]

    # FIX: regex now matches both x and y (and any single-letter variable followed by digits)
    term_re = re.compile(
        r'([+-]?)\s*(\d+(?:\.\d+)?)?\s*\*?\s*([a-zA-Z]\s*\d{1,2})',
        re.IGNORECASE
    )

    def parse_terms(expr):
        d = {}
        expr = re.sub(r'([+-])', r' \1 ', expr)
        expr = re.sub(r'\s+', ' ', expr).strip()
        for sign, coef, var in term_re.findall(expr):
            # skip purely alphabetic tokens that aren't variable references
            var_clean = var.replace(' ', '').lower()
            if not re.match(r'^[a-z]\d+$', var_clean):
                continue
            idx = get_idx(var_clean)
            coef = float(coef) if coef else 1.0
            sv = -1.0 if sign == '-' else 1.0
            d[idx] = d.get(idx, 0) + sv * coef
        return d

    obj_found = False
    for line in lines:
        line_l = line.lower()
        if "maximize" in line_l:
            opt = "Maximize"
        if "minimize" in line_l:
            opt = "Minimize"
        # objective: lines containing z= or w= or max/min keyword with variables
        if re.search(r'[zw]\s*=', line_l) or (
            ("max" in line_l or "min" in line_l) and term_re.search(line)
        ):
            obj_found = True
            rhs_of_eq = re.split(r'[zw]\s*=', line, flags=re.IGNORECASE)[-1]
            d = parse_terms(rhs_of_eq)
            for idx, val in d.items():
                while len(c) <= idx:
                    c.append(0.0)
                c[idx] = val
            continue

        m = re.search(r'(<=|>=|(?<![<>])=(?!=))', line)
        if m:
            sign = m.group(1)
            parts = re.split(r'<=|>=|(?<![<>])=(?!=)', line, maxsplit=1)
            if len(parts) != 2:
                continue
            lhs_d = parse_terms(parts[0])
            nums = re.findall(r'[+-]?\d+(?:\.\d+)?', parts[1].strip())
            if not nums:
                continue
            try:
                rhs_v = float(nums[0])
            except ValueError:
                continue
            A.append(lhs_d)
            b.append(rhs_v)
            ineq.append(sign)

    if not obj_found and not A:
        return [], [], [], [], opt

    if A:
        tv = max(n_vars,
                 max((max(d.keys(), default=-1) + 1) for d in A),
                 len(c))
        tv = max(tv, 1)
        while len(c) < tv:
            c.append(0.0)
        A_dense = []
        for row_d in A:
            row = [0.0] * tv
            for idx, val in row_d.items():
                if idx < tv:
                    row[idx] = val
            A_dense.append(row)
    else:
        tv = len(c)
        A_dense = []

    return c, A_dense, b, ineq, opt

def validate_lp(c, A, b, ineq):
    if not c:
        return False, "No objective function coefficients found."
    if not A:
        return False, "No constraints found."
    n = len(c)
    for i, row in enumerate(A):
        if len(row) != n:
            return False, f"Constraint {i+1} has {len(row)} coefficients but objective has {n}."
    if len(A) != len(b) or len(A) != len(ineq):
        return False, "Mismatch between constraint matrix, RHS, and inequality signs."
    return True, ""

# ============================================================
# PREPROCESSING
# ============================================================
def normalize_constraints(A, b, ineq):
    A_out, b_out, ineq_out = [], [], []
    for i in range(len(b)):
        if b[i] < 0:
            A_out.append([-v for v in A[i]])
            b_out.append(-b[i])
            flip = {"<=": ">=", ">=": "<=", "=": "="}
            ineq_out.append(flip[ineq[i]])
        else:
            A_out.append(list(A[i]))
            b_out.append(b[i])
            ineq_out.append(ineq[i])
    return A_out, b_out, ineq_out

# ============================================================
# DUAL GENERATOR
# ============================================================
def generate_dual(c, A, b, ineq, opt):
    At = np.array(A, dtype=float).T.tolist()
    dual_var_bounds = []
    for s in ineq:
        if s == "<=":
            dual_var_bounds.append((0, None))
        elif s == ">=":
            dual_var_bounds.append((None, 0))
        else:
            dual_var_bounds.append((None, None))
    dineq = [">=" if opt == "Maximize" else "<=" for _ in c]
    dtype = "Minimize" if opt == "Maximize" else "Maximize"
    return list(b), At, list(c), dineq, dtype, dual_var_bounds

# ============================================================
# SIMPLEX METHOD
# ============================================================
def simplex_iterations(c, A, b, max_iter=50):
    """
    Solves a maximization problem in standard form using the Simplex Method.
    Assumes constraints are of the form Ax <= b and x >= 0.
    """
    n_con, n_var = len(A), len(c)
    tableau = []

    # --- 1. INITIALIZE TABLEAU ---
    # Add slack variables to convert inequalities (<=) into equalities
    for i, row in enumerate(A):
        slack = [1.0 if j == i else 0.0 for j in range(n_con)]
        tableau.append(list(row) + slack + [b[i]])

    # Objective function row: Z - c1x1 - c2x2 ... = 0
    obj = [-ci for ci in c] + [0.0] * n_con + [0.0]
    tableau.append(obj)

    # Track which variables are currently in the basis (starting with slacks)
    basis = [f"s{i+1}" for i in range(n_con)]
    col_labels = ([f"x{i+1}" for i in range(n_var)] +
                  [f"s{i+1}" for i in range(n_con)] + ["RHS"])
    
    tableau = np.array(tableau, dtype=float)
    records = []

    # --- 2. ITERATION LOOP ---
    for iteration in range(max_iter):
        # Identify the "Entering Variable" (most negative coefficient in objective row)
        obj_row = tableau[-1, :-1]
        pivot_col = int(np.argmin(obj_row))

        # Optimality Check: If no negative coefficients, we've reached the maximum
        if obj_row[pivot_col] >= -1e-9:
            break

        # --- 3. MINIMUM RATIO TEST ---
        # Determine the "Leaving Variable" to maintain feasibility
        rhs = tableau[:-1, -1]
        col_val = tableau[:-1, pivot_col]
        
        # Avoid division by zero or negative values; only consider positive entries in pivot column
        ratios = np.where(col_val > 1e-9, rhs / col_val, np.inf)
        pivot_row = int(np.argmin(ratios))

        # If all ratios are infinity, the problem is unbounded
        if ratios[pivot_row] == np.inf:
            break

        # Log movement
        entering = col_labels[pivot_col]
        leaving = basis[pivot_row]

        # --- 4. PIVOTING (GAUSS-JORDAN ELIMINATION) ---
        # Normalize the pivot row so the pivot element becomes 1
        tableau[pivot_row] /= tableau[pivot_row, pivot_col]

        # Eliminate the entering variable from all other rows
        for i in range(len(tableau)):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]

        # Update the basis with the entering variable
        basis[pivot_row] = entering

        # Store a snapshot of the current state for visualization/reporting
        df = pd.DataFrame(
            tableau,
            columns=col_labels,
            index=[f"R{i+1}" for i in range(n_con)] + ["Z"]
        ).round(4)
        
        records.append({
            "iteration": iteration + 1,
            "entering": entering,
            "leaving": leaving,
            "basis": list(basis),
            "tableau": df,
            "method": "Simplex"
        })

    # --- 5. EXTRACT FINAL SOLUTION ---
    solution = [0.0] * n_var
    for i, bv in enumerate(basis):
        # If a decision variable (x_i) is in the basis, its value is in the RHS column
        m = re.match(r'x(\d+)', bv)
        if m:
            xi = int(m.group(1)) - 1
            if xi < n_var:
                solution[xi] = round(tableau[i, -1], 6)
    
    # Return all iteration steps, the final x values, and the max Z value
    return records, solution, round(tableau[-1, -1], 6)

# ============================================================
# BIG-M METHOD
# ============================================================
def bigm_iterations(c, A, b, ineq, opt, max_iter=80):
    n_vars, n_cons = len(c), len(A)
    M = BIG_M
    sign = -1.0 if opt == "Maximize" else 1.0
    c_arr = [sign * ci for ci in c]

    slack_map = {}
    artif_map = {}
    k_s = 0
    k_a = 0
    slack_types = []

    for i, iq in enumerate(ineq):
        if iq == "<=":
            slack_types.append(("slack", i))
            k_s += 1
        elif iq == ">=":
            slack_types.append(("surplus", i))
            k_s += 1
            k_a += 1
        else:
            k_a += 1

    n_slack_actual = sum(1 for iq in ineq if iq in ("<=", ">="))
    n_artif_actual = sum(1 for iq in ineq if iq in (">=", "="))
    total_cols = n_vars + n_slack_actual + n_artif_actual + 1

    col_labels = ([f"x{i+1}" for i in range(n_vars)] +
                  [f"s{i+1}" for i in range(n_slack_actual)] +
                  [f"a{i+1}" for i in range(n_artif_actual)] + ["RHS"])

    tableau = np.zeros((n_cons + 1, total_cols))
    k_s2 = 0
    k_a2 = 0

    for i, iq in enumerate(ineq):
        tableau[i, :n_vars] = A[i]
        tableau[i, -1] = b[i]
        if iq == "<=":
            col = n_vars + k_s2
            tableau[i, col] = 1.0
            slack_map[i] = col
            k_s2 += 1
        elif iq == ">=":
            col_s = n_vars + k_s2
            tableau[i, col_s] = -1.0
            slack_map[i] = col_s
            k_s2 += 1
            col_a = n_vars + n_slack_actual + k_a2
            tableau[i, col_a] = 1.0
            artif_map[i] = col_a
            k_a2 += 1
        else:
            col_a = n_vars + n_slack_actual + k_a2
            tableau[i, col_a] = 1.0
            artif_map[i] = col_a
            k_a2 += 1

    basis = [None] * n_cons
    for i, iq in enumerate(ineq):
        if iq == "<=":
            basis[i] = col_labels[slack_map[i]]
        else:
            basis[i] = col_labels[artif_map[i]]

    for j in range(n_vars):
        tableau[-1, j] = c_arr[j]
    for ac in artif_map.values():
        tableau[-1, ac] = M

    for row_i, iq in enumerate(ineq):
        if iq in (">=", "="):
            ac = artif_map[row_i]
            tableau[-1] -= tableau[-1, ac] * tableau[row_i]

    records = []
    for iteration in range(max_iter):
        obj_row = tableau[-1, :-1]
        pivot_col = int(np.argmin(obj_row))
        if obj_row[pivot_col] >= -1e-9:
            break
        rhs = tableau[:-1, -1]
        col_val = tableau[:-1, pivot_col]
        ratios = np.where(col_val > 1e-9, rhs / col_val, np.inf)
        pivot_row = int(np.argmin(ratios))
        if ratios[pivot_row] == np.inf:
            break
        entering = col_labels[pivot_col]
        leaving = basis[pivot_row]
        tableau[pivot_row] /= tableau[pivot_row, pivot_col]
        for i in range(len(tableau)):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]
        basis[pivot_row] = entering
        df = pd.DataFrame(
            tableau,
            columns=col_labels,
            index=[f"R{i+1}" for i in range(n_cons)] + ["Z"]
        ).round(4)
        records.append({
            "iteration": iteration + 1,
            "entering": entering,
            "leaving": leaving,
            "basis": list(basis),
            "tableau": df,
            "method": "Big-M"
        })

    feasible = all(
        not (bv and bv.startswith('a') and abs(tableau[i, -1]) > 1e-6)
        for i, bv in enumerate(basis)
    )
    solution = [0.0] * n_vars
    for i, bv in enumerate(basis):
        if bv:
            m = re.match(r'x(\d+)', bv)
            if m:
                xi = int(m.group(1)) - 1
                if xi < n_vars:
                    solution[xi] = round(tableau[i, -1], 6)
    raw_obj = round(tableau[-1, -1], 6)
    obj_val = -raw_obj if opt == "Maximize" else raw_obj
    return records, solution, obj_val, feasible

# ============================================================
# SMART SOLVER
# ============================================================
def solve_primal(c, A, b, ineq, opt):
    A, b, ineq = normalize_constraints(A, b, ineq)
    needs_bigm = any(iq in (">=", "=") for iq in ineq)
    if needs_bigm:
        records, solution, obj_val, feasible = bigm_iterations(c, A, b, ineq, opt)
        return records, solution, obj_val, "Big-M", feasible
    else:
        try:
            records, solution, obj_val = simplex_iterations(c, [list(r) for r in A], list(b))
            if opt == "Maximize":
                return records, solution, obj_val, "Simplex", True
            else:
                return records, solution, -obj_val, "Simplex", True
        except Exception as e:
            return [], [], 0.0, "Simplex", False

# ============================================================
# DUAL SOLVER
# ============================================================
def solve_dual(dc, dA, db, dineq, dtype, dual_var_bounds=None):
    sign_d = 1 if dtype == "Minimize" else -1
    dc_arr = [sign_d * v for v in dc]

    A_ub_d, b_ub_d = [], []
    A_eq_d, b_eq_d = [], []

    n_constraints = min(len(dA), len(dineq), len(db))
    if len(dA) != len(dineq) or len(dA) != len(db):
        st.warning("Dual dimension mismatch detected; using safe min-size to avoid crash.")

    for i in range(n_constraints):
        row = dA[i]
        if dineq[i] == "<=":
            A_ub_d.append(row)
            b_ub_d.append(db[i])
        elif dineq[i] == ">=":
            A_ub_d.append([-v for v in row])
            b_ub_d.append(-db[i])
        else:
            A_eq_d.append(row)
            b_eq_d.append(db[i])

    bounds = dual_var_bounds if dual_var_bounds is not None else [(0, None)] * len(dc)

    res = linprog(
        dc_arr,
        A_ub=A_ub_d or None,
        b_ub=b_ub_d or None,
        A_eq=A_eq_d or None,
        b_eq=b_eq_d or None,
        bounds=bounds,
        method="highs"
    )
    return res

# ============================================================
# OBSERVATION / ANALYSIS
# ============================================================
def generate_analysis(c, A, b, ineq, opt, primal_sol, primal_obj,
                       dual_sol, dual_obj, method_used, feasible,
                       dc, dA, db, dineq, dtype):
    n = len(c)
    m = len(A)
    lines = []

    if method_used == "Big-M":
        ge_count = sum(1 for iq in ineq if iq == ">=")
        eq_count = sum(1 for iq in ineq if iq == "=")
        lines.append(
            f"**Why Big-M?** Your problem has {ge_count} '>=' and {eq_count} '=' constraint(s). "
            "Standard Simplex only handles '<=' constraints with a natural slack variable basis. "
            "Big-M introduces artificial variables (a1, a2, ...) with a large penalty (M={:.0e}) "
            "to force them out of the solution - ensuring we start with a feasible basis and "
            "reach true optimality.".format(BIG_M)
        )
    else:
        lines.append(
            "**Why Simplex?** All constraints are '<=' type, allowing slack variables (s1, s2, ...) "
            "to form the initial feasible basis directly. The Simplex method pivots toward the "
            "optimal corner of the feasible polyhedron by selecting the most negative reduced cost "
            "column at each step."
        )

    active_vars = [(f"x{i+1}", round(primal_sol[i], 4)) for i in range(n) if abs(primal_sol[i]) > 1e-6]
    zero_vars = [f"x{i+1}" for i in range(n) if abs(primal_sol[i]) <= 1e-6]

    if active_vars:
        av_str = ", ".join(f"{v}={val}" for v, val in active_vars)
        lines.append(
            f"**Optimal Primal Solution:** {av_str}. "
            + (f"Variables {', '.join(zero_vars)} are non-basic (= 0) at optimality - "
               "they don't contribute to the objective." if zero_vars else
               "All variables are active at optimality.")
        )
    else:
        lines.append("**Primal Solution:** All variables are zero - the problem may be trivial or degenerate.")

    gap = abs(primal_obj - dual_obj)
    if gap < 1e-4 and feasible:
        lines.append(
            f"**Strong Duality holds:** Primal Z = {round(primal_obj,4)}, Dual W = {round(dual_obj,4)}. "
            "The fact that both objectives match confirms we've reached a global optimum - "
            "no better solution exists within the feasible region."
        )
    elif not feasible:
        lines.append(
            "WARNING: **Feasibility issue detected:** Artificial variables remained in the basis, "
            "suggesting the primal problem may be infeasible or unbounded. "
            "Review your constraint RHS values."
        )
    else:
        lines.append(
            f"**Note:** Primal Z = {round(primal_obj,4)}, Dual W = {round(dual_obj,4)} - "
            f"gap = {round(gap,6)}. Small numerical discrepancy due to floating-point arithmetic."
        )

    binding = []
    for i in range(m):
        lhs_val = sum(A[i][j] * primal_sol[j] for j in range(n))
        if abs(lhs_val - b[i]) < 1e-4:
            binding.append(f"C{i+1}")
    if binding:
        lines.append(
            f"**Binding Constraints:** {', '.join(binding)} are active (LHS = RHS at optimality). "
            "These define the corner point (vertex) of the feasible region where the optimum occurs. "
            "Relaxing these constraints could improve the objective."
        )

    nonzero_dual = [(f"y{i+1}", round(dual_sol[i], 4)) for i in range(len(dual_sol)) if abs(dual_sol[i]) > 1e-6]
    if nonzero_dual:
        shadow_str = ", ".join(f"{v}={val}" for v, val in nonzero_dual)
        lines.append(
            f"**Shadow Prices (Dual Variables):** {shadow_str}. "
            "Each dual variable yi represents the marginal value of relaxing constraint i by one unit - "
            "i.e., how much the optimal objective would improve if the RHS of that constraint increased by 1."
        )

    lines.append(
        f"**Objective ({opt}):** The problem seeks to {opt.lower()} "
        f"Z = {' + '.join(f'{c[i]}*x{i+1}' for i in range(n))}. "
        f"The optimal value Z* = {round(primal_obj,4)} is the {'maximum' if opt=='Maximize' else 'minimum'} "
        "achievable while satisfying all constraints simultaneously."
    )

    return lines

# ============================================================
# VISUALIZATION
# ============================================================
def plot_results(c, A, b, ineq, opt, primal_sol, primal_obj,
                  dual_sol, dual_obj, method_used):
    n_vars = len(c)
    bg_fig = SURFACE
    tc = TEXT_MED
    cc = CHART_COLORS

    if n_vars == 2:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    fig.patch.set_facecolor(bg_fig)
    for ax in axes:
        ax.set_facecolor(bg_fig)
        ax.tick_params(colors=tc, labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)

    ax1 = axes[0]
    var_names = [f"x{i+1}" for i in range(n_vars)]
    bars = ax1.bar(var_names, primal_sol, color=cc[:n_vars],
                   edgecolor='none', zorder=3, width=0.5, alpha=0.9)
    for bar, val in zip(bars, primal_sol):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(primal_sol) * 0.02 if max(primal_sol) > 0 else 0.1,
                 f'{val:.3f}', ha='center', va='bottom',
                 color=tc, fontsize=9, fontweight='bold')
    ax1.set_title("Primal Solution (x variables)", color=tc, fontsize=11,
                  fontweight='bold', pad=12)
    ax1.set_xlabel("Variable", color=tc, fontsize=9)
    ax1.set_ylabel("Value", color=tc, fontsize=9)
    ax1.yaxis.grid(True, color=BORDER, linestyle='--', alpha=0.5, zorder=0)
    ax1.set_axisbelow(True)
    ax1.axhline(0, color=BORDER, linewidth=0.8)

    ax2 = axes[1]
    vals = np.array(dual_sol)
    if vals.sum() > 1e-9:
        wedges, texts, autotexts = ax2.pie(
            np.abs(vals) + 1e-9,
            labels=[f"y{i+1}={v:.3f}" for i, v in enumerate(dual_sol)],
            colors=cc[:len(dual_sol)],
            autopct="%1.1f%%",
            textprops={"color": tc, "fontsize": 8},
            wedgeprops={"linewidth": 2, "edgecolor": bg_fig},
            startangle=90
        )
        for at in autotexts:
            at.set_color(bg_fig)
            at.set_fontweight('bold')
    else:
        ax2.text(0.5, 0.5, "All dual\nvariables = 0",
                 ha='center', va='center', color=tc, fontsize=11)
    ax2.set_title("Dual Variable Distribution (Shadow Prices)", color=tc,
                  fontsize=11, fontweight='bold', pad=12)

    if n_vars == 2:
        ax3 = axes[2]
        ax3.set_facecolor(bg_fig)
        ax3.tick_params(colors=tc, labelsize=9)
        for sp in ax3.spines.values():
            sp.set_edgecolor(BORDER)

        max_b = max(b) if b else 10
        x_max = max_b * 1.5
        x = np.linspace(0, x_max, 400)
        constraint_colors = [ACCENT3, WARN, ACCENT2, ACCENT, "#FBBF24", "#FB923C"]

        for i, (row, rhs, sign) in enumerate(zip(A, b, ineq)):
            a1, a2 = row[0], row[1]
            col = constraint_colors[i % len(constraint_colors)]
            label = f"C{i+1}: {a1}x1 + {a2}x2 {sign} {rhs}"
            if abs(a2) > 1e-9:
                y = (rhs - a1 * x) / a2
                mask = (y >= -x_max * 0.1) & (y <= x_max * 1.5)
                ax3.plot(x[mask], y[mask], color=col, linewidth=2, label=label, alpha=0.9)
            elif abs(a1) > 1e-9:
                xv = rhs / a1
                ax3.axvline(xv, color=col, linewidth=2, label=label, alpha=0.9)

        try:
            from scipy.spatial import ConvexHull
            pts = []
            test_x = np.linspace(0, x_max, 60)
            test_y = np.linspace(0, x_max, 60)
            for xi in test_x:
                for yi in test_y:
                    ok = True
                    for i, (row, rhs, sign) in enumerate(zip(A, b, ineq)):
                        val = row[0]*xi + row[1]*yi
                        if sign == "<=" and val > rhs + 1e-6:
                            ok = False; break
                        elif sign == ">=" and val < rhs - 1e-6:
                            ok = False; break
                        elif sign == "=" and abs(val - rhs) > 1e-4:
                            ok = False; break
                    if ok:
                        pts.append(np.array([xi, yi]))
            if len(pts) >= 3:
                pts_arr = np.array(pts)
                hull = ConvexHull(pts_arr)
                poly = Polygon(pts_arr[hull.vertices], closed=True,
                               facecolor=ACCENT + "30", edgecolor=ACCENT,
                               linewidth=1.5, linestyle='--', alpha=0.7)
                ax3.add_patch(poly)
        except Exception:
            pass

        ox, oy = primal_sol[0], primal_sol[1]
        ax3.scatter([ox], [oy], color=ACCENT, s=180, zorder=10,
                    edgecolors='white', linewidth=2, label=f"Optimal ({ox:.2f}, {oy:.2f})")
        ax3.annotate(f"Z*={round(primal_obj,2)}",
                     xy=(ox, oy), xytext=(ox + x_max*0.05, oy + x_max*0.05),
                     color=ACCENT, fontsize=9, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1.5))

        ax3.set_xlim(-x_max * 0.05, x_max)
        ax3.set_ylim(-x_max * 0.05, x_max)
        ax3.set_xlabel("x1", color=tc, fontsize=10)
        ax3.set_ylabel("x2", color=tc, fontsize=10)
        ax3.set_title("Feasible Region & Optimal Point", color=tc,
                      fontsize=11, fontweight='bold', pad=12)
        ax3.legend(fontsize=7, framealpha=0.3, facecolor=SURFACE2,
                   edgecolor=BORDER, labelcolor=tc, loc='upper right')
        ax3.yaxis.grid(True, color=BORDER, linestyle='--', alpha=0.3)
        ax3.xaxis.grid(True, color=BORDER, linestyle='--', alpha=0.3)
        ax3.set_axisbelow(True)

    plt.tight_layout(pad=2.5)
    return fig

# ============================================================
# PDF BUILDER
# ============================================================

def sanitize_pdf(text):
    text = str(text)
    replacements = {
        '\u2014': '-', '\u2013': '-', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u2026': '...', '\u00b7': '.',
        '\u2202': 'd', '\u2192': '->', '\u00b1': '+/-', '\u2265': '>=',
        '\u2264': '<=', '\u2260': '!=', '\u2211': 'sum', '\u221e': 'inf',
        '\u03b1': 'a', '\u03b2': 'b', '\u03b3': 'c', '\u03bb': 'lambda',
        '\u03bc': 'u', '\u03c3': 's',
        '\u2080': '0', '\u2081': '1', '\u2082': '2', '\u2083': '3',
        '\u2084': '4', '\u2085': '5', '\u2086': '6', '\u2087': '7',
        '\u2088': '8', '\u2089': '9',
        '\u00b9': '1', '\u00b2': '2', '\u00b3': '3',
        '\u2070': '0', '\u2074': '4', '\u2075': '5',
        '\u2076': '6', '\u2077': '7', '\u2078': '8', '\u2079': '9',
        '\u1d62': 'i', '\u2071': 'i', '\u1d65': 'v',
        '\u00d7': 'x', '\u00f7': '/', '\u2212': '-', '\u00a0': ' ',
        '\u2009': ' ', '\u200b': '', '\u2205': '{}', '\u2208': 'in',
        '\u2209': 'not in', '\u2227': 'and', '\u2228': 'or',
        '\u00ac': 'not', '\u2234': 'therefore', '\u2248': '~=',
        '\u221a': 'sqrt', '\u222b': 'integral', '\u2207': 'del',
        '\u2022': '*', '\u25cf': '*', '\u2714': 'v', '\u2718': 'x',
        '\u26a0': '!',
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    result = ''
    for char in text:
        try:
            char.encode('latin-1')
            result += char
        except (UnicodeEncodeError, UnicodeDecodeError):
            result += '?'
    return result


def build_pdf(primal_c, primal_A, primal_b, primal_ineq, primal_opt,
              dual_c, dual_A, dual_b, dual_ineq, dual_opt,
              sol_x, sol_obj, pivot_records,
              primal_sol, primal_obj, method_used,
              analysis_lines):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_fill_color(37, 99, 235)
    pdf.rect(0, 0, 210, 44, 'F')
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 15, "", ln=True)
    pdf.cell(0, 14, sanitize_pdf("  Duality LP Solver - Solution Report"), ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 8, sanitize_pdf(f"  Method: {method_used}  |  Primal Z = {round(primal_obj,4)}  |  Dual W = {round(sol_obj,4)}"), ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(8)

    def sec(title, r=37, g=99, b_=235):
        pdf.set_fill_color(r, g, b_)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 9, sanitize_pdf(f"  {title}"), ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    def lp_block(obj_str, constraints):
        pdf.set_font("Courier", "", 10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 7, sanitize_pdf(f"  {obj_str}"), ln=True)
        pdf.cell(0, 7, "  Subject to:", ln=True)
        for con in constraints:
            pdf.cell(0, 7, sanitize_pdf(f"    {con}"), ln=True)
        pdf.cell(0, 7, "  All variables >= 0", ln=True)
        pdf.ln(3)

    n = len(primal_c)
    sec("1. Primal Problem")
    lp_block(
        f"{primal_opt}  Z = " + " + ".join([f"{primal_c[i]}*x{i+1}" for i in range(n)]),
        [" + ".join([f"{primal_A[i][j]}*x{j+1}" for j in range(n)]) +
         f"  {primal_ineq[i]}  {primal_b[i]}" for i in range(len(primal_A))]
    )
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 7, sanitize_pdf(f"Optimal Z = {round(primal_obj,6)}  |  x = {[round(v,4) for v in primal_sol]}"), ln=True)
    pdf.ln(4)

    m = len(dual_c)
    sec("2. Dual Problem", 124, 58, 237)
    lp_block(
        f"{dual_opt}  W = " + " + ".join([f"{dual_c[i]}*y{i+1}" for i in range(m)]),
        [" + ".join([f"{dual_A[i][j]}*y{j+1}" for j in range(len(dual_A[i]))]) +
         f"  {dual_ineq[i]}  {dual_b[i]}" for i in range(len(dual_A))]
    )
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 7, sanitize_pdf(f"Optimal W = {round(sol_obj,6)}  |  y = {[round(v,4) for v in sol_x]}"), ln=True)
    pdf.ln(4)

    sec("3. Solution Analysis & Interpretation", 5, 46, 22)
    pdf.set_font("Arial", "", 9)
    for line in analysis_lines:
        clean_line = re.sub(r'\*\*(.+?)\*\*', r'\1', line)
        pdf.multi_cell(0, 6, sanitize_pdf(f"  {clean_line}"))
        pdf.ln(2)
    pdf.ln(4)

    if pivot_records:
        sec(f"4. {method_used} Pivot Table", 15, 23, 42)
        if method_used == "Big-M":
            pdf.set_font("Arial", "I", 9)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(0, 6,
                sanitize_pdf(f"  Big-M penalises artificial variables with M={BIG_M:.0e}. "
                "They must reach 0 for a feasible solution."), ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)

        pdf.set_font("Arial", "B", 9)
        cw = [18, 28, 28, 90, 20]
        for h_, w_ in zip(["Iter", "Entering", "Leaving", "New Basis", "Method"], cw):
            pdf.cell(w_, 7, h_, border=1, align="C")
        pdf.ln()
        pdf.set_font("Arial", "", 8)
        for rec in pivot_records:
            pdf.cell(cw[0], 6, str(rec["iteration"]), border=1, align="C")
            pdf.cell(cw[1], 6, sanitize_pdf(rec["entering"]), border=1, align="C")
            pdf.cell(cw[2], 6, sanitize_pdf(rec["leaving"]), border=1, align="C")
            pdf.cell(cw[3], 6, sanitize_pdf(", ".join(rec["basis"])[:40]), border=1)
            pdf.cell(cw[4], 6, sanitize_pdf(rec["method"]), border=1, align="C")
            pdf.ln()
        pdf.ln(4)

        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 8, "Detailed Tableau per Iteration:", ln=True)
        pdf.ln(2)
        for rec in pivot_records:
            if pdf.get_y() > 240:
                pdf.add_page()
            pdf.set_font("Arial", "B", 9)
            pdf.set_text_color(37, 99, 235)
            pdf.cell(0, 7,
                sanitize_pdf(f"  Iter {rec['iteration']}  Enter:{rec['entering']}  "
                f"Leave:{rec['leaving']}  Basis:{', '.join(rec['basis'])}"), ln=True)
            pdf.set_text_color(0, 0, 0)
            df = rec["tableau"]
            nc = len(df.columns)
            cw2 = min(20, int(190 // (nc + 1)))
            pdf.set_font("Arial", "B", 7)
            pdf.set_fill_color(220, 235, 255)
            pdf.cell(cw2, 5, "Row", border=1, align="C", fill=True)
            for col in df.columns:
                pdf.cell(cw2, 5, sanitize_pdf(str(col)[:6]), border=1, align="C", fill=True)
            pdf.ln()
            pdf.set_font("Courier", "", 6)
            for ridx, rrow in df.iterrows():
                pdf.cell(cw2, 5, sanitize_pdf(str(ridx)[:5]), border=1, align="C")
                for val in rrow:
                    pdf.cell(cw2, 5, f"{val:.2f}", border=1, align="R")
                pdf.ln()
            pdf.ln(3)

    # FIX: pdf.output() returns a bytearray in newer fpdf versions,
    # or a str in older ones. Handle both cases safely.
    raw = pdf.output()
    if isinstance(raw, (bytes, bytearray)):
        buffer = io.BytesIO(bytes(raw))
    else:
        buffer = io.BytesIO(raw.encode('latin-1'))
    buffer.seek(0)
    return buffer


def build_csv(primal_sol, primal_obj, dual_sol, dual_obj):
    rows = []
    for i, v in enumerate(primal_sol):
        rows.append({"Type": "Primal", "Variable": f"x{i+1}", "Value": round(v, 6)})
    rows.append({"Type": "Primal", "Variable": "Z (Objective)", "Value": round(primal_obj, 6)})
    for i, v in enumerate(dual_sol):
        rows.append({"Type": "Dual", "Variable": f"y{i+1}", "Value": round(v, 6)})
    rows.append({"Type": "Dual", "Variable": "W (Objective)", "Value": round(dual_obj, 6)})
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)

# ============================================================
# FORMULA RENDERER
# ============================================================
def render_lp_formula(c, A, b, ineq, opt, is_dual=False):
    vs = "y" if is_dual else "x"
    os_ = "W" if is_dual else "Z"
    n = len(c)

    def term(coef, var):
        coef = int(coef) if coef == int(coef) else coef
        return (f"<span style='color:{ACCENT};font-weight:700;"
                f"font-family:JetBrains Mono,monospace;'>{coef}</span>{var}")

    obj_terms = " + ".join([term(c[i], f"{vs}<sub>{i+1}</sub>") for i in range(n)])
    obj_html = (
        f"<span style='color:{ACCENT2};font-weight:700;'>{opt}</span> &nbsp;"
        f"<span style='color:{TEXT};font-weight:700;font-family:JetBrains Mono,monospace;'>"
        f"{os_} = </span>{obj_terms}"
    )
    con_rows = ""
    for i, row in enumerate(A):
        lhs = " + ".join([term(row[j], f"{vs}<sub>{j+1}</sub>") for j in range(len(row))])
        # FIX: safe int conversion
        try:
            rv = int(b[i]) if float(b[i]) == int(float(b[i])) else b[i]
        except (ValueError, OverflowError):
            rv = b[i]
        ic = ACCENT3 if ineq[i] == "<=" else (WARN if ineq[i] == ">=" else ACCENT2)
        con_rows += (
            f"<div style='margin:4px 0;padding:6px 12px;background:{SURFACE2};"
            f"border-radius:8px;border-left:3px solid {ic};'>"
            f"<span style='color:{TEXT_MUTED};font-size:0.75rem;'>C{i+1}&nbsp;</span>"
            f"{lhs}&nbsp;<span style='color:{ic};font-weight:700;'>{ineq[i]}</span>&nbsp;"
            f"<span style='color:{ACCENT};font-weight:700;"
            f"font-family:JetBrains Mono,monospace;'>{rv}</span></div>"
        )
    nn = ", ".join([f"{vs}<sub>{i+1}</sub>" for i in range(n)])
    return (
        f"<div style='background:{SURFACE};border:1px solid {BORDER};"
        f"border-radius:14px;padding:20px 24px;'>"
        f"<div style='margin-bottom:12px;font-size:1rem;'>{obj_html}</div>"
        f"<div style='color:{TEXT_MUTED};font-size:0.8rem;margin-bottom:8px;"
        f"font-weight:600;text-transform:uppercase;letter-spacing:0.06em;'>Subject to</div>"
        f"{con_rows}"
        f"<div style='margin-top:10px;color:{TEXT_MUTED};font-size:0.8rem;"
        f"font-style:italic;'>{nn} &ge; 0</div></div>"
    )

# ============================================================
# INPUT SECTION
# ============================================================
st.markdown("<div class='lp-card' style='padding-bottom:12px;'>", unsafe_allow_html=True)
st.markdown(f"<span style='font-size:1.1rem;font-weight:700;color:{TEXT}!important;'>&#x1F4E5; Input Mode</span>", unsafe_allow_html=True)
# FIX: label_visibility="collapsed" on all selectboxes with empty label
mode = st.selectbox("Input Mode", ["✏️ Manual Entry", "📝 Text / Equation", "🖼️ Image OCR", "📊 CSV Upload"],
                    label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='lp-card'>", unsafe_allow_html=True)

if mode == "✏️ Manual Entry":
    st.markdown(f"<h3 style='margin-top:0;color:{TEXT}!important;'>Manual Entry</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    num_vars = col1.number_input("Variables", 1, 10, 2)
    num_cons = col2.number_input("Constraints", 1, 10, 2)
    opt = col3.selectbox("Optimisation", ["Maximize", "Minimize"])
    st.markdown(f"<p style='font-weight:700;color:{TEXT}!important;margin-top:12px;'>Objective Coefficients</p>", unsafe_allow_html=True)
    c_cols = st.columns(int(num_vars))
    c = [c_cols[i].number_input(f"c(x{i+1})", value=0.0, key=f"c{i}", format="%.2f")
         for i in range(int(num_vars))]
    st.markdown(f"<p style='font-weight:700;color:{TEXT}!important;margin-top:12px;'>Constraint Matrix</p>", unsafe_allow_html=True)
    A, b, ineq = [], [], []
    for i in range(int(num_cons)):
        cols = st.columns(int(num_vars) + 2)
        row = [cols[j].number_input(f"a{i+1},{j+1}", value=0.0, key=f"a{i}_{j}", format="%.2f")
               for j in range(int(num_vars))]
        # FIX: label_visibility="collapsed" for the sign selectbox
        sign = cols[-2].selectbox("Sign", ["<=", ">=", "="], key=f"s{i}",
                                   label_visibility="collapsed")
        rhs = cols[-1].number_input("RHS", value=0.0, key=f"b{i}", format="%.2f",
                                     label_visibility="collapsed")
        A.append(row); b.append(rhs); ineq.append(sign)
    if st.button("💾 Save Problem", use_container_width=True):
        ok, err = validate_lp(c, A, b, ineq)
        if ok:
            st.session_state.data = (c, A, b, ineq, opt)
            st.success("✓ Problem saved — click Solve below.")
        else:
            st.error(f"Validation error: {err}")

elif mode == "📝 Text / Equation":
    st.markdown(f"<h3 style='margin-top:0;color:{TEXT}!important;'>Text / Equation Input</h3>", unsafe_allow_html=True)
    st.markdown(f"""<div style='background:{SURFACE2};border:1px solid {BORDER};border-radius:10px;
    padding:14px 18px;margin-bottom:14px;font-family:JetBrains Mono,monospace;font-size:0.85rem;color:{TEXT_MED}!important;'>
    <span style='color:{ACCENT};font-weight:700;'>Maximize</span> Z = 5x1 + 4x2<br>
    6x1 + 4x2 &lt;= 24<br>x1 + 2x2 &lt;= 6<br><br>
    <span style='color:{ACCENT};font-weight:700;'>Minimize</span> Z = 2x1 + 3x2<br>
    x1 + x2 >= 4<br>x1 - x2 = 1</div>""", unsafe_allow_html=True)
    text_input = st.text_area("LP Problem", height=160, placeholder="Type your LP problem here…")
    if st.button("🔍 Parse & Save", use_container_width=True):
        cleaned = clean_lp(text_input)
        result = parse_lp(cleaned)
        c, A, b, ineq, opt = result
        ok, err = validate_lp(c, A, b, ineq)
        if ok:
            st.session_state.data = (c, A, b, ineq, opt)
            nb = any(iq in (">=", "=") for iq in ineq)
            st.success(f"✓ Parsed: {len(c)} vars, {len(A)} constraints — will use **{'Big-M' if nb else 'Simplex'}** method")
        else:
            st.error(f"Parse/validation error: {err}")

elif mode == "🖼️ Image OCR":
    st.markdown(f"<h3 style='margin-top:0;color:{TEXT}!important;'>Image OCR (Tesseract)</h3>", unsafe_allow_html=True)
    st.markdown(f"""<div style='background:{SURFACE2};border:1px dashed {BORDER};border-radius:10px;
    padding:12px 16px;margin-bottom:14px;font-size:0.83rem;color:{TEXT_MUTED}!important;'>
    💡 <b style='color:{TEXT_MED}!important;'>Tips for best results:</b><br>
    • Use clear, high-contrast images with printed (not handwritten) text<br>
    • Ensure LP notation: <code>x1, x2</code>, <code>&lt;=</code>, <code>&gt;=</code><br>
    • Write objective as: <code>Maximize Z = 5x1 + 4x2</code><br>
    • Edit extracted text if needed before parsing</div>""", unsafe_allow_html=True)

    tess_ok = get_tesseract_available()
    if not tess_ok:
        st.warning("⚠️ Tesseract not found. Please install Tesseract-OCR and ensure it's in your PATH.")

    file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "bmp", "webp"])
    if file and tess_ok:
        image = Image.open(file).convert("RGB")
        img_np = np.array(image)
        col_a, col_b = st.columns(2)
        col_a.image(image, caption="Original", use_container_width=True)
        preprocessed = preprocess_for_ocr(img_np)
        col_b.image(preprocessed, caption="Preprocessed (for OCR)", use_container_width=True, clamp=True)

        with st.spinner("Running Tesseract OCR…"):
            best_text, all_results = ocr_extract(img_np)
            best_text = normalize_ocr_lp(best_text)

        st.markdown(f"<p style='font-weight:700;color:{TEXT}!important;'>Extracted Text (editable)</p>", unsafe_allow_html=True)
        extracted = st.text_area("OCR Output", best_text, height=200, key="ocr_out")

        with st.expander("🔍 Raw results from each OCR config"):
            for cfg, txt in all_results.items():
                st.markdown(f"**Config `{cfg}`:**")
                st.code(txt)

        col_opt2, col_btn2 = st.columns([2, 3])
        opt_ocr = col_opt2.selectbox("Optimisation Type", ["Maximize", "Minimize"], key="ocr_opt")
        if col_btn2.button("🔍 Parse OCR Output", use_container_width=True):
            cleaned = clean_lp(extracted)
            result = parse_lp(cleaned)
            c, A, b, ineq, opt = result
            if not opt_ocr == opt:
                opt = opt_ocr
            ok, err = validate_lp(c, A, b, ineq)
            if ok:
                st.session_state.data = (c, A, b, ineq, opt)
                nb = any(iq in (">=", "=") for iq in ineq)
                st.success(f"✓ Parsed: {len(c)} vars, {len(A)} constraints — **{'Big-M' if nb else 'Simplex'}** method")
                with st.expander("🔎 Parsed Values"):
                    cc2, cc3 = st.columns(2)
                    cc2.markdown(f"**c:** `{c}`")
                    cc2.markdown(f"**b:** `{b}`")
                    cc3.markdown(f"**A:** `{A}`")
                    cc3.markdown(f"**ineq:** `{ineq}`")
            else:
                st.error(f"Parser error: {err}\n\nPlease edit the extracted text above and try again.")

elif mode == "📊 CSV Upload":
    st.markdown(f"<h3 style='margin-top:0;color:{TEXT}!important;'>CSV Upload</h3>", unsafe_allow_html=True)
    st.markdown(f"""<div style='background:{SURFACE2};border:1px solid {BORDER};border-radius:10px;
    padding:12px 16px;margin-bottom:14px;font-size:0.83rem;color:{TEXT_MUTED}!important;'>
    Format: columns = <code>x1, x2, ..., xn, RHS</code>. One row per constraint.</div>""", unsafe_allow_html=True)
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.dataframe(df, use_container_width=True)
        n_var = df.shape[1] - 1
        c_csv_cols = st.columns(min(n_var, 5))
        c_csv = [c_csv_cols[i % min(n_var, 5)].number_input(f"Obj x{i+1}", value=1.0, key=f"ccsv{i}")
                 for i in range(n_var)]
        col_o, col_b2 = st.columns(2)
        opt_csv = col_o.selectbox("Optimisation", ["Maximize", "Minimize"], key="csv_opt")
        sign_csv = col_b2.selectbox("Constraint type (all)", ["<=", ">=", "="], key="csv_sign")
        if st.button("📥 Load CSV", use_container_width=True):
            A = df.iloc[:, :-1].values.tolist()
            b = df.iloc[:, -1].values.tolist()
            ineq = [sign_csv] * len(A)
            ok, err = validate_lp(c_csv, A, b, ineq)
            if ok:
                st.session_state.data = (c_csv, A, b, ineq, opt_csv)
                st.success("✓ CSV loaded.")
            else:
                st.error(f"Validation error: {err}")

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PROBLEM PREVIEW
# ============================================================
if "data" in st.session_state:
    c_p, A_p, b_p, ineq_p, opt_p = st.session_state.data
    needs_bigm = any(iq in (">=", "=") for iq in ineq_p)
    ml = "Big-M Method" if needs_bigm else "Simplex Method"
    tc_tag = ACCENT2 if needs_bigm else ACCENT3
    st.markdown(f"""<div style='display:flex;align-items:center;gap:8px;margin:14px 0 8px;'>
    <span style='font-size:1.1rem;font-weight:700;color:{TEXT}!important;'>Problem Preview</span>
    <span style='background:{TAG_BG};color:{TAG_TEXT}!important;padding:3px 10px;border-radius:999px;
    font-size:0.72rem;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;'>Primal</span>
    <span style='background:{tc_tag}22;color:{tc_tag}!important;padding:3px 10px;border-radius:999px;
    font-size:0.72rem;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;'>{ml}</span>
    </div>""", unsafe_allow_html=True)
    st.markdown(render_lp_formula(c_p, A_p, b_p, ineq_p, opt_p), unsafe_allow_html=True)

# ============================================================
# SOLVE BUTTON
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
solve_col = st.columns([1, 2, 1])[1]
with solve_col:
    solve_clicked = st.button("🚀 Solve Problem", use_container_width=True)

if solve_clicked:
    if "data" not in st.session_state:
        st.error("Please enter/parse a problem first.")
        st.stop()

    c, A, b, ineq, opt = st.session_state.data
    ok, err = validate_lp(c, A, b, ineq)
    if not ok:
        st.error(f"Problem data invalid: {err}")
        st.stop()

    with st.spinner("Solving primal…"):
        pivot_records, primal_sol, primal_obj, method_used, feasible = solve_primal(c, A, b, ineq, opt)

    dc, dA, db, dineq, dtype, d_bounds = generate_dual(c, A, b, ineq, opt)
    with st.spinner("Solving dual…"):
        res = solve_dual(dc, dA, db, dineq, dtype, dual_var_bounds=d_bounds)

    if not res.success:
        st.error(f"Dual solver failed: {res.message}")
        st.stop()

    sol_x = res.x
    sol_obj = -res.fun if dtype == "Maximize" else res.fun

    st.markdown(f"""<div style='margin:28px 0 18px;'>
    <span style='font-size:0.75rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:{ACCENT};'>Results</span>
    <h2 style='margin:4px 0 0;color:{TEXT}!important;font-size:1.8rem;font-weight:800;'>Solution Report</h2>
    </div>""", unsafe_allow_html=True)

    sc = ACCENT3 if feasible else WARN
    st.markdown(f"""<div style='background:{sc}22;border:1px solid {sc};border-radius:12px;
    padding:12px 20px;margin-bottom:20px;color:{sc}!important;font-weight:600;'>
    {"✓ Optimal solution found using " + method_used + " Method" if feasible else
     "⚠ Problem may be infeasible — artificial variables remain in the basis. Review your constraints."}
    </div>""", unsafe_allow_html=True)

    st.markdown(f"<p style='font-weight:700;color:{TEXT_MUTED}!important;font-size:0.78rem;"
                f"text-transform:uppercase;letter-spacing:0.08em;'>Primal Solution (x variables)</p>",
                unsafe_allow_html=True)
    pcols = st.columns(len(primal_sol) + 1)
    for i, v in enumerate(primal_sol):
        pcols[i].metric(f"x{i+1}", f"{v:.4f}")
    pcols[-1].metric("Primal Z", f"{primal_obj:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"<p style='font-weight:700;color:{TEXT_MUTED}!important;font-size:0.78rem;"
                f"text-transform:uppercase;letter-spacing:0.08em;'>Dual Solution (y variables / Shadow Prices)</p>",
                unsafe_allow_html=True)
    dcols_m = st.columns(len(sol_x) + 1)
    for i, v in enumerate(sol_x):
        dcols_m[i].metric(f"y{i+1}", f"{v:.4f}")
    dcols_m[-1].metric("Dual W", f"{sol_obj:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)

    analysis_lines = generate_analysis(
        c, A, b, ineq, opt,
        primal_sol, primal_obj,
        sol_x, sol_obj,
        method_used, feasible,
        dc, dA, db, dineq, dtype
    )

    tabs = st.tabs(["📐 Primal LP", "🔀 Dual LP",
                    f"📋 {method_used} Iterations", "📈 Charts", "🧠 Analysis"])

    with tabs[0]:
        st.markdown(render_lp_formula(c, A, b, ineq, opt, is_dual=False), unsafe_allow_html=True)

    with tabs[1]:
        # FIX: was render_lp_formula(db, dA, dc, ...) — wrong arg order; should be (dc, dA, db, ...)
        st.markdown(render_lp_formula(dc, dA, db, dineq, dtype, is_dual=True), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame({"Variable": [f"y{i+1}" for i in range(len(sol_x))],
                          "Value": np.round(sol_x, 6),
                          "Interpretation": ["Shadow price of constraint " + str(i+1)
                                             for i in range(len(sol_x))]}),
            use_container_width=True
        )

    with tabs[2]:
        if pivot_records:
            mtc = ACCENT2 if method_used == "Big-M" else ACCENT3
            st.markdown(f"""<div style='margin-bottom:16px;'>
            <span style='background:{mtc}22;color:{mtc}!important;padding:4px 12px;border-radius:999px;
            font-size:0.78rem;font-weight:700;'>{method_used} Method — {len(pivot_records)} iteration(s)</span></div>""",
                        unsafe_allow_html=True)

            if method_used == "Big-M":
                st.markdown(f"""<div style='background:{SURFACE2};border:1px solid {BORDER};border-radius:10px;
                padding:12px 16px;margin-bottom:14px;font-size:0.83rem;color:{TEXT_MED}!important;'>
                <b style='color:{TEXT}!important;'>Big-M Method:</b> Artificial variables (a1, a2, ...) are added
                to >= and = constraints with a large penalty M={BIG_M:.0e}.
                Surplus variables (s) handle >= constraints.
                The artificials are driven to zero to achieve feasibility.</div>""", unsafe_allow_html=True)

            summary = pd.DataFrame([{
                "Iter": r["iteration"],
                "Method": r["method"],
                "Entering": r["entering"],
                "Leaving": r["leaving"],
                "Basis": ", ".join(r["basis"])
            } for r in pivot_records])
            st.dataframe(summary, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📐 Full Tableau at Each Iteration", expanded=False):
                for rec in pivot_records:
                    st.markdown(f"""<div style='background:{SURFACE2};border-left:3px solid {ACCENT};
                    border-radius:0 8px 8px 0;padding:8px 14px;margin:8px 0 4px;'>
                    <span style='font-weight:700;color:{TEXT}!important;'>Iteration {rec["iteration"]}</span>
                    &nbsp; Enter: <code>{rec["entering"]}</code>
                    &nbsp; Leave: <code>{rec["leaving"]}</code>
                    &nbsp; Basis: <code>{", ".join(rec["basis"])}</code></div>""", unsafe_allow_html=True)
                    st.dataframe(rec["tableau"], use_container_width=True)
        else:
            st.info("No pivot iterations recorded — the initial basis was already optimal, "
                    "or all RHS values were zero.")

    with tabs[3]:
        fig = plot_results(c, A, b, ineq, opt, primal_sol, primal_obj,
                           sol_x, sol_obj, method_used)
        st.pyplot(fig)
        plt.close(fig)
        if len(c) != 2:
            st.info("💡 Feasible region plot is only available for 2-variable problems.")

    with tabs[4]:
        st.markdown(f"""<div style='background:{SURFACE};border:1px solid {BORDER};
        border-radius:14px;padding:24px 28px;'>
        <h3 style='margin-top:0;color:{TEXT}!important;'>🧠 Solution Analysis & Interpretation</h3>""",
                    unsafe_allow_html=True)
        for i, line in enumerate(analysis_lines):
            display = re.sub(r'\*\*(.+?)\*\*',
                             f"<b style='color:{TEXT}!important;'>\\1</b>", line)
            color = [ACCENT3, ACCENT, ACCENT2, WARN, ACCENT3, ACCENT][i % 6]
            st.markdown(
                f"<div style='margin:10px 0;padding:12px 16px;background:{SURFACE2};"
                f"border-radius:10px;border-left:4px solid {color};'>"
                f"<p style='margin:0;color:{TEXT_MED}!important;font-size:0.9rem;"
                f"line-height:1.7;'>{display}</p></div>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<div style='text-align:center;margin-bottom:12px;'>"
                f"<span style='font-size:0.85rem;color:{TEXT_MUTED}!important;'>"
                f"Export the full solution including pivot tables, dual formulation, and analysis.</span></div>",
                unsafe_allow_html=True)

    with st.spinner("Building PDF…"):
        pdf_buffer = build_pdf(
            c, A, b, ineq, opt,
            dc, dA, db, dineq, dtype,
            sol_x, sol_obj, pivot_records,
            primal_sol, primal_obj, method_used,
            analysis_lines
        )

    csv_data = build_csv(primal_sol, primal_obj, sol_x, sol_obj)

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="📄 Download Full Solution PDF",
            data=pdf_buffer,
            file_name="duality_solution.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    with dl_col2:
        st.download_button(
            label="📊 Download Results CSV",
            data=csv_data,
            file_name="lp_results.csv",
            mime="text/csv",
            use_container_width=True
        )

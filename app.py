"""
app.py  –  Iris Disease Detection System
Streamlit prototype for detecting Healthy / Glaucoma / Myopia iris conditions.

Pages:
  1. Home            – system overview & dataset stats
  2. Disease Detection – upload image → diagnose with chosen model
  3. Model Evaluation  – per-model accuracy / FAR / FRR / EER / ROC
  4. Model Comparison  – side-by-side table + charts for all models
  5. Train Models      – launch training script
  6. About
"""

import os, json, pickle, subprocess, time, sys
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image
from io import BytesIO

# ── Page config (must be the very first Streamlit call) ────────────────────
st.set_page_config(
    page_title="Iris Disease Detection",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Path constants (always absolute, regardless of CWD) ────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
METRICS_PATH     = os.path.join(SAVED_MODELS_DIR, "metrics.json")
UPLOAD_DIR       = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

CLASSES = ['Healthy', 'Glaucoma', 'Myopia']
CLASS_COLORS = {
    'Healthy':  '#2E7D32',
    'Glaucoma': '#C62828',
    'Myopia':   '#1565C0',
}
CLASS_ICONS = {
    'Healthy':  '✅',
    'Glaucoma': '⚠️',
    'Myopia':   '🔵',
}

MODEL_PATHS = {
    'ResNet50':    os.path.join(SAVED_MODELS_DIR, 'resnet50_iris.h5'),
    'SVM (Gabor)': os.path.join(SAVED_MODELS_DIR, 'svm_gabor.pkl'),
    'RF (Wavelet)':os.path.join(SAVED_MODELS_DIR, 'rf_wavelet.pkl'),
}

DATA_PATHS = {
    'Healthy':  os.path.join(BASE_DIR, 'disease_model', 'Healthy-20260312T025031Z-3-001', 'Healthy'),
    'Glaucoma': os.path.join(BASE_DIR, 'disease_model', 'Glaucoma-20260312T025028Z-3-001', 'Glaucoma'),
    'Myopia':   os.path.join(BASE_DIR, 'disease_model', 'Myopia-20260312T025034Z-3-001', 'Myopia'),
}

IMG_SIZE_DL  = (224, 224)
IMG_SIZE_CNN = (64, 64)
IMG_EXTS     = {'.jpg', '.jpeg', '.png', '.bmp'}

DISEASE_INFO = {
    'Healthy': {
        'description': (
            'No iris abnormalities detected. The iris appears consistent with a healthy eye.'
        ),
        'symptoms': [
            'Clear, detailed iris texture',
            'Normal pupillary light response',
            'No visible lesions or discolouration',
        ],
        'recommendation': (
            'Continue routine annual eye examinations to maintain ocular health.'
        ),
        'icon': '✅',
        'color': '#2E7D32',
        'bg': '#E8F5E9',
    },
    'Glaucoma': {
        'description': (
            'Glaucoma is a group of eye conditions that damage the optic nerve, '
            'often linked to elevated intraocular pressure. It is a leading cause '
            'of irreversible blindness if left untreated.'
        ),
        'symptoms': [
            'Gradual peripheral (tunnel) vision loss',
            'Halos or starbursts around lights',
            'Eye redness, pain, or headache (acute type)',
            'Blurred or hazy vision',
        ],
        'recommendation': (
            'Consult an ophthalmologist promptly. '
            'Early intervention with eye drops, laser therapy, or surgery '
            'can halt further optic nerve damage.'
        ),
        'icon': '⚠️',
        'color': '#C62828',
        'bg': '#FFEBEE',
    },
    'Myopia': {
        'description': (
            'Myopia (nearsightedness) is a refractive error in which the eye '
            'focuses light in front of the retina, making distant objects blurry. '
            'It is the most common refractive disorder worldwide.'
        ),
        'symptoms': [
            'Difficulty seeing distant objects clearly',
            'Squinting to improve focus',
            'Eye strain, fatigue, or headaches',
            'Reduced visibility when driving at night',
        ],
        'recommendation': (
            'Corrective glasses or contact lenses provide immediate relief. '
            'Refractive surgery (LASIK/PRK) may offer a permanent solution. '
            'Schedule a comprehensive eye examination to confirm the prescription.'
        ),
        'icon': '🔵',
        'color': '#1565C0',
        'bg': '#E3F2FD',
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# CSS – modern design system
# ═══════════════════════════════════════════════════════════════════════════
def apply_custom_css():
    st.markdown("""
    <style>
    /* ── Tokens ─────────────────────────────────────────── */
    :root{
        --blue-900:#0D47A1;--blue-700:#1565C0;--blue-500:#1976D2;
        --blue-100:#E3F2FD;--blue-50:#F0F7FF;
        --green-800:#1B5E20;--green-700:#2E7D32;--green-100:#E8F5E9;
        --red-800:#B71C1C;--red-700:#C62828;--red-100:#FFEBEE;
        --amber-700:#F9A825;--amber-50:#FFFDE7;
        --grey-800:#263238;--grey-600:#546E7A;--grey-300:#B0BEC5;
        --grey-100:#F5F7FA;--grey-50:#FAFBFC;
        --radius-sm:6px;--radius-md:10px;--radius-lg:16px;
        --shadow-sm:0 1px 4px rgba(0,0,0,.08);
        --shadow-md:0 4px 16px rgba(0,0,0,.10);
        --shadow-lg:0 8px 32px rgba(0,0,0,.12);
        --transition:.18s ease;
    }

    /* ── Global body tweaks ─────────────────────────────── */
    .stApp{background:#F5F7FA}
    section[data-testid="stSidebar"]{background:#0D47A1 !important}
    section[data-testid="stSidebar"] *{color:#E3F2FD !important}
    section[data-testid="stSidebar"] .stRadio label{
        background:rgba(255,255,255,.08);border-radius:var(--radius-sm);
        padding:.45rem .8rem;margin:.15rem 0;display:block;
        transition:background var(--transition);cursor:pointer;
    }
    section[data-testid="stSidebar"] .stRadio label:hover{
        background:rgba(255,255,255,.18);
    }

    /* ── Page headers ───────────────────────────────────── */
    .main-header{
        font-size:2.1rem;font-weight:800;
        background:linear-gradient(135deg,#1565C0 0%,#0D47A1 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;text-align:center;margin-bottom:.3rem;
        letter-spacing:-.5px;
    }
    .page-sub{
        font-size:1.05rem;color:var(--grey-600);
        text-align:center;margin-bottom:1.8rem;
    }
    .section-label{
        font-size:.72rem;font-weight:700;letter-spacing:1.1px;
        text-transform:uppercase;color:var(--grey-600);
        margin:1.2rem 0 .5rem;
    }

    /* ── Cards ──────────────────────────────────────────── */
    .card{
        border-radius:var(--radius-md);padding:1.4rem 1.6rem;
        background:#fff;box-shadow:var(--shadow-sm);
        margin-bottom:1rem;border:1px solid rgba(0,0,0,.04);
        transition:box-shadow var(--transition);
    }
    .card:hover{box-shadow:var(--shadow-md)}

    .kpi-card{
        border-radius:var(--radius-md);padding:1.2rem 1.4rem;
        background:#fff;box-shadow:var(--shadow-sm);
        border:1px solid rgba(0,0,0,.04);text-align:center;
        transition:transform var(--transition),box-shadow var(--transition);
    }
    .kpi-card:hover{transform:translateY(-3px);box-shadow:var(--shadow-md)}
    .kpi-number{font-size:2.2rem;font-weight:800;line-height:1.1;margin:.3rem 0}
    .kpi-label{font-size:.78rem;font-weight:600;letter-spacing:.7px;
               text-transform:uppercase;color:var(--grey-600)}

    .hero-band{
        background:linear-gradient(135deg,#0D47A1 0%,#1565C0 55%,#1976D2 100%);
        border-radius:var(--radius-lg);padding:2.2rem 2.5rem;margin-bottom:1.6rem;
        color:#fff;position:relative;overflow:hidden;
    }
    .hero-band::after{
        content:'👁';position:absolute;right:2rem;top:50%;
        transform:translateY(-50%);font-size:5rem;opacity:.12;
    }
    .hero-title{font-size:2rem;font-weight:800;margin:0 0 .4rem;letter-spacing:-.5px}
    .hero-sub{font-size:1rem;opacity:.85;margin:0}

    /* ── Result banners ─────────────────────────────────── */
    .result-healthy{
        background:var(--green-100);border-left:5px solid var(--green-700);
        padding:1.1rem 1.5rem;border-radius:var(--radius-md);
        font-size:1.2rem;font-weight:700;margin:.6rem 0;
        display:flex;align-items:center;gap:.6rem;
    }
    .result-glaucoma{
        background:var(--red-100);border-left:5px solid var(--red-700);
        padding:1.1rem 1.5rem;border-radius:var(--radius-md);
        font-size:1.2rem;font-weight:700;margin:.6rem 0;
        display:flex;align-items:center;gap:.6rem;
    }
    .result-myopia{
        background:var(--blue-100);border-left:5px solid var(--blue-700);
        padding:1.1rem 1.5rem;border-radius:var(--radius-md);
        font-size:1.2rem;font-weight:700;margin:.6rem 0;
        display:flex;align-items:center;gap:.6rem;
    }

    /* ── Status / badge pills ───────────────────────────── */
    .badge{display:inline-flex;align-items:center;gap:.3rem;
           border-radius:20px;padding:.25rem .75rem;font-size:.78rem;font-weight:700}
    .badge-green{background:#C8E6C9;color:#1B5E20}
    .badge-red{background:#FFCDD2;color:#B71C1C}
    .badge-blue{background:#BBDEFB;color:#0D47A1}
    .badge-amber{background:#FFF9C4;color:#F57F17}
    .badge-grey{background:#ECEFF1;color:#455A64}

    /* ── Metric tiles (evaluation page) ─────────────────── */
    .metric-tile{
        background:#fff;border-radius:var(--radius-md);
        padding:1rem 1.2rem;border:1px solid rgba(0,0,0,.04);
        box-shadow:var(--shadow-sm);text-align:center;
    }
    .metric-tile .val{font-size:1.6rem;font-weight:800;color:var(--blue-700)}
    .metric-tile .lbl{font-size:.72rem;text-transform:uppercase;
                      letter-spacing:.8px;color:var(--grey-600);font-weight:600}
    .metric-tile .sub{font-size:.8rem;color:var(--grey-600);margin-top:.2rem}

    /* ── Step indicator ─────────────────────────────────── */
    .steps{display:flex;gap:.5rem;align-items:center;margin:.8rem 0 1.2rem;flex-wrap:wrap}
    .step{display:flex;align-items:center;gap:.4rem;
          background:#fff;border:1.5px solid var(--grey-300);
          border-radius:20px;padding:.3rem .9rem;font-size:.82rem;
          font-weight:600;color:var(--grey-600)}
    .step.active{background:var(--blue-700);border-color:var(--blue-700);color:#fff}
    .step.done{background:var(--green-100);border-color:var(--green-700);color:var(--green-800)}
    .step-arrow{color:var(--grey-300);font-size:.9rem}

    /* ── Warn / info boxes ──────────────────────────────── */
    .warn-box{
        background:var(--amber-50);border-left:4px solid var(--amber-700);
        padding:.85rem 1.1rem;border-radius:var(--radius-sm);margin:.6rem 0;
        font-size:.9rem;
    }
    .info-box{
        background:var(--blue-50);border-left:4px solid var(--blue-500);
        padding:.85rem 1.1rem;border-radius:var(--radius-sm);margin:.6rem 0;
        font-size:.9rem;
    }

    /* ── Divider label ──────────────────────────────────── */
    .divider-label{
        display:flex;align-items:center;gap:.8rem;margin:1.4rem 0;
        color:var(--grey-600);font-size:.8rem;font-weight:600;
        text-transform:uppercase;letter-spacing:.8px;
    }
    .divider-label::before,.divider-label::after{
        content:'';flex:1;height:1px;background:rgba(0,0,0,.08);
    }

    /* ── Rank badge ─────────────────────────────────────── */
    .rank-1{background:linear-gradient(135deg,#FFD700,#FFA000);
             color:#fff;border-radius:50%;width:28px;height:28px;
             display:inline-flex;align-items:center;justify-content:center;
             font-weight:800;font-size:.85rem;box-shadow:0 2px 8px rgba(255,160,0,.4)}
    .rank-2{background:linear-gradient(135deg,#CFD8DC,#90A4AE);
             color:#fff;border-radius:50%;width:28px;height:28px;
             display:inline-flex;align-items:center;justify-content:center;
             font-weight:800;font-size:.85rem}
    .rank-3{background:linear-gradient(135deg,#FFAB91,#FF7043);
             color:#fff;border-radius:50%;width:28px;height:28px;
             display:inline-flex;align-items:center;justify-content:center;
             font-weight:800;font-size:.85rem}

    /* ── Sticker table styling ──────────────────────────── */
    .stDataFrame{border-radius:var(--radius-md) !important;overflow:hidden}

    /* ── Sidebar brand block ────────────────────────────── */
    .sb-brand{padding:.8rem .6rem 1rem;border-bottom:1px solid rgba(255,255,255,.15);
              margin-bottom:.8rem}
    .sb-brand-title{font-size:1.15rem;font-weight:800;color:#fff !important;
                    letter-spacing:-.3px}
    .sb-brand-sub{font-size:.72rem;opacity:.6;color:#fff !important}

    .sb-section{font-size:.65rem;font-weight:700;letter-spacing:1.2px;
                text-transform:uppercase;opacity:.5;
                color:#fff !important;margin:.9rem 0 .3rem .2rem}
    .sb-stat{display:flex;justify-content:space-between;align-items:center;
             padding:.18rem 0;font-size:.82rem}
    .sb-stat-n{background:rgba(255,255,255,.15);border-radius:10px;
               padding:.1rem .5rem;font-size:.75rem;font-weight:700}

    /* ── Footer ─────────────────────────────────────────── */
    .footer{text-align:center;color:var(--grey-300);font-size:.78rem;
            margin-top:2.5rem;padding-top:1rem;
            border-top:1px solid rgba(0,0,0,.06)}

    /* ── Streamlit widget polish ────────────────────────── */
    div[data-testid="stFileUploader"]{
        border:2px dashed #90CAF9;border-radius:var(--radius-md);
        background:var(--blue-50);padding:.5rem;
    }
    div[data-testid="stFileUploader"]:hover{border-color:var(--blue-700)}
    .stButton>button{border-radius:var(--radius-sm) !important;font-weight:600 !important}
    .stButton>button[kind="primary"]{
        background:linear-gradient(135deg,var(--blue-700),var(--blue-900)) !important;
    }
    div[data-testid="stExpander"]{
        border:1px solid rgba(0,0,0,.06) !important;
        border-radius:var(--radius-md) !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════
def _count_images(folder):
    if not os.path.isdir(folder):
        return 0
    return sum(1 for f in os.listdir(folder)
               if os.path.splitext(f)[1].lower() in IMG_EXTS)


def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return {}
    with open(METRICS_PATH) as f:
        return json.load(f)


def model_available(model_name):
    return os.path.exists(MODEL_PATHS[model_name])


def available_models():
    return [m for m in MODEL_PATHS if model_available(m)]


@st.cache_resource(show_spinner=False)
def _load_keras_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path)


@st.cache_resource(show_spinner=False)
def _load_pickle_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def preprocess_for_dl(img_np, size=IMG_SIZE_DL):
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 1:
        img_np = cv2.cvtColor(img_np[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    img = cv2.resize(img_np, size).astype(np.float32) / 255.0
    return img[np.newaxis]


def preprocess_for_cnn(img_np, size=IMG_SIZE_CNN):
    if img_np.ndim == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np.copy()
    img = cv2.resize(gray, size).astype(np.float32) / 255.0
    return img[np.newaxis, :, :, np.newaxis]


def preprocess_for_ml(img_np, method):
    """Extract Gabor or Wavelet feature vector from a numpy image."""
    if img_np.ndim == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np.copy()
    if method == 'gabor':
        from models.gabor_model import extract_features
    else:
        from models.wavelet_model import extract_features
    return extract_features(gray)


def run_inference(img_np, model_name):
    """
    Run inference with the chosen model.
    Returns (predicted_class_str, confidence_float, probabilities_list).
    """
    path = MODEL_PATHS[model_name]

    if model_name in ('SVM (Gabor)', 'RF (Wavelet)'):
        method = 'gabor' if 'Gabor' in model_name else 'wavelet'
        clf = _load_pickle_model(path)
        feat = preprocess_for_ml(img_np, method).reshape(1, -1)
        proba = clf.predict_proba(feat)[0]
    else:
        # ResNet50 deep learning model
        model = _load_keras_model(path)
        tensor = preprocess_for_dl(img_np)
        proba = model.predict(tensor, verbose=0)[0]

    pred_idx     = int(np.argmax(proba))
    pred_class   = CLASSES[pred_idx]
    confidence   = float(proba[pred_idx]) * 100
    return pred_class, confidence, proba.tolist()


def fig_to_streamlit(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=110)
    buf.seek(0)
    plt.close(fig)
    return buf


# ═══════════════════════════════════════════════════════════════════════════
# Image quality & iris detection helpers
# ═══════════════════════════════════════════════════════════════════════════
def assess_image_quality(img_np):
    """Return quality flags and scores for the uploaded image."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.ndim == 3 else img_np
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(gray))
    h, w = gray.shape[:2]
    return {
        'blur_score': blur_score,
        'brightness': brightness,
        'resolution': (w, h),
        'is_blurry':  blur_score < 50,
        'too_dark':   brightness < 40,
        'too_bright': brightness > 220,
        'low_res':    min(w, h) < 64,
    }


def detect_iris_overlay(img_np):
    """
    Detect the iris boundary using Hough Circle Transform.
    Returns (annotated_img, found_bool).
    """
    annotated = img_np.copy()
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.ndim == 3 else img_np
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    h, w = gray.shape[:2]
    min_r = int(min(h, w) * 0.18)
    max_r = int(min(h, w) * 0.62)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=min(h, w) // 2,
        param1=100, param2=28,
        minRadius=min_r, maxRadius=max_r,
    )
    found = circles is not None
    if found:
        c = np.round(circles[0, 0]).astype(int)
        cx, cy, r = c
        cv2.circle(annotated, (cx, cy), r, (0, 200, 0), 3)
        cv2.circle(annotated, (cx, cy), max(4, r // 8), (255, 80, 0), -1)
    return annotated, found


# ═══════════════════════════════════════════════════════════════════════════
# Detection result display helpers
# ═══════════════════════════════════════════════════════════════════════════
def _show_single_result(model_name, pred_class, confidence, proba):
    color   = CLASS_COLORS.get(pred_class, '#546E7A')
    icon    = CLASS_ICONS.get(pred_class, '🔍')
    css_cls = f"result-{pred_class.lower()}"

    # Confidence badge colour
    if confidence >= 80:
        conf_note = "High confidence"
    elif confidence >= 55:
        conf_note = "Moderate confidence"
    else:
        conf_note = "Low confidence — consider a better image or another model"

    st.markdown(
        f"<div class='{css_cls}'>"
        f"{icon}&nbsp;<b>{pred_class}</b>"
        f"&nbsp;—&nbsp;Confidence: <b>{confidence:.1f}%</b>"
        f"&nbsp;<span style='font-size:.82rem;color:#546E7A'>({conf_note} · {model_name})</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if confidence < 55:
        st.markdown(
            "<div class='warn-box'>🔎 Confidence is below 55 %. "
            "Try uploading a clearer close-up iris image or switching to a different model.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Probability Breakdown")
    if len(proba) == len(CLASSES):
        fig, ax = plt.subplots(figsize=(6, 2.5))
        bar_colors = [CLASS_COLORS[c] for c in CLASSES]
        bars = ax.barh(CLASSES, [p * 100 for p in proba], color=bar_colors)
        for bar, val in zip(bars, [p * 100 for p in proba]):
            ax.text(min(val + 0.5, 106), bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%', va='center', fontsize=9)
        ax.set_xlim(0, 116)
        ax.set_xlabel('Probability (%)')
        ax.set_title(f'{model_name} — Output Probabilities', fontsize=10)
        ax.invert_yaxis()
        plt.tight_layout()
        st.image(fig_to_streamlit(fig), width='content')
    else:
        st.warning(f"Unexpected probability vector length {len(proba)}.")


def _show_multi_model_results(results):
    """Summary table + consensus + per-model probability comparison."""
    from collections import Counter

    rows = []
    for mname, r in results.items():
        if 'error' in r:
            rows.append({'Model': mname, 'Prediction': '❌ Error',
                         'Confidence': '—', 'Note': r['error'][:60]})
        else:
            icon = CLASS_ICONS.get(r['pred'], '')
            rows.append({
                'Model':      mname,
                'Prediction': f"{icon} {r['pred']}",
                'Confidence': f"{r['conf']:.1f}%",
                'Note': '',
            })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    # Consensus banner
    preds = [r['pred'] for r in results.values() if 'pred' in r]
    if preds:
        counts   = Counter(preds)
        majority, votes = counts.most_common(1)[0]
        total    = len(preds)
        icon     = CLASS_ICONS.get(majority, '🔍')
        css      = f"result-{majority.lower()}"
        agree    = votes / total * 100
        st.markdown(
            f"<div class='{css}'>"
            f"{icon}&nbsp;<b>Consensus: {majority}</b>"
            f"&nbsp;—&nbsp;{votes} of {total} models agree ({agree:.0f}%)"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Probability comparison chart
    valid = {m: r for m, r in results.items()
             if 'proba' in r and len(r['proba']) == len(CLASSES)}
    if valid:
        st.markdown("#### Probability Comparison Across Models")
        n = len(valid)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
        if n == 1:
            axes = [axes]
        for ax, (mname, r) in zip(axes, valid.items()):
            bars = ax.barh(CLASSES, [p * 100 for p in r['proba']],
                           color=[CLASS_COLORS[c] for c in CLASSES])
            for bar, val in zip(bars, [p * 100 for p in r['proba']]):
                ax.text(min(val + 1, 110), bar.get_y() + bar.get_height() / 2,
                        f'{val:.0f}%', va='center', fontsize=8)
            ax.set_xlim(0, 122)
            ax.set_title(mname, fontsize=9)
            ax.invert_yaxis()
            ax.tick_params(labelsize=8)
        plt.tight_layout()
        st.image(fig_to_streamlit(fig), width='stretch')


def _show_disease_info(pred_class):
    """Render a disease info card for the predicted class."""
    info = DISEASE_INFO.get(pred_class, DISEASE_INFO['Healthy'])
    syms = "".join(f"<li>{s}</li>" for s in info['symptoms'])
    st.markdown(
        f"<div class='card' style='border-left:6px solid {info['color']};"
        f"background:{info['bg']}'>"
        f"<h3 style='color:{info['color']}'>{info['icon']} {pred_class}</h3>"
        f"<p>{info['description']}</p>"
        f"<b>Common Signs & Symptoms:</b><ul>{syms}</ul>"
        f"<b>Recommendation:</b> {info['recommendation']}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Preprocessing visualisation helper
# ═══════════════════════════════════════════════════════════════════════════
def show_preprocessing_steps(img_np):
    """Show a 4-panel figure: original, grayscale, CLAHE, resized."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    # 1. Original
    axes[0].imshow(img_np if img_np.ndim == 3 else img_np, cmap='gray' if img_np.ndim == 2 else None)
    axes[0].set_title("Original", fontsize=10)

    # Convert to grayscale for further steps
    if img_np.ndim == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np.copy()

    # 2. Grayscale
    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title("Grayscale", fontsize=10)

    # 3. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    axes[2].imshow(enhanced, cmap='gray')
    axes[2].set_title("CLAHE Enhanced", fontsize=10)

    # 4. Resized (224×224 for DL)
    resized = cv2.resize(enhanced, IMG_SIZE_DL)
    axes[3].imshow(resized, cmap='gray')
    axes[3].set_title(f"Resized {IMG_SIZE_DL[0]}×{IMG_SIZE_DL[1]}", fontsize=10)

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════
def sidebar_menu():
    # ── Brand block ───────────────────────────────────────────────────────
    st.sidebar.markdown(
        "<div class='sb-brand'>"
        "<div class='sb-brand-title'>👁 IrisAI</div>"
        "<div class='sb-brand-sub'>Eye Disease Detection System</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Navigation ────────────────────────────────────────────────────────
    st.sidebar.markdown("<div class='sb-section'>Navigation</div>",
                        unsafe_allow_html=True)
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "🔍 Disease Detection",
         "📊 Model Evaluation", "📈 Model Comparison",
         "⚙️ Train Models", "ℹ️ About"],
        label_visibility="collapsed",
    )

    # ── Dataset stats ─────────────────────────────────────────────────────
    st.sidebar.markdown("<div class='sb-section'>Dataset</div>",
                        unsafe_allow_html=True)
    total = 0
    for cls in CLASSES:
        n = _count_images(DATA_PATHS[cls])
        total += n
        st.sidebar.markdown(
            f"<div class='sb-stat'>"
            f"<span>{CLASS_ICONS[cls]}&nbsp;{cls}</span>"
            f"<span class='sb-stat-n'>{n:,}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.sidebar.markdown(
        f"<div class='sb-stat' style='border-top:1px solid rgba(255,255,255,.15);"
        f"margin-top:.4rem;padding-top:.4rem'>"
        f"<span style='font-weight:700'>Total</span>"
        f"<span class='sb-stat-n'>{total:,}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Model status ──────────────────────────────────────────────────────
    st.sidebar.markdown("<div class='sb-section'>Models</div>",
                        unsafe_allow_html=True)
    for mname in MODEL_PATHS:
        if model_available(mname):
            badge = "<span class='badge badge-green' style='font-size:.65rem;padding:.12rem .5rem'>✓ ready</span>"
        else:
            badge = "<span class='badge badge-grey' style='font-size:.65rem;padding:.12rem .5rem'>not trained</span>"
        st.sidebar.markdown(
            f"<div class='sb-stat'><span style='font-size:.8rem'>{mname}</span>{badge}</div>",
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    return page


# ═══════════════════════════════════════════════════════════════════════════
# Page 1 – Home
# ═══════════════════════════════════════════════════════════════════════════
def show_home():
    # ── Hero banner ────────────────────────────────────────────────────────
    totals = {cls: _count_images(DATA_PATHS[cls]) for cls in CLASSES}
    total_imgs = sum(totals.values())
    n_trained  = len(available_models())

    st.markdown(
        "<div class='hero-band'>"
        "<div class='hero-title'>Iris Disease Detection System</div>"
        "<div class='hero-sub'>AI-powered screening for Glaucoma, Myopia, "
        "and Healthy iris conditions using Deep Learning &amp; Machine Learning</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── KPI strip ─────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    kpi_data = [
        (k1, str(total_imgs) if total_imgs else "—", "Training Images",
         CLASS_COLORS['Myopia']),
        (k2, "3", "Detectable Conditions", CLASS_COLORS['Glaucoma']),
        (k3, "6", "Model Architectures", CLASS_COLORS['Healthy']),
        (k4, f"{n_trained}/6", "Models Ready", "#6A1B9A" if n_trained < 6 else CLASS_COLORS['Healthy']),
    ]
    for col, number, label, color in kpi_data:
        with col:
            st.markdown(
                f"<div class='kpi-card' style='border-top:4px solid {color}'>"
                f"<div class='kpi-number' style='color:{color}'>{number}</div>"
                f"<div class='kpi-label'>{label}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='divider-label'>Dataset Breakdown</div>",
                unsafe_allow_html=True)

    # ── Per-class cards ────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    for col, cls in zip([col1, col2, col3], CLASSES):
        color = CLASS_COLORS[cls]
        info  = DISEASE_INFO[cls]
        n     = totals[cls]
        pct   = (n / total_imgs * 100) if total_imgs else 0
        with col:
            st.markdown(
                f"<div class='card' style='border-top:4px solid {color}'>"
                f"<h3 style='color:{color};margin:0 0 .4rem'>{CLASS_ICONS[cls]} {cls}</h3>"
                f"<div style='font-size:2rem;font-weight:800;color:{color}'>{n:,}</div>"
                f"<div style='font-size:.78rem;color:#546E7A;margin:.2rem 0 .6rem'>"
                f"{pct:.1f}% of dataset</div>"
                f"<p style='font-size:.82rem;color:#546E7A;margin:0'>{info['description']}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='divider-label'>Methodology &amp; Models</div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🔬 Pipeline Overview")
        st.markdown("""
| Step | Description |
|:----:|-------------|
| **1** | Collect iris images (Healthy / Glaucoma / Myopia) |
| **2** | Preprocess: grayscale → CLAHE → resize to 224×224 |
| **3** | Feature extraction: **Gabor Filter** → SVM · **Wavelet** → RF |
| **4** | Deep learning: **ResNet50** (transfer learning) |
| **5** | Evaluate: Accuracy · FAR · FRR · EER · ROC-AUC |
| **6** | Prototype screening system in Streamlit |
""")

    with c2:
        st.markdown("### 🤖 Model Architecture Summary")
        data = {
            'Model': ['ResNet50', 'SVM (Gabor)', 'RF (Wavelet)'],
            'Category': ['Deep Learning 🧠', 'Machine Learning ⚙️', 'Machine Learning ⚙️'],
            'Feature Extraction': ['ImageNet transfer', 'Gabor filter', '2D Wavelet DWT'],
            'Validation': ['Train/Val/Test split', '5-Fold CV', '5-Fold CV'],
        }
        st.dataframe(pd.DataFrame(data), width='stretch', hide_index=True)

    # ── CTA ────────────────────────────────────────────────────────────────
    st.markdown("<div class='divider-label'>Get Started</div>", unsafe_allow_html=True)
    if not available_models():
        st.markdown(
            "<div class='warn-box'>⚠️ <b>No models trained yet.</b> Navigate to "
            "<b>⚙️ Train Models</b> to train them before running detection.</div>",
            unsafe_allow_html=True,
        )
    else:
        cta_col, _, _ = st.columns([1, 1, 2])
        with cta_col:
            if st.button("🔍 Run Disease Detection →", type="primary",
                         width='stretch'):
                st.session_state.page = "🔍 Disease Detection"
                st.rerun()

    st.markdown(
        "<div class='footer'>IrisAI v2.0 &nbsp;·&nbsp; Research Prototype &nbsp;·&nbsp; "
        "Not for clinical use &nbsp;·&nbsp; 2026</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Page 2 – Disease Detection
# ═══════════════════════════════════════════════════════════════════════════
def show_detection():
    st.markdown('<h1 class="main-header">Iris Disease Detection</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Upload any iris or eye image — the system will detect '
        'Healthy, Glaucoma, or Myopia conditions</p>',
        unsafe_allow_html=True,
    )
    # Step indicator
    st.markdown(
        "<div class='steps'>"
        "<div class='step done'>1 Upload Image</div>"
        "<div class='step-arrow'>›</div>"
        "<div class='step'>2 Choose Model</div>"
        "<div class='step-arrow'>›</div>"
        "<div class='step'>3 View Diagnosis</div>"
        "<div class='step-arrow'>›</div>"
        "<div class='step'>4 Read Condition Info</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── How-to guide ──────────────────────────────────────────────────────
    with st.expander("ℹ️ How to use & tips for best results", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**✅ Good images**")
            st.markdown(
                "- Clear, in-focus close-up iris photo\n"
                "- Slit-lamp or dedicated iris camera\n"
                "- Minimum 64 × 64 pixels\n"
                "- JPG, PNG or BMP format"
            )
        with c2:
            st.markdown("**❌ Avoid**")
            st.markdown(
                "- Blurry or out-of-focus shots\n"
                "- Very dark or overexposed images\n"
                "- Whole-face photos (iris too small)\n"
                "- Heavily compressed screenshots"
            )
        with c3:
            st.markdown("**📷 Ideal sources**")
            st.markdown(
                "- Medical slit-lamp photographs\n"
                "- Dedicated iris recognition cameras\n"
                "- High-resolution eye-closeup photos\n"
                "- Kaggle iris / eye disease datasets"
            )

    # ── Model availability check ──────────────────────────────────────────
    avail = available_models()
    if not avail:
        st.error("No trained models found in `saved_models/`.")
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔄 Check again", type="primary"):
                st.cache_resource.clear()
                st.rerun()
        st.markdown("**Expected model files:**")
        for name, path in MODEL_PATHS.items():
            tag = "✅" if os.path.exists(path) else "❌"
            st.markdown(f"- {tag} `{path}`")
        st.info("Navigate to **⚙️ Train Models** to train them first.")
        return

    # ── Upload + Options ──────────────────────────────────────────────────
    col_up, col_opt = st.columns([3, 2])
    with col_up:
        uploaded = st.file_uploader(
            "Upload iris image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a close-up iris or eye photo (JPG / PNG / BMP)",
        )
    with col_opt:
        st.markdown("**Inference Settings**")
        run_all = st.checkbox(
            "Run all available models",
            value=False,
            help="Run every trained model and compare predictions",
        )
        if not run_all:
            model_choice = st.selectbox("Select model", avail,
                                        help="Choose which model to use")
        else:
            model_choice = None
        show_iris_detect  = st.checkbox("Show iris detection overlay", value=True)
        show_preprocess   = st.checkbox("Show preprocessing pipeline", value=False)
        if st.button("🔄 Refresh models"):
            st.cache_resource.clear()
            st.rerun()

    # ── Placeholder cards when nothing uploaded ───────────────────────────
    if uploaded is None:
        st.markdown("---")
        st.markdown("#### Detectable Conditions")
        col_a, col_b, col_c = st.columns(3)
        for col, cls in zip([col_a, col_b, col_c], CLASSES):
            info = DISEASE_INFO[cls]
            with col:
                st.markdown(
                    f"<div class='card' style='border-top:4px solid {info['color']};"
                    f"text-align:center'>"
                    f"<h3 style='color:{info['color']}'>{info['icon']} {cls}</h3>"
                    f"<p style='font-size:.85rem;color:#546E7A'>{info['description']}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        return

    # ── Load & validate image ─────────────────────────────────────────────
    try:
        pil_img = Image.open(uploaded).convert('RGB')
        img_np  = np.array(pil_img)
    except Exception as e:
        st.error(f"Could not open image: {e}")
        return

    st.markdown("---")

    # ── Quality warnings ──────────────────────────────────────────────────
    quality = assess_image_quality(img_np)
    issues  = []
    if quality['is_blurry']:
        issues.append(f"Image may be blurry (sharpness: {quality['blur_score']:.1f})")
    if quality['too_dark']:
        issues.append(f"Image is very dark (brightness: {quality['brightness']:.0f}/255)")
    if quality['too_bright']:
        issues.append(f"Image is overexposed (brightness: {quality['brightness']:.0f}/255)")
    if quality['low_res']:
        w, h = quality['resolution']
        issues.append(f"Low resolution ({w}×{h} px) — results may be less reliable")
    if issues:
        bullets = "".join(f"<li>⚠️ {i}</li>" for i in issues)
        st.markdown(
            f"<div class='warn-box'><b>Image quality notices:</b><ul>{bullets}</ul>"
            f"Inference will still proceed, but a clearer image may improve accuracy.</div>",
            unsafe_allow_html=True,
        )

    # ── Image display ─────────────────────────────────────────────────────
    col_img, col_overlay = st.columns(2)
    w, h = quality['resolution']
    with col_img:
        st.markdown("**Uploaded Image**")
        st.image(img_np, caption=f"{w}×{h} px  ·  {uploaded.name}",
                 width='stretch')

    with col_overlay:
        if show_iris_detect:
            st.markdown("**Iris Detection Overlay**")
            with st.spinner("Detecting iris boundary…"):
                annotated, iris_found = detect_iris_overlay(img_np)
            if iris_found:
                st.image(annotated,
                         caption="🟢 Green circle = iris boundary  ·  Orange dot = centre",
                         width='stretch')
            else:
                st.image(img_np,
                         caption="⚠️ No iris circle detected automatically",
                         width='stretch')
                st.markdown(
                    "<div class='warn-box'>Could not auto-detect the iris boundary. "
                    "Make sure the image is a close-up eye or iris photo. "
                    "Detection will still proceed on the full image.</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("**Image Properties**")
            st.markdown(f"""
| Property | Value |
|----------|-------|
| Resolution | {w} × {h} px |
| Sharpness score | {quality['blur_score']:.1f} |
| Mean brightness | {quality['brightness']:.1f} / 255 |
| File type | {uploaded.type} |
""")

    # ── Preprocessing pipeline (collapsible) ─────────────────────────────
    if show_preprocess:
        with st.expander("🔬 Preprocessing Pipeline", expanded=True):
            with st.spinner("Preprocessing image…"):
                try:
                    fig = show_preprocessing_steps(img_np)
                    st.image(fig_to_streamlit(fig),
                             caption="Original → Grayscale → CLAHE → Resized",
                             width='stretch')
                except Exception as e:
                    st.warning(f"Preprocessing visualisation failed: {e}")

    # ── Run inference ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Diagnosis")

    models_to_run = avail if run_all else [model_choice]
    results       = {}
    progress_bar  = st.progress(0, text="Running inference…")

    for i, mname in enumerate(models_to_run):
        with st.spinner(f"Running {mname}…"):
            try:
                pred_class, confidence, proba = run_inference(img_np, mname)
                results[mname] = {'pred': pred_class, 'conf': confidence, 'proba': proba}
            except FileNotFoundError as e:
                results[mname] = {'error': str(e)}
            except Exception as e:
                results[mname] = {'error': str(e)}
                import traceback
                results[mname]['traceback'] = traceback.format_exc()
        progress_bar.progress(
            (i + 1) / len(models_to_run),
            text=f"Completed {i + 1}/{len(models_to_run)} model(s)",
        )
    progress_bar.empty()

    # ── Display results ───────────────────────────────────────────────────
    if run_all and len(results) > 1:
        _show_multi_model_results(results)
        # Determine majority for disease info
        from collections import Counter
        preds    = [r['pred'] for r in results.values() if 'pred' in r]
        majority = Counter(preds).most_common(1)[0][0] if preds else 'Healthy'
    else:
        mname = models_to_run[0]
        r     = results.get(mname, {})
        if 'error' in r:
            st.error(f"Inference error ({mname}): {r['error']}")
            if 'traceback' in r:
                with st.expander("Show traceback"):
                    st.code(r['traceback'])
            return
        _show_single_result(mname, r['pred'], r['conf'], r['proba'])
        majority = r['pred']

    # ── Disease information card ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### About the Detected Condition")
    _show_disease_info(majority)

    # ── Medical disclaimer ────────────────────────────────────────────────
    if majority != 'Healthy':
        st.markdown(
            "<div class='warn-box'>⚠️ <b>Medical Disclaimer:</b> This tool is a research "
            "prototype and is <b>not</b> a substitute for professional medical advice, "
            "diagnosis, or treatment. Always consult a qualified ophthalmologist for "
            "clinical evaluation and diagnosis.</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Page 3 – Model Evaluation
# ═══════════════════════════════════════════════════════════════════════════
def show_evaluation():
    st.markdown('<h1 class="main-header">Model Evaluation</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Deep-dive into any trained model — '
        'Accuracy · FAR · FRR · EER · ROC curves · Per-class breakdown</p>',
        unsafe_allow_html=True,
    )

    metrics_all = load_metrics()
    if not metrics_all:
        st.markdown(
            "<div class='warn-box'>⚠️ No evaluation data found. "
            "Train the models first via <b>⚙️ Train Models</b>.</div>",
            unsafe_allow_html=True,
        )
        st.info("Once models are trained, evaluation metrics are saved automatically.")
        return

    model_names = list(metrics_all.keys())
    selected    = st.selectbox("Select model to inspect", model_names,
                               help="Choose a trained model to view its full metrics")
    m           = metrics_all[selected]

    # ── Metric tile row ───────────────────────────────────────────────────
    st.markdown("<div class='divider-label'>Overall Metrics</div>", unsafe_allow_html=True)

    METRIC_META = [
        ("Accuracy",  "accuracy",  "Higher is better",  "#1565C0"),
        ("Precision", "precision", "Higher is better",  "#2E7D32"),
        ("Recall",    "recall",    "Higher is better",  "#6A1B9A"),
        ("FAR",       "far",       "Lower is better",   "#C62828"),
        ("FRR",       "frr",       "Lower is better",   "#E65100"),
        ("EER",       "eer",       "Lower is better",   "#AD1457"),
    ]
    cols = st.columns(6)
    for col, (label, key, hint, color) in zip(cols, METRIC_META):
        val = m.get(key, None)
        display = f"{float(val)*100:.2f}%" if isinstance(val, (int, float)) else "N/A"
        with col:
            st.markdown(
                f"<div class='metric-tile'>"
                f"<div class='val' style='color:{color}'>{display}</div>"
                f"<div class='lbl'>{label}</div>"
                f"<div class='sub'>{hint}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Per-class metrics table ───────────────────────────────────────────
    st.markdown("<div class='divider-label'>Per-Class Metrics (One-vs-Rest)</div>",
                unsafe_allow_html=True)
    per = m.get('per_class', {})
    if per:
        rows = []
        for cls in CLASSES:
            if cls in per:
                d = per[cls]
                eer_val = float(d.get('eer', 0)) * 100
                auc_val = float(d.get('roc_auc', 0))
                # Colour-code EER: good < 10%, ok < 20%, poor >= 20%
                eer_color = ('#2E7D32' if eer_val < 10
                             else '#F9A825' if eer_val < 20 else '#C62828')
                auc_color = ('#2E7D32' if auc_val >= 0.9
                             else '#F9A825' if auc_val >= 0.75 else '#C62828')
                rows.append({
                    'Class':   f"{CLASS_ICONS[cls]} {cls}",
                    'FAR':     f"{float(d.get('far', 0))*100:.2f}%",
                    'FRR':     f"{float(d.get('frr', 0))*100:.2f}%",
                    'EER':     f"{eer_val:.2f}%",
                    'ROC-AUC': f"{auc_val:.4f}",
                })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    else:
        st.info("Per-class breakdown not available for this model.")

    # ── ROC curves + EER bar ──────────────────────────────────────────────
    st.markdown("<div class='divider-label'>Visual Analysis</div>", unsafe_allow_html=True)
    col_roc, col_eer = st.columns(2)

    with col_roc:
        st.markdown("**ROC Curves**")
        if per and all('fpr' in per.get(c, {}) for c in CLASSES if c in per):
            from utils.evaluation import plot_roc_curves
            fig_roc = plot_roc_curves(m, CLASSES, title=f"ROC — {selected}")
            st.image(fig_to_streamlit(fig_roc), width='stretch')
        else:
            st.markdown(
                "<div class='info-box'>ℹ️ Full ROC data is only stored for deep learning "
                "models. ML models (SVM, RF) record averaged metrics.</div>",
                unsafe_allow_html=True,
            )

    with col_eer:
        st.markdown("**Equal Error Rate per Class**")
        if per:
            cls_labels = [c for c in CLASSES if c in per]
            eer_vals   = [float(per[c].get('eer', 0)) * 100 for c in cls_labels]
            colors     = [CLASS_COLORS[c] for c in cls_labels]
            fig3, ax3  = plt.subplots(figsize=(5, 3.2))
            fig3.patch.set_facecolor('#FAFBFC')
            ax3.set_facecolor('#FAFBFC')
            bars = ax3.bar(cls_labels, eer_vals, color=colors,
                           edgecolor='white', linewidth=1.5, zorder=3)
            ax3.yaxis.grid(True, color='#E0E0E0', zorder=0)
            ax3.set_axisbelow(True)
            for bar, val in zip(bars, eer_vals):
                ax3.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.25,
                         f'{val:.2f}%', ha='center', fontsize=9, fontweight='bold')
            ax3.set_ylabel('EER (%)', fontsize=9)
            ax3.set_title(f'Equal Error Rate — {selected}', fontsize=10, fontweight='bold')
            ax3.set_ylim(0, max(eer_vals) * 1.4 + 2 if eer_vals else 10)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            plt.tight_layout()
            st.image(fig_to_streamlit(fig3), width='stretch')


# ═══════════════════════════════════════════════════════════════════════════
# Page 4 – Model Comparison
# ═══════════════════════════════════════════════════════════════════════════
def show_comparison():
    st.markdown('<h1 class="main-header">Model Comparison</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Side-by-side performance of all trained models — '
        'identify the best architecture for iris disease detection</p>',
        unsafe_allow_html=True,
    )

    metrics_all = load_metrics()
    if not metrics_all:
        st.markdown(
            "<div class='warn-box'>⚠️ No models evaluated yet. "
            "Train at least one model first via <b>⚙️ Train Models</b>.</div>",
            unsafe_allow_html=True,
        )
        return

    accs     = {m: float(v.get('accuracy', 0)) for m, v in metrics_all.items()}
    eers     = {m: float(v.get('eer', 1)) for m, v in metrics_all.items()}
    best_acc = max(accs, key=accs.get)
    best_eer = min(eers, key=eers.get)

    # ── Winner cards ──────────────────────────────────────────────────────
    st.markdown("<div class='divider-label'>Top Performers</div>",
                unsafe_allow_html=True)
    w1, w2, w3 = st.columns(3)
    sorted_acc = sorted(accs.items(), key=lambda x: -x[1])
    sorted_eer = sorted(eers.items(), key=lambda x: x[1])
    rank_icons = ['<span class="rank-1">1</span>',
                  '<span class="rank-2">2</span>',
                  '<span class="rank-3">3</span>']

    with w1:
        st.markdown(
            "<div class='card' style='border-top:4px solid #1565C0'>"
            "<div class='kpi-label'>🏆 Highest Accuracy</div>"
            f"<div class='kpi-number' style='color:#1565C0;font-size:1.5rem'>{best_acc}</div>"
            f"<div style='color:#546E7A'>{accs[best_acc]*100:.2f}%</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with w2:
        st.markdown(
            "<div class='card' style='border-top:4px solid #2E7D32'>"
            "<div class='kpi-label'>🎯 Lowest EER</div>"
            f"<div class='kpi-number' style='color:#2E7D32;font-size:1.5rem'>{best_eer}</div>"
            f"<div style='color:#546E7A'>{eers[best_eer]*100:.2f}%</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with w3:
        n = len(metrics_all)
        st.markdown(
            "<div class='card' style='border-top:4px solid #6A1B9A'>"
            "<div class='kpi-label'>📊 Models Evaluated</div>"
            f"<div class='kpi-number' style='color:#6A1B9A;font-size:1.5rem'>{n}</div>"
            "<div style='color:#546E7A'>of 6 architectures</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Ranked summary table ───────────────────────────────────────────────
    st.markdown("<div class='divider-label'>Full Metrics Table</div>",
                unsafe_allow_html=True)

    rows = []
    for rank_i, (mname, _) in enumerate(sorted_acc):
        mv = metrics_all[mname]
        badge = rank_icons[rank_i] if rank_i < 3 else f"#{rank_i+1}"
        rows.append({
            'Rank':      badge,
            'Model':     mname,
            'Accuracy':  f"{float(mv.get('accuracy', 0))*100:.2f}%",
            'Precision': f"{float(mv.get('precision', 0))*100:.2f}%",
            'Recall':    f"{float(mv.get('recall', 0))*100:.2f}%",
            'F1':        f"{float(mv.get('f1', 0))*100:.2f}%",
            'FAR':       f"{float(mv.get('far', 0))*100:.2f}%",
            'FRR':       f"{float(mv.get('frr', 0))*100:.2f}%",
            'EER':       f"{float(mv.get('eer', 0))*100:.2f}%",
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    # ── Visual comparison charts ───────────────────────────────────────────
    st.markdown("<div class='divider-label'>Visual Comparison</div>",
                unsafe_allow_html=True)
    model_labels = [r['Model'] for r in rows]   # ranked order
    col_a, col_b = st.columns(2)

    def _hbar_chart(labels, values, title, xlabel, highlight, hi_color, lo_color, fmt='.1f'):
        fig, ax = plt.subplots(figsize=(6, max(3, len(labels) * 0.55)))
        fig.patch.set_facecolor('#FAFBFC')
        ax.set_facecolor('#FAFBFC')
        colors = [hi_color if m == highlight else lo_color for m in labels]
        bars = ax.barh(labels, values, color=colors,
                       edgecolor='white', linewidth=1.2, zorder=3)
        ax.xaxis.grid(True, color='#E0E0E0', zorder=0)
        ax.set_axisbelow(True)
        for bar, val in zip(bars, values):
            ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                    f'{val:{fmt}}%', va='center', fontsize=9, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, max(values) * 1.22 + 1 if values else 10)
        plt.tight_layout()
        return fig

    with col_a:
        st.markdown("**Accuracy Ranking**")
        acc_vals = [accs[m] * 100 for m in model_labels]
        fig_a = _hbar_chart(model_labels, acc_vals,
                            'Accuracy (%)', 'Accuracy (%)',
                            best_acc, '#1565C0', '#90CAF9')
        st.image(fig_to_streamlit(fig_a), width='stretch')

    with col_b:
        st.markdown("**EER Ranking** *(lower = better)*")
        eer_vals = [eers[m] * 100 for m in model_labels]
        fig_e = _hbar_chart(model_labels, eer_vals,
                            'Equal Error Rate (%)', 'EER (%)',
                            best_eer, '#2E7D32', '#A5D6A7', fmt='.2f')
        st.image(fig_to_streamlit(fig_e), width='stretch')

    # ── ROC-AUC heat table ────────────────────────────────────────────────
    st.markdown("<div class='divider-label'>ROC-AUC by Class</div>",
                unsafe_allow_html=True)
    roc_rows = []
    for mname in model_labels:
        per = metrics_all[mname].get('per_class', {})
        roc_rows.append({
            'Model': mname,
            **{cls: round(float(per.get(cls, {}).get('roc_auc', 0)), 4)
               for cls in CLASSES},
        })
    if roc_rows:
        st.dataframe(pd.DataFrame(roc_rows), width='stretch', hide_index=True)
        st.markdown(
            "<div class='info-box'>ℹ️ ROC-AUC closer to 1.0 is better. "
            "ML models (SVM, RF) may show 0.0 if full curve data was not stored.</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Page 5 – Train Models
# ═══════════════════════════════════════════════════════════════════════════
def show_training():
    st.markdown('<h1 class="main-header">Train Models</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Train any combination of models on the iris disease dataset · '
        'ML models finish in minutes · Deep learning models may take hours</p>',
        unsafe_allow_html=True,
    )

    # ── Dataset overview ──────────────────────────────────────────────────
    st.markdown("<div class='divider-label'>Dataset Overview</div>", unsafe_allow_html=True)
    total_imgs = sum(_count_images(DATA_PATHS[cls]) for cls in CLASSES)
    d_cols = st.columns(4)
    with d_cols[0]:
        st.markdown(
            f"<div class='kpi-card' style='border-top:4px solid #1565C0'>"
            f"<div class='kpi-number' style='color:#1565C0'>{total_imgs:,}</div>"
            f"<div class='kpi-label'>Total Images</div></div>",
            unsafe_allow_html=True,
        )
    for col, cls in zip(d_cols[1:], CLASSES):
        n = _count_images(DATA_PATHS[cls])
        color = CLASS_COLORS[cls]
        pct = (n / total_imgs * 100) if total_imgs else 0
        with col:
            st.markdown(
                f"<div class='kpi-card' style='border-top:4px solid {color}'>"
                f"<div class='kpi-number' style='color:{color}'>{n:,}</div>"
                f"<div class='kpi-label'>{CLASS_ICONS[cls]} {cls}</div>"
                f"<div style='font-size:.75rem;color:#546E7A'>{pct:.1f}% of dataset</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Model status cards ────────────────────────────────────────────────
    st.markdown("<div class='divider-label'>Current Model Status</div>",
                unsafe_allow_html=True)
    MODEL_TYPES = {
        'ResNet50':    ('Deep Learning', '🧠'),
        'SVM (Gabor)': ('Machine Learning', '⚙️'),
        'RF (Wavelet)':('Machine Learning', '⚙️'),
    }
    TRAIN_TIME = {
        'ResNet50':    '30–60 min',
        'SVM (Gabor)': '~5–15 min',  'RF (Wavelet)':'~5–15 min',
    }
    status_cols = st.columns(len(MODEL_PATHS))
    for col, mname in zip(status_cols, MODEL_PATHS):
        mtype, micon = MODEL_TYPES[mname]
        ready = model_available(mname)
        border = '#2E7D32' if ready else '#B0BEC5'
        badge = ("<span class='badge badge-green'>✓ Ready</span>"
                 if ready else
                 "<span class='badge badge-grey'>Not trained</span>")
        with col:
            st.markdown(
                f"<div class='card' style='border-top:3px solid {border};text-align:center'>"
                f"<div style='font-size:1.4rem'>{micon}</div>"
                f"<b style='font-size:.88rem'>{mname}</b><br>"
                f"<span style='font-size:.72rem;color:#546E7A'>{mtype}</span><br>"
                f"<div style='margin:.5rem 0'>{badge}</div>"
                f"<div style='font-size:.72rem;color:#90A4AE'>~{TRAIN_TIME[mname]}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    if st.button("🔄 Refresh model status", help="Re-scan saved_models/ folder"):
        st.cache_resource.clear()
        st.rerun()

    st.markdown("<div class='divider-label'>Training Options</div>",
                unsafe_allow_html=True)

    tab_a, tab_b = st.tabs([
        "💻  Option A — Terminal (Recommended)",
        "⚙️  Option B — Launch from App"
    ])

    venv_python = sys.executable.replace("\\", "/")

    # ── Tab A: Terminal ───────────────────────────────────────────────────
    with tab_a:
        st.markdown(
            "<div class='info-box'>ℹ️ Running in a terminal gives you full live logs "
            "and avoids Streamlit timeouts on long training runs.</div>",
            unsafe_allow_html=True,
        )
        BASE_FWD = BASE_DIR.replace(chr(92), '/')
        steps = [
            ("Step 1 — ML models (fastest, minutes)",
             f'cd "{BASE_FWD}" && "{venv_python}" train_disease_models.py --models svm,rf'),
            ("Step 2 — ResNet50 deep learning (~30–60 min)",
             f'cd "{BASE_FWD}" && "{venv_python}" train_disease_models.py --models resnet50'),
            ("Or train all 3 models at once",
             f'cd "{BASE_FWD}" && "{venv_python}" train_disease_models.py --models svm,rf,resnet50'),
        ]
        for label, cmd_str in steps:
            st.markdown(f"**{label}**")
            st.code(cmd_str, language='bash')
        st.markdown("##### Estimated Training Times")
        st.dataframe(pd.DataFrame([
            {'Model': 'SVM (Gabor)',  'Type': 'ML',            'Estimated Time': '~5–15 min'},
            {'Model': 'RF (Wavelet)', 'Type': 'ML',            'Estimated Time': '~5–15 min'},
            {'Model': 'ResNet50',     'Type': 'DL Transfer',   'Estimated Time': '~30–60 min'},
        ]), width='stretch', hide_index=True)

    # ── Tab B: Launch from App ────────────────────────────────────────────
    with tab_b:
        st.markdown(
            "<div class='warn-box'>⚠️ Streamlit may time out for ResNet50 training. "
            "Use the Terminal tab for long-running jobs.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("**Select models to train:**")
        model_flags = {
            'ResNet50':    st.checkbox('ResNet50', value=False),
            'SVM (Gabor)': st.checkbox('SVM + Gabor (5-fold CV)', value=True),
            'RF (Wavelet)':st.checkbox('RF + Wavelet (5-fold CV)', value=True),
        }
        MODEL_KEYS = {
            'ResNet50': 'resnet50',
            'SVM (Gabor)': 'svm', 'RF (Wavelet)': 'rf',
        }
        chosen = [MODEL_KEYS[m] for m, sel in model_flags.items() if sel]
        if not chosen:
            st.warning("Please select at least one model.")
            return
        if st.button("🚀 Start Training", type="primary"):
            models_arg = ','.join(chosen)
            train_script_path = os.path.join(BASE_DIR, 'train_disease_models.py')
            cmd = [sys.executable, train_script_path, '--models', models_arg]
            st.info(f"Running: `{' '.join(cmd)}`")
            log_placeholder = st.empty()
            full_log = ""
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=BASE_DIR,
                )
                for line in proc.stdout:
                    full_log += line
                    lines = full_log.strip().split('\n')
                    log_placeholder.code('\n'.join(lines[-30:]), language='')
                proc.wait()
                if proc.returncode == 0:
                    st.success("✅ Training completed! Click 'Refresh model status' above.")
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error(
                        f"Training exited with code {proc.returncode}. "
                        "Try running in a terminal using the Terminal tab."
                    )
            except Exception as e:
                st.error(f"Error launching training: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Page 6 – About
# ═══════════════════════════════════════════════════════════════════════════
def show_about():
    st.markdown('<h1 class="main-header">About the System</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Research prototype · Study overview · Technical details</p>',
        unsafe_allow_html=True,
    )

    tab_ov, tab_pl, tab_ref = st.tabs(["📋 Overview", "🔄 Pipeline", "📖 References"])

    with tab_ov:
        st.markdown(
            "<div class='card'>"
            "<h3 style='margin-top:0'>🔬 Study Overview</h3>"
            "<p>This prototype compares <b>feature-extraction + machine learning</b> with "
            "<b>deep learning</b> for iris disease detection.</p>"
            "<h4>Aims &amp; Objectives</h4>"
            "<ol>"
            "<li>Collect relevant images of normal and infected iris images</li>"
            "<li>Apply <b>Gabor Filter</b> and <b>2D Wavelet Transform</b> for ML feature extraction; "
            "apply <b>ResNet50</b> as the deep learning method</li>"
            "<li>Evaluate models using Accuracy, Precision, FAR, FRR, EER, ROC Curve</li>"
            "<li>Implement a prototype screening system (this app) using the best-performing model</li>"
            "</ol>"
            "</div>",
            unsafe_allow_html=True,
        )
        col_s, col_t = st.columns(2)
        with col_s:
            st.markdown(
                "<div class='card'><h4 style='margin-top:0'>🛠 Technology Stack</h4>"
                "<ul><li><b>Python 3.10</b></li>"
                "<li><b>TensorFlow / Keras</b> — deep learning</li>"
                "<li><b>scikit-learn</b> — SVM, Random Forest</li>"
                "<li><b>OpenCV</b> — image processing</li>"
                "<li><b>Streamlit</b> — web app</li>"
                "<li><b>Matplotlib / Pandas</b> — visualisation</li>"
                "</ul></div>",
                unsafe_allow_html=True,
            )
        with col_t:
            st.markdown(
                "<div class='card'><h4 style='margin-top:0'>⚠️ Disclaimer</h4>"
                "<p>This application is a <b>research prototype only</b>.</p>"
                "<ul>"
                "<li>Results are <b>not clinically validated</b></li>"
                "<li>Do <b>not</b> use for medical decision-making</li>"
                "<li>Always consult a qualified <b>ophthalmologist</b></li>"
                "</ul></div>",
                unsafe_allow_html=True,
            )

    with tab_pl:
        st.markdown("### 🔄 Technical Pipeline")
        st.markdown("""
| Stage | Detail |
|-------|--------|
| **Dataset** | Kaggle — Healthy, Glaucoma, Myopia iris images |
| **Preprocessing** | Grayscale → CLAHE → Gaussian blur → Resize to 224×224 |
| **Iris segmentation** | Hough Circle Transform (OpenCV) |
| **ML features** | Gabor filter (→ SVM) · 2D Wavelet DWT (→ RF, 5-fold CV) |
| **DL models** | ResNet50 (ImageNet transfer learning + fine-tune) |
| **Metrics** | Accuracy, Precision, Recall, F1, FAR, FRR, EER, ROC-AUC |
| **Framework** | Python · TensorFlow/Keras · scikit-learn · Streamlit |
""")

    with tab_ref:
        st.markdown(
            "<div class='card'>"
            "<h3 style='margin-top:0'>📖 References</h3>"
            "<ol>"
            "<li>Daugman, J. (2004). How iris recognition works. "
            "<i>IEEE Trans. Circuits Syst. Video Technol.</i></li>"
            "<li>He, K. et al. (2016). Deep residual learning for image recognition. <i>CVPR</i></li>"
            "<li>Howard, A. et al. (2017). MobileNets: Efficient CNNs for mobile vision applications. "
            "<i>arXiv:1704.04861</i></li>"
            "<li>Chollet, F. (2017). Xception: Deep learning with depthwise separable convolutions. "
            "<i>CVPR</i></li>"
            "</ol>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='footer'>IrisAI v2.0 &nbsp;&middot;&nbsp; Research Prototype "
        "&nbsp;&middot;&nbsp; <b>Not for clinical use</b> "
        "&nbsp;&middot;&nbsp; 2026</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# App entry
# ═══════════════════════════════════════════════════════════════════════════
def main():
    apply_custom_css()
    if "page" not in st.session_state:
        st.session_state.page = "🏠 Home"

    page = sidebar_menu()
    if st.session_state.page != page:
        st.session_state.page = page

    if st.session_state.page == "🏠 Home":
        show_home()
    elif st.session_state.page == "🔍 Disease Detection":
        show_detection()
    elif st.session_state.page == "📊 Model Evaluation":
        show_evaluation()
    elif st.session_state.page == "📈 Model Comparison":
        show_comparison()
    elif st.session_state.page == "⚙️ Train Models":
        show_training()
    else:
        show_about()


if __name__ == "__main__":
    main()

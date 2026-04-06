# ==============================================================================
# app.py — PneumoAI  |  Chest X-Ray Analysis
# 5-tab Streamlit UI  |  PyTorch + CUDA  |  DenseNet121 / ResNet50 / EfficientNetB0
# Run:  streamlit run app.py
# ==============================================================================

import json
import io
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T

import streamlit as st

# ==============================================================================
# PAGE CONFIG
# ==============================================================================

st.set_page_config(
    page_title="PneumoAI — Chest X-Ray Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CONSTANTS
# ==============================================================================

MODEL_DIR             = Path("model")
NUM_CLASSES           = 3
DROPOUT_RATE          = 0.4
IMG_SIZE              = (224, 224)
IMAGENET_MEAN         = [0.485, 0.456, 0.406]
IMAGENET_STD          = [0.229, 0.224, 0.225]
UNCERTAINTY_THRESHOLD = 0.35

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_COLORS = {
    "NORMAL":    "#4CAF50",
    "PNEUMONIA": "#f44336",
    "COVID":     "#FF9800",
    "COVID-19":  "#FF9800",
}

PREPROCESS = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ==============================================================================
# MODEL BUILDERS  (must match training notebook architecture exactly)
# ==============================================================================

def _custom_head(in_features: int, num_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.BatchNorm1d(in_features), nn.Dropout(dropout),
        nn.Linear(in_features, 256), nn.ReLU(inplace=True),
        nn.BatchNorm1d(256),         nn.Dropout(dropout),
        nn.Linear(256, num_classes),
    )


def build_densenet(num_classes: int = NUM_CLASSES, dropout: float = DROPOUT_RATE):
    m = tv_models.densenet121(weights=None)
    m.classifier = _custom_head(m.classifier.in_features, num_classes, dropout)
    return m


def build_resnet(num_classes: int = NUM_CLASSES, dropout: float = DROPOUT_RATE):
    m = tv_models.resnet50(weights=None)
    m.fc = _custom_head(m.fc.in_features, num_classes, dropout)
    return m


def build_efficientnet(num_classes: int = NUM_CLASSES, dropout: float = DROPOUT_RATE):
    m = tv_models.efficientnet_b0(weights=None)
    m.classifier = _custom_head(m.classifier[1].in_features, num_classes, dropout)
    return m


# Registry: display_name → (model_type_key, builder_fn, .pt_path)
MODEL_REGISTRY = {
    "DenseNet121":    ("densenet",     build_densenet,    MODEL_DIR / "densenet_3class.pt"),
    "ResNet50":       ("resnet",       build_resnet,      MODEL_DIR / "resnet_3class.pt"),
    "EfficientNetB0": ("efficientnet", build_efficientnet, MODEL_DIR / "efficientnet_3class.pt"),
}

# ==============================================================================
# LOAD MODELS & CONFIG  (@st.cache_resource — runs once per session)
# ==============================================================================

@st.cache_resource(show_spinner="Loading models…")
def load_everything():
    """Load all .pt models + config.json.  Returns (models_dict, idx_to_class, ens_weights, cfg)."""
    cfg = {}
    cfg_path = MODEL_DIR / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)

    classes      = cfg.get("classes",      ["COVID", "NORMAL", "PNEUMONIA"])
    class_to_idx = cfg.get("class_to_idx", {c: i for i, c in enumerate(classes)})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    raw_w        = cfg.get("ensemble_weights", [1 / 3, 1 / 3, 1 / 3])
    ens_weights  = np.array(raw_w, dtype=np.float32)

    loaded = {}
    for display_name, (mtype, builder, pt_path) in MODEL_REGISTRY.items():
        if not pt_path.exists():
            continue
        ckpt  = torch.load(str(pt_path), map_location=DEVICE)
        model = builder(
            num_classes=ckpt.get("num_classes", NUM_CLASSES),
            dropout=ckpt.get("dropout_rate", DROPOUT_RATE),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(DEVICE).eval()
        loaded[display_name] = {"model": model, "type": mtype}

    return loaded, idx_to_class, ens_weights, cfg


loaded_models, IDX_TO_CLASS, ENS_WEIGHTS, CONFIG = load_everything()

if not loaded_models:
    st.error(
        "No trained `.pt` model files found in `model/`.\n\n"
        "Run the training notebook to generate `densenet_3class.pt`, "
        "`resnet_3class.pt`, and `efficientnet_3class.pt` first."
    )
    st.stop()

MODEL_NAMES_ORDERED = list(loaded_models.keys())   # ordered insertion, Python 3.7+
NUM_LOADED          = len(MODEL_NAMES_ORDERED)

# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:
    st.title("⚙️ Settings")

    confidence_threshold = st.slider(
        "Confidence Threshold", min_value=0.10, max_value=1.00,
        value=0.50, step=0.01,
    )

    show_gradcam = st.checkbox("Show Grad-CAM", value=True)

    mc_enabled = st.checkbox(
        "Uncertainty Mode (slower)", value=False,
        help="Runs MC Dropout — multiple stochastic forward passes per model.",
    )
    mc_passes = 50
    if mc_enabled:
        mc_passes = st.slider("MC Dropout Passes", 10, 100, 50, step=10)

    model_choice = st.selectbox(
        "Primary Model (Tab 1)",
        options=["All (Ensemble)"] + MODEL_NAMES_ORDERED,
    )

    st.divider()
    st.caption(f"Device: `{DEVICE}`")
    st.caption("Classes: " + " · ".join(IDX_TO_CLASS.values()))

    st.divider()
    st.markdown(
        """
        <div style='font-size:0.78rem; line-height:1.8; color:#555;'>
            <b style='font-size:0.85rem; color:inherit;'>Rajib Roy</b><br>
            SR No. 24459 &nbsp;·&nbsp; M.Tech (AI)<br>
            IISc, Bengaluru<br>
            <a href='mailto:rajibroy@iisc.ac.in' style='color:inherit;'>rajibroy@iisc.ac.in</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# INFERENCE HELPERS
# ==============================================================================

def to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """PIL (any mode) → (1, 3, 224, 224) float32 tensor on DEVICE."""
    return PREPROCESS(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)


@torch.no_grad()
def _single_pass(model: nn.Module, tensor: torch.Tensor) -> np.ndarray:
    """Deterministic forward pass.  Returns (C,) numpy softmax probs."""
    model.eval()
    return F.softmax(model(tensor), dim=1)[0].cpu().numpy()


def _mc_forward(model: nn.Module, tensor: torch.Tensor, n: int) -> tuple[np.ndarray, np.ndarray]:
    """MC Dropout: n stochastic passes with dropout active.  Returns (mean, std) each (C,)."""
    model.eval()
    # Enable only Dropout layers (BatchNorm must stay in eval to work with batch size 1)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    with torch.no_grad():
        stack = np.stack([
            F.softmax(model(tensor), dim=1)[0].cpu().numpy()
            for _ in range(n)
        ])
    model.eval()
    return stack.mean(0), stack.std(0)


def run_all_models(
    tensor: torch.Tensor,
    mc: bool = False,
    passes: int = 50,
) -> dict:
    """
    Run all loaded models and compute ensemble.

    Returns dict keyed by model name + "Ensemble".
    Each value:
        probs       (C,) ndarray   — mean probabilities
        std         (C,) ndarray   — std (MC only, else zeros)
        pred_idx    int
        pred_label  str
        confidence  float          — max probability
        [Ensemble only]
        disagreement float
        uncertainty  float
        refer        bool
    """
    per_model  = {}
    probs_list = []

    for name in MODEL_NAMES_ORDERED:
        entry = loaded_models[name]
        model = entry["model"]

        if mc:
            mean_p, std_p = _mc_forward(model, tensor, passes)
        else:
            mean_p = _single_pass(model, tensor)
            std_p  = np.zeros_like(mean_p)

        pred_idx   = int(mean_p.argmax())
        per_model[name] = {
            "probs":      mean_p,
            "std":        std_p,
            "pred_idx":   pred_idx,
            "pred_label": IDX_TO_CLASS.get(pred_idx, str(pred_idx)),
            "confidence": float(mean_p[pred_idx]),
        }
        probs_list.append(mean_p)

    # ── Weighted ensemble ──────────────────────────────────────────────────────
    probs_stack = np.stack(probs_list)                              # (M, C)
    n_m = len(probs_list)
    w   = ENS_WEIGHTS[:n_m] if len(ENS_WEIGHTS) >= n_m else np.ones(n_m, dtype=np.float32)
    w   = w / w.sum()
    ens_probs    = (probs_stack * w[:, None]).sum(axis=0)           # (C,)
    ens_pred_idx = int(ens_probs.argmax())

    # Uncertainty
    disagreement = float(probs_stack.std(axis=0).mean())
    entropy      = float(-np.sum(ens_probs * np.log(ens_probs + 1e-8)))
    norm_entropy = entropy / np.log(max(len(ens_probs), 2))
    uncertainty  = 0.6 * norm_entropy + 0.4 * disagreement

    per_model["Ensemble"] = {
        "probs":        ens_probs,
        "std":          np.zeros_like(ens_probs),
        "pred_idx":     ens_pred_idx,
        "pred_label":   IDX_TO_CLASS.get(ens_pred_idx, str(ens_pred_idx)),
        "confidence":   float(ens_probs[ens_pred_idx]),
        "disagreement": disagreement,
        "uncertainty":  uncertainty,
        "refer":        bool(uncertainty > UNCERTAINTY_THRESHOLD),
    }

    return per_model

# ==============================================================================
# GRAD-CAM
# ==============================================================================

def _target_layer(model: nn.Module, model_type: str) -> nn.Module:
    if model_type == "densenet":     return model.features.denseblock4
    elif model_type == "resnet":     return model.layer4
    elif model_type == "efficientnet": return model.features[7]
    raise ValueError(f"Unknown model_type: {model_type!r}")


def _gradcam(
    model: nn.Module,
    model_type: str,
    tensor: torch.Tensor,
    class_idx: int,
) -> np.ndarray:
    """Return normalised Grad-CAM heatmap (H, W) in [0, 1]."""
    acts, grads = {}, {}
    layer = _target_layer(model, model_type)

    fh = layer.register_forward_hook(
        lambda _m, _i, out: acts.__setitem__("v", out.detach())
    )
    bh = layer.register_full_backward_hook(
        lambda _m, _gi, go: grads.__setitem__("v", go[0].detach())
    )

    model.eval()
    output = model(tensor)
    model.zero_grad()
    output[0, class_idx].backward()

    fh.remove()
    bh.remove()

    pooled  = grads["v"][0].mean(dim=[1, 2])            # (C,)
    heatmap = (acts["v"][0] * pooled.view(-1, 1, 1)).sum(dim=0)
    heatmap = F.relu(heatmap).cpu().numpy()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return cv2.resize(heatmap, IMG_SIZE)


def _overlay(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend Grad-CAM jet colormap over image.  Returns RGB uint8 (H, W, 3)."""
    base = np.array(pil_img.resize(IMG_SIZE)).astype(np.float32)
    jet  = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    jet  = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB).astype(np.float32)
    out  = (1 - alpha) * base + alpha * jet
    return np.clip(out, 0, 255).astype(np.uint8)


def build_gradcam_grid(
    pil_img: Image.Image,
    tensor: torch.Tensor,
    all_results: dict,
) -> tuple[plt.Figure, bytes]:
    """
    Build 2×2 matplotlib figure:
        [Original]        [DenseNet121 Grad-CAM]
        [ResNet50 Grad-CAM] [EfficientNetB0 Grad-CAM]
    Returns (fig, png_bytes).
    """
    ens_class = all_results["Ensemble"]["pred_idx"]

    panels = [("Original", np.array(pil_img.resize(IMG_SIZE)))]
    for name in MODEL_NAMES_ORDERED:
        if name not in loaded_models:
            continue
        entry   = loaded_models[name]
        hmap    = _gradcam(entry["model"], entry["type"], tensor, ens_class)
        ol_img  = _overlay(pil_img, hmap)
        r       = all_results[name]
        caption = f"{name}\n{r['pred_label']}  {r['confidence']*100:.1f}%"
        panels.append((caption, ol_img))

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, (title, img) in zip(axes.flatten(), panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=8, pad=4)
        ax.axis("off")
    # hide unused cells
    for ax in axes.flatten()[len(panels):]:
        ax.set_visible(False)

    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return fig, buf.read()

# ==============================================================================
# SHARED UI COMPONENTS
# ==============================================================================

def _badge(label: str):
    """Coloured prediction badge rendered via HTML."""
    color = CLASS_COLORS.get(label.upper(), "#9E9E9E")
    st.markdown(
        f'<div style="background:{color};color:white;padding:14px 18px;'
        f'border-radius:10px;text-align:center;font-size:22px;font-weight:700;'
        f'letter-spacing:1px;margin-bottom:8px;">{label}</div>',
        unsafe_allow_html=True,
    )


def _class_prob_bars(probs: np.ndarray):
    """Per-class probability rows with inline HTML progress bars."""
    for idx in sorted(IDX_TO_CLASS.keys()):
        cls   = IDX_TO_CLASS[idx]
        prob  = float(probs[idx])
        color = CLASS_COLORS.get(cls.upper(), "#9E9E9E")
        bar   = (
            f'<div style="background:#e0e0e0;border-radius:4px;height:10px;">'
            f'<div style="width:{prob*100:.1f}%;background:{color};height:10px;'
            f'border-radius:4px;"></div></div>'
        )
        c1, c2, c3 = st.columns([2, 1, 4])
        c1.markdown(f"**{cls}**")
        c2.markdown(f"{prob*100:.1f}%")
        c3.markdown(bar, unsafe_allow_html=True)


def _plotly_hbar(probs: np.ndarray) -> go.Figure:
    """Horizontal bar chart for per-class probabilities (plotly)."""
    classes = [IDX_TO_CLASS.get(i, str(i)) for i in sorted(IDX_TO_CLASS.keys())]
    colors  = [CLASS_COLORS.get(c.upper(), "#9E9E9E") for c in classes]
    values  = [float(probs[i]) * 100 for i in sorted(IDX_TO_CLASS.keys())]
    fig = go.Figure(go.Bar(
        x=values, y=classes, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        xaxis=dict(title="Probability (%)", range=[0, 120]),
        yaxis=dict(autorange="reversed"),
        height=170,
        margin=dict(l=0, r=0, t=10, b=10),
        showlegend=False,
    )
    return fig

# ==============================================================================
# TABS
# ==============================================================================

st.markdown(
    """
    <div style='text-align:center; padding:12px 0 4px 0;'>
        <span style='font-size:2.4rem; font-weight:800; letter-spacing:2px;'>🫁 PneumoAI</span><br>
        <span style='font-size:1rem; color:grey;'>AI-Powered Chest X-Ray Analysis · DenseNet121 · ResNet50 · EfficientNet-B0</span>
    </div>
    <hr style='margin:8px 0 16px 0;'>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 Single Predict",
    "⚖️ Model Comparison",
    "📂 Batch Inference",
    "📊 Performance Dashboard",
    "ℹ️ About",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — Single Predict
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🔬 Single Image Prediction")

    f1 = st.file_uploader(
        "Upload a chest X-ray image", type=["jpg", "jpeg", "png"], key="t1"
    )

    if f1:
        pil_img1 = Image.open(f1).convert("RGB")
        tensor1  = to_tensor(pil_img1)
        st.image(pil_img1, caption="Uploaded X-Ray", width=300)

        with st.spinner("Running inference…"):
            results1 = run_all_models(tensor1, mc=mc_enabled, passes=mc_passes)

        # ── Determine primary result ───────────────────────────────────────────
        primary = (
            results1["Ensemble"]
            if model_choice == "All (Ensemble)"
            else results1.get(model_choice, results1["Ensemble"])
        )

        left, right = st.columns([4, 6])

        # ── LEFT: metrics ──────────────────────────────────────────────────────
        with left:
            _badge(primary["pred_label"])

            conf = primary["confidence"]
            st.metric("Confidence", f"{conf * 100:.2f}%")
            st.progress(conf)

            st.markdown("**Per-class probabilities**")
            _class_prob_bars(primary["probs"])

            if mc_enabled:
                ens1 = results1["Ensemble"]
                st.divider()
                col_a, col_b = st.columns(2)
                col_a.metric("Uncertainty Score",  f"{ens1.get('uncertainty',  0):.2f}")
                col_b.metric("Disagreement",        f"{ens1.get('disagreement', 0):.2f}")
                if ens1.get("refer", False):
                    st.warning("⚠️ High uncertainty — Refer to Radiologist")

        # ── RIGHT: Grad-CAM ────────────────────────────────────────────────────
        with right:
            if show_gradcam:
                with st.spinner("Generating Grad-CAM overlays…"):
                    fig1, png1 = build_gradcam_grid(pil_img1, tensor1, results1)
                st.pyplot(fig1, clear_figure=True)
                plt.close(fig1)
                st.download_button(
                    "📥 Download Grad-CAM Grid (PNG)",
                    data=png1,
                    file_name="gradcam_overlay.png",
                    mime="image/png",
                )
            else:
                st.info("Enable **Show Grad-CAM** in the sidebar to see overlays.")

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — Model Comparison
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.header("⚖️ Model Comparison")

    f2 = st.file_uploader(
        "Upload a chest X-ray image", type=["jpg", "jpeg", "png"], key="t2"
    )

    if f2:
        pil_img2 = Image.open(f2).convert("RGB")
        tensor2  = to_tensor(pil_img2)

        with st.spinner("Running all models…"):
            results2 = run_all_models(tensor2, mc=mc_enabled, passes=mc_passes)

        display_order = MODEL_NAMES_ORDERED + ["Ensemble"]
        cols2 = st.columns(len(display_order))

        for col, name in zip(cols2, display_order):
            r = results2.get(name)
            if r is None:
                continue
            with col:
                st.markdown(f"#### {name}")
                _badge(r["pred_label"])
                st.metric("Confidence", f"{r['confidence'] * 100:.2f}%")

                fig_bar2 = _plotly_hbar(r["probs"])
                st.plotly_chart(fig_bar2, use_container_width=True)

                if show_gradcam and name in loaded_models:
                    with st.spinner(f"Grad-CAM ({name})…"):
                        entry2 = loaded_models[name]
                        hmap2  = _gradcam(
                            entry2["model"], entry2["type"],
                            tensor2, r["pred_idx"],
                        )
                        ol2 = _overlay(pil_img2, hmap2)
                    st.image(ol2, caption=f"{name} Grad-CAM", use_column_width=True)

        # ── Agreement analysis ─────────────────────────────────────────────────
        st.divider()
        st.subheader("Agreement Analysis")

        model_preds  = {n: results2[n]["pred_label"] for n in MODEL_NAMES_ORDERED}
        unique_preds = set(model_preds.values())

        if len(unique_preds) == 1:
            st.success(f"✅ All models agree: **{next(iter(unique_preds))}**")
        else:
            parts = ", ".join(f"{k} → {v}" for k, v in model_preds.items())
            st.warning(f"⚠️ Models disagree: {parts}")

        # Pairwise confidence Δ
        delta_rows = []
        names2 = MODEL_NAMES_ORDERED
        for i in range(len(names2)):
            for j in range(i + 1, len(names2)):
                n1, n2 = names2[i], names2[j]
                d = abs(results2[n1]["confidence"] - results2[n2]["confidence"])
                delta_rows.append({"Model A": n1, "Model B": n2,
                                   "Confidence Δ": f"{d * 100:.2f}%"})
        if delta_rows:
            st.table(pd.DataFrame(delta_rows))

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — Batch Inference
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📂 Batch Inference")

    batch_files = st.file_uploader(
        "Upload multiple chest X-ray images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="t3",
    )

    if batch_files:
        rows3   = []
        prog3   = st.progress(0, text="Processing…")
        n_files = len(batch_files)

        for i, bf in enumerate(batch_files):
            img3    = Image.open(bf).convert("RGB")
            tensor3 = to_tensor(img3)

            with st.spinner(f"Inferring {bf.name}…"):
                res3 = run_all_models(tensor3, mc=False)

            row = {"Filename": bf.name}

            # Per-model columns (abbreviated names to keep table narrow)
            for name in MODEL_NAMES_ORDERED:
                r3 = res3[name]
                abbr = name[:8]
                row[f"{abbr} Pred"]  = r3["pred_label"]
                row[f"{abbr} Conf%"] = round(r3["confidence"] * 100, 2)

            # Ensemble
            ens3 = res3["Ensemble"]
            row["Ensemble Pred"]  = ens3["pred_label"]
            row["Ensemble Conf%"] = round(ens3["confidence"] * 100, 2)

            # Agreement across individual models
            m_preds = [res3[n]["pred_label"] for n in MODEL_NAMES_ORDERED]
            row["Agreement"] = "Yes" if len(set(m_preds)) == 1 else "No"

            # Confidence range across models
            confs3 = [res3[n]["confidence"] for n in MODEL_NAMES_ORDERED]
            row["Δ Conf%"] = round((max(confs3) - min(confs3)) * 100, 2)

            rows3.append(row)
            prog3.progress((i + 1) / n_files, text=f"Processed {i+1}/{n_files}")

        prog3.empty()
        df3 = pd.DataFrame(rows3)
        df3.index = df3.index + 1

        # ── Styling ────────────────────────────────────────────────────────────
        pred_cols3 = [c for c in df3.columns if "Pred" in c]

        def _style_pred(val):
            v = str(val).upper()
            if "PNEUMONIA" in v: return "background-color:#ffcccc;color:black;"
            if "NORMAL"    in v: return "background-color:#ccffcc;color:black;"
            if "COVID"     in v: return "background-color:#ffe0b2;color:black;"
            return ""

        def _style_agree(val):
            return "background-color:#fff3cd;color:black;" if val == "No" else ""

        conf_fmt_cols = {c: "{:.2f}" for c in df3.columns if "Conf%" in c or "Δ" in c}

        styled3 = (
            df3.style
            .map(_style_pred,  subset=pred_cols3)
            .map(_style_agree, subset=["Agreement"])
            .format(conf_fmt_cols)
        )

        st.subheader("📋 Results")
        st.dataframe(styled3, use_container_width=True)

        # ── Summary charts ─────────────────────────────────────────────────────
        st.subheader("📈 Summary")
        c3a, c3b = st.columns(2)

        with c3a:
            pred_counts = df3["Ensemble Pred"].value_counts().reset_index()
            pred_counts.columns = ["Class", "Count"]
            color_map = {c: CLASS_COLORS.get(c.upper(), "#9E9E9E")
                         for c in pred_counts["Class"]}
            fig_pie = px.pie(
                pred_counts, names="Class", values="Count",
                title="Ensemble Prediction Distribution",
                color="Class", color_discrete_map=color_map,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

        with c3b:
            avg_conf3 = (
                df3.groupby("Ensemble Pred")["Ensemble Conf%"]
                .mean()
                .reset_index()
            )
            avg_conf3.columns = ["Class", "Avg Confidence (%)"]
            fig_avg = px.bar(
                avg_conf3, x="Class", y="Avg Confidence (%)",
                title="Avg Confidence per Predicted Class",
                color="Class",
                color_discrete_map={c: CLASS_COLORS.get(c.upper(), "#9E9E9E")
                                    for c in avg_conf3["Class"]},
                text_auto=".1f",
            )
            fig_avg.update_layout(showlegend=False,
                                  yaxis=dict(range=[0, 115]))
            st.plotly_chart(fig_avg, use_container_width=True)

        # ── CSV download ───────────────────────────────────────────────────────
        csv3 = df3.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Results (CSV)", csv3,
            file_name="batch_predictions.csv", mime="text/csv",
        )

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — Performance Dashboard
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.header("📊 Performance Dashboard")

    metrics_raw = CONFIG.get("metrics", [])
    if not metrics_raw:
        st.info(
            "No evaluation metrics found in `model/config.json`. "
            "Train the models using the notebook to populate this dashboard."
        )
    else:
        mdf = pd.DataFrame(metrics_raw)

        # ── Metrics summary table ──────────────────────────────────────────────
        st.subheader("Model Metrics Summary")
        num_cols4 = [c for c in mdf.columns if c != "Model"]
        styled4   = (
            mdf.style
            .format({c: "{:.4f}" for c in num_cols4})
            .highlight_max(subset=num_cols4, color="#d4edda", axis=0)
        )
        st.dataframe(styled4, use_container_width=True)

        # ── Grouped bar: per-class F1 ──────────────────────────────────────────
        f1_cols4 = [c for c in mdf.columns if c.startswith("F1-")]
        if f1_cols4:
            st.subheader("Per-Class F1 Score")
            fig_f1 = go.Figure()
            for col in f1_cols4:
                cls_name = col.replace("F1-", "")
                color    = CLASS_COLORS.get(cls_name.upper(), "#9E9E9E")
                fig_f1.add_trace(go.Bar(
                    name=cls_name,
                    x=mdf["Model"],
                    y=mdf[col],
                    marker_color=color,
                    text=mdf[col].map(lambda v: f"{v:.3f}"),
                    textposition="outside",
                ))
            fig_f1.update_layout(
                barmode="group",
                yaxis=dict(title="F1 Score", range=[0, 1.12]),
                legend_title="Class",
                height=380,
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig_f1, use_container_width=True)

        # ── Radar chart ────────────────────────────────────────────────────────
        radar_candidates = ["Accuracy", "Macro AUC", "Macro F1"]
        radar_cols4      = [c for c in radar_candidates if c in mdf.columns]

        if radar_cols4:
            st.subheader("Radar: Accuracy / Macro AUC / Macro F1")
            fig_radar = go.Figure()
            theta4 = radar_cols4 + [radar_cols4[0]]   # close the polygon

            for _, row4 in mdf.iterrows():
                vals4 = [float(row4[c]) for c in radar_cols4] + [float(row4[radar_cols4[0]])]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals4, theta=theta4,
                    fill="toself", name=str(row4["Model"]),
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True, height=420,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── Architecture expander ──────────────────────────────────────────────
        with st.expander("🏗️ Model Architecture Summary"):
            arch_rows = []
            grad_layer_map = {
                "densenet":     "features.denseblock4",
                "resnet":       "layer4",
                "efficientnet": "features[7]",
            }
            for dname, entry4 in loaded_models.items():
                mdl    = entry4["model"]
                total  = sum(p.numel() for p in mdl.parameters())
                n_layers = len(list(mdl.modules()))
                arch_rows.append({
                    "Model":          dname,
                    "Total Params":   f"{total:,}",
                    "Modules":        n_layers,
                    "Grad-CAM Layer": grad_layer_map.get(entry4["type"], "—"),
                })
            st.table(pd.DataFrame(arch_rows))

# ══════════════════════════════════════════════════════════════════════
# TAB 5 — About
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.header("ℹ️ About")

    st.markdown("""
**PneumoAI** is an end-to-end deep learning system for automated chest X-ray
classification distinguishing **Normal**, **Pneumonia**, and **COVID-19** findings.
    """)

    st.subheader("👤 Developer")
    st.markdown(
        """
        <div style='line-height:2; font-size:0.95rem;'>
            <b>Rajib Roy</b><br>
            <span style='color:grey;'>SR No. 24459</span><br>
            M.Tech in Artificial Intelligence<br>
            Indian Institute of Science (IISc), Bengaluru<br>
            <a href='mailto:rajibroy@iisc.ac.in'>rajibroy@iisc.ac.in</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Dataset ───────────────────────────────────────────────────────────────
    st.subheader("📁 Dataset")
    st.table(pd.DataFrame([
        {"Source": "chest-xray-pneumonia (Kermany, Kaggle)",
         "Classes": "Normal, Pneumonia", "Images": "~5,863", "License": "CC BY 4.0"},
        {"Source": "COVID-19 Radiography Database (Kaggle)",
         "Classes": "COVID-19, Normal, Viral Pneumonia, Lung Opacity",
         "Images": "~21,165", "License": "CC BY 4.0"},
        {"Source": "Combined unified dataset",
         "Classes": "NORMAL, PNEUMONIA, COVID",
         "Images": "~27,000", "License": "—"},
    ]))

    # ── Models ────────────────────────────────────────────────────────────────
    st.subheader("🧠 Model Architectures")
    st.table(pd.DataFrame([
        {"Model": "DenseNet121",    "Params": "~8 M",    "Backbone": "Dense blocks (121 layers)",
         "Head": "BN→Drop→Linear(256)→ReLU→BN→Drop→Linear(3)",
         "Grad-CAM Layer": "features.denseblock4"},
        {"Model": "ResNet50",       "Params": "~26 M",   "Backbone": "Residual blocks (50 layers)",
         "Head": "BN→Drop→Linear(256)→ReLU→BN→Drop→Linear(3)",
         "Grad-CAM Layer": "layer4"},
        {"Model": "EfficientNetB0", "Params": "~5.3 M",  "Backbone": "MBConv compound scaling",
         "Head": "BN→Drop→Linear(256)→ReLU→BN→Drop→Linear(3)",
         "Grad-CAM Layer": "features[7]"},
    ]))

    # ── Training ──────────────────────────────────────────────────────────────
    st.subheader("⚙️ Training Setup")
    st.markdown("""
| Stage | Detail |
|-------|--------|
| Loss | Focal Loss (γ=2, α=0.25) + inverse-frequency class weights |
| Phase 1 | Frozen backbone · 5 epochs · lr = 1e-3 |
| Phase 2 | Unfreeze last block · up to 20 epochs · lr = 1e-5 |
| Regularisation | EarlyStopping (patience=5) · ReduceLROnPlateau (factor=0.3) |
| Augmentation | Horizontal flip · rotation ±15° · affine shift/scale · colour jitter |
| Normalisation | ImageNet mean/std |
| Hardware | NVIDIA RTX 3060 Laptop · CUDA 12.4 · AMP (FP16) |
    """)

    # ── Uncertainty ───────────────────────────────────────────────────────────
    st.subheader("🎲 Uncertainty Quantification")
    st.markdown("""
Two complementary signals are fused into a **combined uncertainty score**:

| Signal | Source | Interpretation |
|--------|--------|----------------|
| **Aleatoric (entropy)** | MC Dropout — 50 stochastic forward passes per model | Data-level ambiguity |
| **Epistemic (disagreement)** | Std of per-model probability vectors across ensemble | Model-level uncertainty |

**Combined score** = 0.6 × normalised entropy + 0.4 × disagreement

If the combined score exceeds **0.35**, the case is flagged:
> ⚠️ *High uncertainty — Refer to Radiologist*
    """)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.divider()
    st.warning(
        "⚠️ **Disclaimer:** PneumoAI is for **educational and research purposes only**. "
        "It must **not** be used as a substitute for professional medical diagnosis, "
        "radiological expertise, or clinical judgement. "
        "Always consult a qualified healthcare professional for medical decisions."
    )

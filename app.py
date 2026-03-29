from __future__ import annotations

import textwrap
from io import BytesIO

import streamlit as st
from PIL import Image

from pneumonia_app.inference import get_model_stats, predict_image, prepare_display_image
from pneumonia_app.visuals import create_lung_overlay, render_lung_status_card

st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detector",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

        :root {
            --cream: #f7efe3;
            --paper: #fffaf2;
            --slate: #21333a;
            --muted: #5b686d;
            --teal: #1b8f72;
            --warning: #c44536;
        }

        html, body, [class*="css"]  {
            font-family: "Space Grotesk", sans-serif;
            color: var(--slate);
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(244, 162, 97, 0.18), transparent 24%),
                radial-gradient(circle at top left, rgba(27, 143, 114, 0.12), transparent 26%),
                linear-gradient(180deg, #f9f2e8 0%, #f4efe7 42%, #eef5f4 100%);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        h1, h2, h3 {
            font-family: "Fraunces", Georgia, serif;
            color: #173139;
            letter-spacing: -0.02em;
        }

        .hero-shell {
            background: linear-gradient(135deg, rgba(255,250,242,0.92), rgba(255,244,227,0.95));
            border: 1px solid rgba(33, 51, 58, 0.08);
            border-radius: 28px;
            padding: 1.7rem 1.7rem 1.4rem 1.7rem;
            box-shadow: 0 18px 50px rgba(34, 55, 60, 0.08);
            margin-bottom: 1.2rem;
        }

        .hero-kicker {
            display: inline-block;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: var(--teal);
            background: rgba(27, 143, 114, 0.10);
            padding: 0.45rem 0.7rem;
            border-radius: 999px;
            margin-bottom: 0.9rem;
            font-weight: 700;
        }

        .hero-title {
            font-size: 3.1rem;
            line-height: 1.02;
            margin: 0;
        }

        .hero-copy {
            font-size: 1.02rem;
            color: var(--muted);
            max-width: 760px;
            margin-top: 0.85rem;
            line-height: 1.65;
        }

        .metric-card, .panel-card {
            background: rgba(255, 250, 242, 0.92);
            border: 1px solid rgba(33, 51, 58, 0.08);
            border-radius: 24px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 30px rgba(34, 55, 60, 0.06);
            height: 100%;
        }

        .metric-label {
            font-size: 0.84rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .metric-value {
            font-size: 1.7rem;
            font-weight: 700;
            margin-top: 0.35rem;
            color: #10262d;
        }

        .metric-sub {
            font-size: 0.92rem;
            margin-top: 0.35rem;
            color: var(--muted);
            line-height: 1.5;
        }

        .section-title {
            margin-top: 0.35rem;
            margin-bottom: 0.7rem;
        }

        .prob-shell {
            width: 100%;
            height: 14px;
            background: rgba(33, 51, 58, 0.08);
            border-radius: 999px;
            overflow: hidden;
        }

        .prob-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #f2b46e 0%, #c44536 100%);
        }

        .prob-fill.safe {
            background: linear-gradient(90deg, #98e9c3 0%, #1b8f72 100%);
        }

        .info-pill {
            display: inline-block;
            margin-top: 0.55rem;
            padding: 0.4rem 0.7rem;
            border-radius: 999px;
            font-size: 0.86rem;
            font-weight: 600;
            background: rgba(33, 51, 58, 0.07);
            color: #173139;
        }

        .lung-hero-card {
            background: transparent;
            min-height: 420px;
        }

        [data-testid="stFileUploader"] > div {
            background: rgba(255, 250, 242, 0.88);
            border-radius: 24px;
            border: 1px dashed rgba(27, 143, 114, 0.32);
            padding: 0.7rem;
        }

        .disclaimer {
            background: rgba(255, 250, 242, 0.9);
            border-left: 5px solid #c44536;
            border-radius: 16px;
            padding: 0.9rem 1rem;
            color: #5e4b44;
            line-height: 1.6;
            font-size: 0.95rem;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, subtext: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_meter(label: str, probability: float, positive: bool) -> None:
    width = max(4, min(100, round(probability * 100)))
    fill_class = "prob-fill" if positive else "prob-fill safe"
    st.markdown(
        f"""
        <div class="panel-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{probability * 100:.1f}%</div>
          <div class="prob-shell">
            <div class="{fill_class}" style="width: {width}%;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-shell">
          <div class="hero-kicker">AI Chest Screening</div>
          <h1 class="hero-title">Chest X-Ray Pneumonia Detector</h1>
          <p class="hero-copy">
            Upload a chest X-ray image to get an instant AI-based screening result, confidence scores,
            and a visual lung response view. The experience is designed for simple, fast use by end users.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_stats(stats: dict) -> None:
    st.markdown('<h2 class="section-title">Model Statistics</h2>', unsafe_allow_html=True)

    cards = st.columns(4)
    with cards[0]:
        render_metric_card("Architecture", "MobileNet", "Keras binary classifier restored from your saved model bundle.")
    with cards[1]:
        param_text = f"{stats['parameter_count']:,}" if stats.get("parameter_count") else "Unavailable"
        render_metric_card("Parameters", param_text, "Loaded at runtime when TensorFlow/Keras is available locally.")
    with cards[2]:
        render_metric_card("Input Shape", "224 x 224 x 3", "Matches the preprocessing path defined in the Kaggle notebook.")
    with cards[3]:
        render_metric_card("Saved Bundle", f"{stats.get('weights_size_mb', 'N/A')} MB", f"Keras {stats.get('saved_keras_version', 'Unknown')}")

    dataset = stats["dataset"]
    reported = stats["reported_metrics"]

    st.markdown(
        f"""
        <div class="stats-grid">
          <div class="panel-card">
            <div class="metric-label">Dataset Split</div>
            <div class="metric-sub">
              Total: <strong>{dataset['total_images']}</strong><br>
              Train: <strong>{dataset['train_images']}</strong><br>
              Test: <strong>{dataset['test_images']}</strong><br>
              Validation: <strong>{dataset['validation_images']}</strong>
            </div>
          </div>
          <div class="panel-card">
            <div class="metric-label">Class Balance</div>
            <div class="metric-sub">
              Pneumonia: <strong>{dataset['pneumonia_images']}</strong><br>
              Normal: <strong>{dataset['normal_images']}</strong><br>
              Output classes: <strong>2</strong>
            </div>
          </div>
          <div class="panel-card">
            <div class="metric-label">Reported Metrics</div>
            <div class="metric-sub">
              Validation Accuracy: <strong>{reported['validation_accuracy']}</strong><br>
              Validation Loss: <strong>{reported['validation_loss']}</strong><br>
              ROC-AUC: <strong>{reported['roc_auc']}</strong>
            </div>
          </div>
          <div class="panel-card">
            <div class="metric-label">Training Setup</div>
            <div class="metric-sub">
              Optimizer: <strong>{stats['optimizer']}</strong><br>
              Loss: <strong>{stats['loss']}</strong><br>
              Last conv layer: <strong>{stats.get('last_conv_layer') or 'Auto-detected at runtime'}</strong>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if stats.get("load_warning"):
        st.warning(
            "The app could not fully inspect the model in this environment yet, so parameter count may appear unavailable. "
            "Once TensorFlow is installed locally, the loader should resolve the full stats automatically."
        )


def uploaded_image_to_pil(file) -> Image.Image:
    data = BytesIO(file.getvalue())
    return Image.open(data).convert("RGB")


def main() -> None:
    inject_styles()
    render_header()

    stats = get_model_stats()
    render_model_stats(stats)

    st.markdown('<h2 class="section-title">Analyze a Chest X-Ray</h2>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a JPG, JPEG, or PNG chest X-ray",
        type=["jpg", "jpeg", "png"],
        help="The image is preprocessed exactly like the notebook: grayscale -> 224x224 -> RGB stack -> normalization.",
    )

    if not uploaded:
        st.markdown(
            """
            <div class="panel-card">
              <div class="metric-label">Ready for Demo</div>
              <div class="metric-sub">
                Upload an X-ray to view the original scan, the AI-enhanced lung overlay, a stylized lung status card,
                prediction probabilities, and the model summary panel.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    image = uploaded_image_to_pil(uploaded)
    display_image = prepare_display_image(image)
    try:
        result = predict_image(image)
    except ModuleNotFoundError as exc:
        missing_package = getattr(exc, "name", "a required package")
        st.error(
            f"The app cannot run prediction yet because `{missing_package}` is not installed in the active Python environment."
        )
        st.info(
            "Activate your `.venv` and install the app dependencies again with "
            "`python -m pip install --upgrade pip` then `python -m pip install -r requirements.txt`."
        )
        st.stop()
    except Exception as exc:
        st.error("The model could not be loaded for prediction.")
        st.code(str(exc))
        st.stop()
    overlay_image = create_lung_overlay(
        image=display_image,
        label=result.label,
        confidence=result.confidence,
        heatmap=result.heatmap,
    )

    summary_cols = st.columns([1.1, 0.9, 0.9, 0.9])
    with summary_cols[0]:
        diagnosis_copy = "Healthy-looking pattern" if result.label == "Normal" else "Pneumonia pattern detected"
        render_metric_card("Diagnosis", result.label, diagnosis_copy)
    with summary_cols[1]:
        render_metric_card("Confidence", f"{result.confidence * 100:.1f}%", result.risk_band)
    with summary_cols[2]:
        render_metric_card("Notebook Layer", result.last_conv_layer or "Unavailable", "Last convolutional layer used for visual response mapping.")
    with summary_cols[3]:
        render_metric_card("Threshold", f"{result.threshold:.2f}", "Binary decision cutoff ported as a simple local inference rule.")

    prob_cols = st.columns(2)
    with prob_cols[0]:
        render_probability_meter("Pneumonia Probability", result.pneumonia_probability, positive=True)
    with prob_cols[1]:
        render_probability_meter("Normal Probability", result.normal_probability, positive=False)

    viewer_cols = st.columns([1.05, 1.05, 0.9])
    with viewer_cols[0]:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.image(display_image, caption="Uploaded chest X-ray", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with viewer_cols[1]:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.image(overlay_image, caption="AI lung overlay", use_container_width=True)
        st.caption(
            "This overlay is presentation-focused. When pneumonia is predicted, it blends a Grad-CAM style hotspot map with a lung mask. "
            "When normal is predicted, it uses a calm healthy lung tint."
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with viewer_cols[2]:
        st.markdown(render_lung_status_card(result.label, result.confidence, result.pneumonia_probability), unsafe_allow_html=True)
        st.markdown(
            f'<span class="info-pill">{result.risk_band}</span>',
            unsafe_allow_html=True,
        )

    with st.expander("Notebook Port Details", expanded=False):
        st.markdown(
            textwrap.dedent(
                """
                - Preprocessing matches the notebook: grayscale conversion, resize to `224 x 224`, channel stacking to RGB, normalization by `255.0`.
                - Model architecture matches the saved bundle: `MobileNet(weights=None, include_top=False)` + `GlobalAveragePooling2D` + `Dense(1, sigmoid)`.
                - Training notes displayed in the UI come from the project notebook/report summary you shared.
                - The heatmap is for explanation and demo value only, not exact clinical localization.
                """
            )
        )

    st.markdown(
        """
        <div class="disclaimer">
          This project UI is built for academic demos and local experimentation. It should not be used as a standalone
          clinical diagnostic tool or as a substitute for a radiologist's judgment.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

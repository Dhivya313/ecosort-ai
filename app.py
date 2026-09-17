"""
EcoSort AI — Main Streamlit Application
AI-powered waste identification and sustainable disposal assistant.
"""

import hashlib
import streamlit as st

from waste_analyzer import (
    analyze_waste_image,
    MissingAPIKeyError,
    UnsupportedImageError,
    InvalidResponseError,
    WasteAnalysisError,
    VALID_CATEGORIES,
)
from rag_engine import RAGEngine

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Master CSS — fonts, colours, layout, blobs, glassmorphism, responsive
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ==========================================================================
   FONTS
   ========================================================================== */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ==========================================================================
   COLOUR TOKENS  (CSS custom properties)
   ========================================================================== */
:root {
    --bg:          #F6F9F8;
    --ink:         #16302E;
    --teal:        #2F6F6E;
    --accent:      #3B7A85;
    --body:        #202A28;
    --conf-good:   #3D7A5B;
    --conf-med:    #B8863C;
    --conf-low:    #B4553F;
    --glass-bg:    rgba(255, 255, 255, 0.55);
    --glass-border:rgba(255, 255, 255, 0.45);
    --glass-shadow: 0 8px 32px rgba(22, 48, 46, 0.08);
    --radius:      20px;
}

/* ==========================================================================
   BASE RESETS
   ========================================================================== */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--body);
}
.stApp {
    background: var(--bg) !important;
}
h1, h2, h3, h4, h5, h6,
.eco-headline, .eco-label, .step-num, .step-title {
    font-family: 'Space Grotesk', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }
.block-container { max-width: 820px; padding-top: 1rem; }

/* Upload container glass styling */
div[data-testid="stFileUploader"] {
    position: relative; z-index: 1;
}
.upload-container {
    background: var(--glass-bg);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    box-shadow: var(--glass-shadow);
    padding: 2rem 2.2rem 1rem;
    margin-bottom: 1.8rem;
    position: relative; z-index: 1;
}

/* ==========================================================================
   ANIMATED BACKGROUND BLOBS  (CSS only)
   ========================================================================== */
.blob-container {
    position: fixed; inset: 0; z-index: 0;
    pointer-events: none; overflow: hidden;
}
.blob {
    position: absolute;
    border-radius: 50%;
    filter: blur(70px);
    opacity: 0.32;
    will-change: transform;
}
.blob-1 {
    width: 520px; height: 520px;
    background: #8FC1B5;
    top: -10%; left: -8%;
    animation: drift1 24s ease-in-out infinite;
}
.blob-2 {
    width: 440px; height: 440px;
    background: #2F6F6E;
    top: 40%; right: -12%;
    animation: drift2 28s ease-in-out infinite;
}
.blob-3 {
    width: 480px; height: 480px;
    background: #C9DED7;
    bottom: -8%; left: 25%;
    animation: drift3 22s ease-in-out infinite;
}
@keyframes drift1 {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50%      { transform: translate(40px, 30px) scale(1.06); }
}
@keyframes drift2 {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50%      { transform: translate(-35px, -25px) scale(1.05); }
}
@keyframes drift3 {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50%      { transform: translate(25px, -30px) scale(1.07); }
}
@media (prefers-reduced-motion: reduce) {
    .blob { animation: none !important; }
}

/* ==========================================================================
   GLASS CARD
   ========================================================================== */
.glass {
    background: var(--glass-bg);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    box-shadow: var(--glass-shadow);
    padding: 2rem 2.2rem;
    margin-bottom: 1.8rem;
    position: relative;
    z-index: 1;
}

/* ==========================================================================
   HERO
   ========================================================================== */
.hero { text-align: center; padding: 3.5rem 1rem 1.5rem; position: relative; z-index: 1; }
.hero-brand {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem; font-weight: 600;
    color: var(--teal); letter-spacing: 0.06em;
    margin-bottom: 1rem; display: inline-block;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem; font-weight: 700;
    color: var(--ink); line-height: 1.18;
    margin: 0 auto 0.6rem; max-width: 620px;
}
.hero-tagline {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem; font-weight: 500;
    color: var(--teal); letter-spacing: 0.08em;
    margin-bottom: 0.9rem;
}
.hero-sub {
    font-size: 0.97rem; color: #4a5f5c;
    max-width: 500px; margin: 0 auto;
    line-height: 1.55;
}

/* ==========================================================================
   UPLOAD CARD
   ========================================================================== */
.upload-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem; font-weight: 600;
    color: var(--ink); margin-bottom: 0.2rem;
}
.upload-hint {
    font-size: 0.88rem; color: #5a706c;
    margin-bottom: 1rem;
}

/* ==========================================================================
   HOW IT WORKS — timeline
   ========================================================================== */
.timeline-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem; font-weight: 600;
    color: var(--ink); margin-bottom: 1.5rem;
    text-align: center;
}
.timeline {
    display: flex; align-items: flex-start;
    justify-content: center; gap: 0;
}
.step {
    flex: 1; text-align: center;
    max-width: 220px; padding: 0 0.6rem;
    position: relative;
}
.step-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 38px; height: 38px; border-radius: 50%;
    background: var(--teal); color: #fff;
    font-size: 0.85rem; font-weight: 700;
    margin-bottom: 0.6rem;
}
.step-title {
    font-size: 0.95rem; font-weight: 600;
    color: var(--ink); margin-bottom: 0.3rem;
}
.step-desc {
    font-size: 0.82rem; color: #5a706c;
    line-height: 1.45;
}
/* connector arrows */
.step-arrow {
    flex: 0 0 36px; display: flex;
    align-items: center; justify-content: center;
    padding-top: 8px; color: var(--teal);
    font-size: 1.1rem; opacity: 0.6;
}

/* ==========================================================================
   RESULT CARD
   ========================================================================== */
.result-section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--teal); text-align: center;
    margin-bottom: 0.6rem;
}
.result-grid {
    display: flex; gap: 1.8rem;
    align-items: flex-start;
}
.result-thumb {
    flex: 0 0 180px;
}
.result-thumb img {
    width: 100%; border-radius: 14px;
    object-fit: cover; aspect-ratio: 1;
    box-shadow: 0 4px 16px rgba(22,48,46,0.10);
}
.result-details { flex: 1; min-width: 0; }
.result-item-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem; font-weight: 700;
    color: var(--ink); margin-bottom: 0.55rem;
    text-transform: capitalize;
}
/* pills */
.pill-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.8rem; }
.cat-pill {
    display: inline-block;
    padding: 0.28rem 0.9rem; border-radius: 999px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.82rem; font-weight: 600;
    color: #fff; background: var(--teal);
}
.conf-pill {
    display: inline-block;
    padding: 0.28rem 0.9rem; border-radius: 999px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.82rem; font-weight: 600;
    color: #fff;
}
/* confidence bar */
.conf-bar-wrap { margin-bottom: 0.2rem; }
.conf-bar-labels {
    display: flex; justify-content: space-between;
    font-size: 0.7rem; color: #7a8e8a;
    margin-bottom: 0.2rem;
}
.conf-bar-bg {
    height: 10px; border-radius: 999px;
    background: #dde8e5; overflow: hidden;
}
.conf-bar-fill {
    height: 100%; border-radius: 999px;
    transition: width 0.6s ease;
}
/* reasoning */
.reasoning-box {
    background: rgba(47, 111, 110, 0.06);
    border-left: 3px solid var(--teal);
    border-radius: 0 10px 10px 0;
    padding: 0.85rem 1.1rem;
    margin-top: 0.9rem;
    font-size: 0.9rem; line-height: 1.55;
    color: var(--body);
}
.reasoning-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem; font-weight: 600;
    color: var(--teal); text-transform: uppercase;
    letter-spacing: 0.08em; margin-bottom: 0.3rem;
}

/* ==========================================================================
   GUIDANCE
   ========================================================================== */
.guidance-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem; font-weight: 600;
    color: var(--ink); margin: 1.4rem 0 0.6rem;
}
.eco-tip-box {
    background: rgba(61, 122, 91, 0.08);
    border: 1px solid rgba(61, 122, 91, 0.18);
    border-radius: 14px; padding: 1rem 1.2rem;
    margin-top: 0.5rem;
}
.eco-tip-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.85rem; font-weight: 600;
    color: var(--conf-good); margin-bottom: 0.3rem;
}

/* ==========================================================================
   DISCLAIMER
   ========================================================================== */
.disclaimer-box {
    background: rgba(59, 122, 133, 0.06);
    border: 1px solid rgba(59, 122, 133, 0.12);
    border-radius: 12px; padding: 0.85rem 1.1rem;
    margin-top: 1.5rem;
    font-size: 0.82rem; color: #5a706c;
    text-align: center; line-height: 1.5;
}

/* ==========================================================================
   FOOTER
   ========================================================================== */
.eco-footer {
    text-align: center; padding: 2rem 0 1.5rem;
    font-size: 0.78rem; color: #8a9e9a;
    position: relative; z-index: 1;
}
.eco-footer span { color: var(--teal); font-weight: 600; }

/* ==========================================================================
   SAMPLE OUTPUT PREVIEW
   ========================================================================== */
.sample-preview {
    opacity: 0.45; pointer-events: none;
    filter: grayscale(0.3);
}
.sample-badge {
    display: inline-block; padding: 0.2rem 0.7rem;
    border-radius: 999px; font-size: 0.68rem; font-weight: 600;
    background: var(--teal); color: #fff;
    letter-spacing: 0.08em; margin-bottom: 0.6rem;
}

/* ==========================================================================
   RESPONSIVE
   ========================================================================== */
@media (max-width: 768px) {
    .hero h1 { font-size: 1.75rem; }
    .hero-tagline { font-size: 0.92rem; }
    .hero-sub { font-size: 0.88rem; }
    .glass { padding: 1.4rem 1.2rem; }
    .timeline { flex-direction: column; align-items: center; gap: 0.3rem; }
    .step { max-width: 100%; padding: 0.5rem 0; }
    .step-arrow { transform: rotate(90deg); padding-top: 0; }
    .result-grid { flex-direction: column; }
    .result-thumb { flex: 0 0 auto; max-width: 200px; margin: 0 auto; }
    .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
}
@media (max-width: 480px) {
    .hero { padding: 2rem 0.5rem 1rem; }
    .hero h1 { font-size: 1.45rem; }
    .result-item-name { font-size: 1.1rem; }
}

/* Camera input styling */
div[data-testid="stCameraInput"] img {
    border-radius: 14px;
}
</style>

<!-- Background blobs -->
<div class="blob-container">
    <div class="blob blob-1"></div>
    <div class="blob blob-2"></div>
    <div class="blob blob-3"></div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 1. HERO
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-brand">♻️ EcoSort AI</div>
    <h1>Every item has a right way to be thrown away.</h1>
    <div class="hero-tagline">Identify. Sort. Sustain.</div>
    <p class="hero-sub">Upload a photo of any waste item and get instant AI-powered identification and disposal guidance.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Initialise RAG engine (cached so it only runs once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _init_rag() -> RAGEngine:
    rag = RAGEngine()
    rag.index_knowledge_base()
    return rag


try:
    rag_engine = _init_rag()
except Exception as exc:
    st.error(f"Failed to initialize knowledge base: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# MIME mapping
# ---------------------------------------------------------------------------
_EXT_TO_MIME = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
}

# ---------------------------------------------------------------------------
# 2. UPLOAD CARD (using st.container so uploader sits inside the glass card)
# ---------------------------------------------------------------------------
with st.container():
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    st.markdown('<p class="upload-label">Add Waste Image</p>'
                '<p class="upload-hint">Upload a photo or use your camera to capture a waste item.</p>',
                unsafe_allow_html=True)

    input_mode = st.radio(
        "Choose input method",
        ["📁 Upload Image", "📷 Use Camera"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if input_mode == "📁 Upload Image":
        active_image = st.file_uploader(
            "Upload waste image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG",
            label_visibility="collapsed",
        )
    else:
        active_image = st.camera_input(
            "📷 Take a photo of your waste item",
            label_visibility="collapsed",
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 3. HOW IT WORKS — timeline
# ---------------------------------------------------------------------------
st.markdown("""
<div class="glass">
    <p class="timeline-title">How It Works</p>
    <div class="timeline">
        <div class="step">
            <div class="step-num">01</div>
            <p class="step-title">Upload Image</p>
            <p class="step-desc">Add a photo of your waste item.</p>
        </div>
        <div class="step-arrow">→</div>
        <div class="step">
            <div class="step-num">02</div>
            <p class="step-title">AI Analysis</p>
            <p class="step-desc">Gemini identifies the item and classifies it into a waste category with a confidence score.</p>
        </div>
        <div class="step-arrow">→</div>
        <div class="step">
            <div class="step-num">03</div>
            <p class="step-title">Get Guidance</p>
            <p class="step-desc">RAG retrieves category-specific disposal instructions and eco tips from the knowledge base.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: build result HTML
# ---------------------------------------------------------------------------
def _conf_color(conf: int) -> str:
    if conf >= 80:
        return "var(--conf-good)"
    elif conf >= 50:
        return "var(--conf-med)"
    return "var(--conf-low)"


def _conf_label(conf: int) -> str:
    if conf >= 80:
        return "High"
    elif conf >= 50:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# 4. ANALYSIS RESULT
# ---------------------------------------------------------------------------
_has_result = False

if active_image is not None:
    # Determine MIME type
    ext = active_image.name.rsplit(".", 1)[-1].lower() if "." in active_image.name else ""
    media_type = _EXT_TO_MIME.get(ext, "image/jpeg")
    image_bytes = active_image.getvalue()

    # ---- Dedup: skip re-analysis if the same image is seen again ----------
    _img_hash = hashlib.md5(image_bytes).hexdigest()
    _is_cached = (
        st.session_state.get("_last_img_hash") == _img_hash
        and "_last_result" in st.session_state
    )

    if _is_cached:
        result = st.session_state["_last_result"]
    else:
        # ---- Analyze with Gemini ------------------------------------------
        with st.spinner("🔍 Analyzing your waste item with AI…"):
            try:
                result = analyze_waste_image(image_bytes, media_type)
            except MissingAPIKeyError:
                st.error(
                    "⚠️ **API key not configured.**  \n"
                    "Please add your Gemini API key to the `.env` file."
                )
                st.stop()
            except UnsupportedImageError as exc:
                st.error(f"⚠️ **Unsupported image:** {exc}")
                st.stop()
            except InvalidResponseError:
                st.error(
                    "⚠️ The AI returned an unexpected response. "
                    "Please try again with a clearer image."
                )
                st.stop()
            except WasteAnalysisError as exc:
                st.error(f"⚠️ **Analysis error:** {exc}")
                st.stop()
            except Exception:
                st.error(
                    "⚠️ Something went wrong while analyzing the image. "
                    "Please try again."
                )
                st.stop()
        st.session_state["_last_img_hash"] = _img_hash
        st.session_state["_last_result"] = result

    _has_result = True

    item = result["item"]
    category = result["category"]
    confidence = result["confidence"]
    reasoning = result["reasoning"]
    c_color = _conf_color(confidence)
    c_label = _conf_label(confidence)

    # ---- Retrieve guidance via RAG ----------------------------------------
    if _is_cached and "_last_guidance" in st.session_state:
        guidance = st.session_state["_last_guidance"]
    else:
        with st.spinner("📚 Retrieving disposal guidance…"):
            guidance = rag_engine.get_disposal_guidance(category)
        st.session_state["_last_guidance"] = guidance

    # Parse guidance sections (EXISTING logic, unchanged)
    _sections = {
        "Recommended Disposal": None,
        "Environmental Impact": None,
        "Eco Tip": None,
    }
    if guidance:
        for section_name in _sections:
            marker = f"## {section_name}"
            if marker in guidance:
                start = guidance.index(marker) + len(marker)
                next_section = guidance.find("\n## ", start)
                chunk = guidance[start:next_section].strip() if next_section != -1 else guidance[start:].strip()
                _sections[section_name] = chunk

    # ---- Render result card -----------------------------------------------
    st.markdown('<p class="result-section-label">Analysis Result</p>',
                unsafe_allow_html=True)

    # Show uploaded image and result details side by side
    img_col, detail_col = st.columns([1, 2])
    with img_col:
        st.image(active_image, use_container_width=True)
    with detail_col:
        st.markdown(f"""
        <p class="result-item-name">{item}</p>
        <div class="pill-row">
            <span class="cat-pill">{category}</span>
            <span class="conf-pill" style="background:{c_color}">Confidence: {confidence}% — {c_label}</span>
        </div>
        <div class="conf-bar-wrap">
            <div class="conf-bar-labels"><span>Low</span><span>High</span></div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{confidence}%;background:{c_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # AI Reasoning
    st.markdown(f"""
    <div class="reasoning-box">
        <p class="reasoning-label">AI Reasoning</p>
        {reasoning}
    </div>
    """, unsafe_allow_html=True)

    # ---- Guidance sections ------------------------------------------------
    if _sections["Recommended Disposal"]:
        st.markdown('<p class="guidance-title">♻️ Disposal Guidance</p>',
                    unsafe_allow_html=True)
        st.markdown(_sections["Recommended Disposal"])

    if _sections["Environmental Impact"]:
        st.markdown('<p class="guidance-title">🌍 Environmental Impact</p>',
                    unsafe_allow_html=True)
        st.markdown(_sections["Environmental Impact"])

    if _sections["Eco Tip"]:
        st.markdown(f"""
        <div class="eco-tip-box">
            <p class="eco-tip-label">🌱 Eco Tip</p>
            {_sections["Eco Tip"]}
        </div>
        """, unsafe_allow_html=True)

    if not guidance:
        st.info(
            f"No specific disposal guidance found for **{category}**. "
            "Please check your local municipal guidelines."
        )


# ---------------------------------------------------------------------------
# Sample output preview (shown only when NO image has been analyzed)
# ---------------------------------------------------------------------------
if not _has_result:
    st.markdown("""
    <p class="result-section-label">Sample Output</p>
    <div class="glass sample-preview">
        <div class="result-grid">
            <div class="result-thumb">
                <div style="width:100%;aspect-ratio:1;border-radius:14px;
                            background:linear-gradient(135deg,#C9DED7,#8FC1B5);
                            display:flex;align-items:center;justify-content:center;
                            font-size:2.2rem;">📷</div>
            </div>
            <div class="result-details">
                <p class="result-item-name">Plastic Water Bottle</p>
                <div class="pill-row">
                    <span class="cat-pill">Plastic</span>
                    <span class="conf-pill" style="background:var(--conf-good)">Confidence: 95% — High</span>
                </div>
                <div class="conf-bar-wrap">
                    <div class="conf-bar-labels"><span>Low</span><span>High</span></div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill" style="width:95%;background:var(--conf-good);"></div>
                    </div>
                </div>
                <div class="reasoning-box">
                    <p class="reasoning-label">AI Reasoning</p>
                    PET plastic bottle identified by shape, transparency, and recycling symbol.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 5. RESPONSIBLE AI DISCLAIMER (always visible)
# ---------------------------------------------------------------------------
st.markdown("""
<div class="disclaimer-box">
    ⚠️ Disposal rules can vary by location. Please verify the recommendation
    with your local municipal waste-management guidelines.
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 6. FOOTER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="eco-footer">
    <span>♻️ EcoSort AI</span> — AI-powered waste classification for a sustainable future.
</div>
""", unsafe_allow_html=True)

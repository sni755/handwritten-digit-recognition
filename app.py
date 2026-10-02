import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="AI Handwritten Digit Recognition", page_icon="✍️", layout="wide")

DARK = "#1c2127"
NAVY = "#0b385f"
BLUE = "#3373b0"
LIGHT_BLUE = "#bed4e9"
BACKGROUND = "#e7f1fb"
MAX_FILE_SIZE = 200 * 1024  # 200 KB

st.markdown(f"""
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background-color: {BACKGROUND} !important;
}}
h1, h2, h3 {{ color: {NAVY} !important; }}
p, label {{ color: {DARK} !important; }}
h1 {{ text-align:center !important; font-size:42px !important; font-weight:800 !important; }}
.subtitle {{ text-align:center; color:{DARK} !important; font-size:18px; margin-bottom:30px; }}
.section-title {{ color:{NAVY} !important; font-size:23px; font-weight:800; margin-top:20px; margin-bottom:12px; }}
.info-box, .upload-note {{
    background-color:{LIGHT_BLUE} !important; border:2px solid {BLUE}; border-radius:14px;
    padding:15px 18px; color:{DARK} !important; margin-bottom:20px;
}}
.upload-note {{ font-weight:700; }}
div.stButton > button {{
    background-color:{NAVY} !important; border:2px solid {NAVY} !important; border-radius:12px !important;
    min-height:48px !important; color:white !important; font-weight:800 !important; font-size:16px !important;
}}
div.stButton > button p, div.stButton > button span {{ color:white !important; }}
div.stButton > button:hover {{ background-color:{BLUE} !important; border-color:{BLUE} !important; color:white !important; }}
[data-testid="stFileUploader"] {{
    background-color:{LIGHT_BLUE} !important; border:2px solid {BLUE} !important;
    border-radius:14px !important; padding:15px !important;
}}
[data-testid="stFileUploader"] section {{ background-color:{BACKGROUND} !important; border:none !important; }}
[data-testid="stFileUploader"] button {{ background-color:{NAVY} !important; color:white !important; border:none !important; }}
[data-testid="stFileUploader"] button span {{ color:white !important; }}
/* Hide Streamlit's default file-size helper; our visible note says 200 KB. */
[data-testid="stFileUploader"] small {{ display:none !important; }}
/* Style only the native bordered result containers. */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color:{LIGHT_BLUE} !important; border:3px solid {BLUE} !important;
    border-radius:20px !important; padding:18px !important;
}}
[data-testid="stVerticalBlockBorderWrapper"] h3 {{
    color:{NAVY} !important; text-align:center !important; letter-spacing:1px;
}}
[data-testid="stVerticalBlockBorderWrapper"] h1 {{
    color:{DARK} !important; text-align:center !important; font-size:78px !important;
    line-height:1 !important; margin:12px 0 !important;
}}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stCaptionContainer"] {{ text-align:center !important; }}
.reference-title {{ color:{NAVY}; font-size:21px; font-weight:800; text-align:center; margin-bottom:15px; }}
.footer {{ text-align:center; color:{DARK}; margin-top:40px; padding:20px; border-top:2px solid {BLUE}; font-size:15px; }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_model.keras")

model = load_model()

if "mode" not in st.session_state: st.session_state.mode = None
if "prediction" not in st.session_state: st.session_state.prediction = None
if "confidence" not in st.session_state: st.session_state.confidence = None
if "input_source" not in st.session_state: st.session_state.input_source = None
if "canvas_key" not in st.session_state: st.session_state.canvas_key = 0
if "upload_key" not in st.session_state: st.session_state.upload_key = 0

def preprocess_image(image):
    image = image.convert("L")
    img = np.array(image)
    corners = [img[0,0], img[0,-1], img[-1,0], img[-1,-1]]
    if np.mean(corners) > 127:
        img = 255 - img
    img[img < 40] = 0
    coordinates = np.argwhere(img > 40)
    if len(coordinates) == 0:
        return None
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0)
    digit = img[y_min:y_max + 1, x_min:x_max + 1]
    height, width = digit.shape
    size = max(height, width)
    padding = int(size * 0.25)
    canvas_size = size + padding * 2
    square = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    y_offset = (canvas_size - height) // 2
    x_offset = (canvas_size - width) // 2
    square[y_offset:y_offset + height, x_offset:x_offset + width] = digit
    processed = Image.fromarray(square).resize((28,28), Image.Resampling.LANCZOS)
    processed = np.array(processed).astype("float32") / 255.0
    return processed.reshape(1,28,28,1)

def predict_digit(image):
    processed = preprocess_image(image)
    if processed is None:
        return None, None
    prediction = model.predict(processed, verbose=0)
    digit = int(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0]) * 100)
    return digit, confidence

def create_reference_digit(digit):
    image = Image.new("RGB", (300,300), BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = None
    font_paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    ]
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, 220)
            break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()
    text = str(digit)
    bbox = draw.textbbox((0,0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = (300 - width) / 2
    y = (300 - height) / 2 - 20
    draw.text((x,y), text, fill=NAVY, font=font)
    return image

st.title("✍️ AI Handwritten Digit Recognition")
st.markdown('<div class="subtitle">Draw or upload a handwritten digit and let our CNN recognize it.</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Step 1 — Choose your input method</div>', unsafe_allow_html=True)
st.markdown('<div class="info-box">Choose one method below to provide your handwritten digit.</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("✍️ Draw a Digit", use_container_width=True):
        st.session_state.mode = "draw"
        st.session_state.prediction = None
        st.session_state.confidence = None
        st.session_state.input_source = None
with col2:
    if st.button("📤 Upload an Image", use_container_width=True):
        st.session_state.mode = "upload"
        st.session_state.prediction = None
        st.session_state.confidence = None
        st.session_state.input_source = None

if st.session_state.mode == "draw":
    st.markdown("---")
    st.markdown('<div class="section-title">Step 2 — Draw your digit</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.write("Draw one digit from 0 to 9.")
        canvas = st_canvas(
            fill_color="white", stroke_width=18, stroke_color="white",
            background_color="black", width=400, height=330,
            drawing_mode="freedraw", key=f"canvas_{st.session_state.canvas_key}",
            return_image_data=True
        )
        identify, clear = st.columns(2)
        with identify:
            identify_draw = st.button("🔍 Identify Digit", key="identify_draw", use_container_width=True)
        with clear:
            clear_draw = st.button("🧹 Clear", key="clear_draw", use_container_width=True)
    with right:
        st.markdown('<div class="reference-title">Computer Reference</div>', unsafe_allow_html=True)
        if st.session_state.prediction is not None:
            st.image(create_reference_digit(st.session_state.prediction), width=300)
        else:
            st.info("The computer reference will appear after recognition.")
    if identify_draw:
        if canvas.image_data is None:
            st.warning("Please draw a digit first.")
        else:
            image = Image.fromarray(canvas.image_data.astype("uint8"))
            digit, confidence = predict_digit(image)
            if digit is None:
                st.warning("No digit detected. Please draw more clearly.")
            else:
                st.session_state.prediction = digit
                st.session_state.confidence = confidence
                st.session_state.input_source = "Drawn digit"
                st.rerun()
    if clear_draw:
        st.session_state.canvas_key += 1
        st.session_state.prediction = None
        st.session_state.confidence = None
        st.session_state.input_source = None
        st.rerun()

if st.session_state.mode == "upload":
    st.markdown("---")
    st.markdown('<div class="section-title">Step 2 — Upload your digit</div>', unsafe_allow_html=True)
    st.markdown('''<div class="upload-note">📌 Upload 1 image only — PNG, JPG or JPEG.<br>📦 Maximum file size: 200 KB.</div>''', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose your handwritten digit",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        key=f"digit_upload_{st.session_state.upload_key}"
    )

    if uploaded_file is not None and uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"File is too large. Please upload an image smaller than {MAX_FILE_SIZE // 1024} KB.")
        uploaded_file = None

    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file).convert("RGB")
        left, right = st.columns(2)
        with left:
            st.markdown('<div class="section-title">Your Uploaded Digit</div>', unsafe_allow_html=True)
            st.image(uploaded_image, width=300)
            identify, clear = st.columns(2)
            with identify:
                identify_upload = st.button("🔍 Identify Digit", key="identify_upload", use_container_width=True)
            with clear:
                clear_upload = st.button("🧹 Clear", key="clear_upload", use_container_width=True)
        with right:
            st.markdown('<div class="reference-title">Computer Reference</div>', unsafe_allow_html=True)
            if st.session_state.prediction is not None:
                st.image(create_reference_digit(st.session_state.prediction), width=300)
            else:
                st.info("The computer reference will appear after recognition.")
        if identify_upload:
            digit, confidence = predict_digit(uploaded_image)
            if digit is None:
                st.error("I could not detect a clear digit. Please upload an image containing one digit.")
            else:
                st.session_state.prediction = digit
                st.session_state.confidence = confidence
                st.session_state.input_source = "Uploaded image"
                st.rerun()
        if clear_upload:
            st.session_state.prediction = None
            st.session_state.confidence = None
            st.session_state.input_source = None
            st.session_state.upload_key += 1
            st.rerun()

if st.session_state.prediction is not None and st.session_state.confidence is not None:
    st.markdown("---")
    st.markdown('<div class="section-title">Recognition Result</div>', unsafe_allow_html=True)
    if st.session_state.input_source:
        st.write(f"Result for: **{st.session_state.input_source}**")
    result1, result2 = st.columns(2)
    with result1:
        # Native Streamlit components prevent HTML/Python source from appearing as text.
        with st.container(border=True):
            st.markdown("### PREDICTED DIGIT")
            st.markdown(f"# {st.session_state.prediction}")
            st.caption("CNN prediction")
    with result2:
        with st.container(border=True):
            st.markdown("### MODEL CONFIDENCE")
            st.markdown(f"# {st.session_state.confidence:.2f}%")
            st.caption("Confidence of the CNN prediction")
    st.progress(min(max(st.session_state.confidence / 100, 0.0), 1.0))

st.markdown('''<div class="footer"><b>AI-Based Handwritten Digit Recognition System</b><br><br>Convolutional Neural Network • TensorFlow • Keras • MNIST</div>''', unsafe_allow_html=True)

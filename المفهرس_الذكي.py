import streamlit as st

from PIL import Image
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import re
import torch
import os
import base64

# مكتبات النماذج
from ultralytics import YOLO
import easyocr
from joblib import load

# ======================================
# إعداد تيسراكت (لو انتي على ويندوز)
# ======================================
try:
    import pytesseract
    pytesseract_available = True
    # عدلي المسار لو مختلف عندك
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except Exception:
    pytesseract_available = False

# ======================================
# دوال Base64 للخطوط والصور
# ======================================
def get_font_base64(font_path):
    """تحويل ملف الخط إلى base64"""
    try:
        with open(font_path, "rb") as font_file:
            return base64.b64encode(font_file.read()).decode()
    except Exception as e:
        st.error(f"خطأ في تحميل الخط: {e}")
        return ""

def get_logo_base64(logo_path):
    """تحويل الشعار إلى base64"""
    try:
        with open(logo_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.error(f"خطأ في تحميل الشعار: {e}")
        return ""

    
# ======================================
# إعداد صفحة ستريملايت
# ======================================

# تحميل الخط byt-Regular.otf وتحويله إلى base64
FONT_PATH = os.path.abspath("byt-Regular.otf")
byt_font_base64 = get_font_base64(FONT_PATH)
if byt_font_base64:
    st.markdown(f"""
    <style>
    @font-face {{
        font-family: 'BYT';
        src: url(data:font/opentype;base64,{byt_font_base64}) format('opentype');
        font-weight: normal;
        font-style: normal;
    }}
    html, body, [class^="st"], [class*="st"], * {{
        font-family: 'BYT', 'Tajawal', Arial, sans-serif !important;
        letter-spacing: 0.5px !important;
    }}
    </style>
    """, unsafe_allow_html=True)
st.set_page_config(
    page_title="المفهرس الذكي",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ======================================
# تحميل الخط والشعار من المسار المناسب
# ======================================
# تعريف مسار الشعار وتحويله إلى base64
LOGO_PATH = r"C:/Users/Rahaf/Downloads/dataset/images/train/شعار بيت الثقافة بالخط الابيض.png"
try:
    logo_base64 = get_logo_base64(LOGO_PATH)
except Exception:
    logo_base64 = ""


# ======================================
# تنسيق كحلي + تبويبات بخط أحمر + الخط العربي
# ======================================
st.markdown("""
<style>
    .main-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
        flex-direction: row-reverse;
    }
    .header-title {
        text-align: right;
    }
    .header-title h1 {
        margin: 0;
        color: #ffffff !important;
        font-size: 3.5rem;
        font-weight: bold;
        letter-spacing: 1px;
    }
    .header-logo {
        height: 60px;
        width: auto;
        margin-left: 24px;
    }
    .stApp {
        background: #001a33 !important;
    }
    body {
        background: #001a33 !important;
    }
</style>
""", unsafe_allow_html=True)

# ======================================
# مسارات النماذج (عدليها حسب جهازك)
# ======================================
COVER_MODEL_PATH   = r"C:\Users\Rahaf\Downloads\yolo_best_model_improved.pt"
DEWEY_MODEL_PATH   = r"C:\Users\Rahaf\Downloads\yolov8x.pt"
BARCODE_MODEL_PATH = r"C:\Users\Rahaf\Downloads\runs\detect\barcode_fast5\weights\best.pt"

VECTORIZER_PATH    = r"C:\Users\Rahaf\Downloads\text_vectorizer_svm.joblib"
CLASSIFIER_PATH    = r"C:\Users\Rahaf\Downloads\text_classifier_svm.joblib"

# ======================================
# تحميل النماذج مرة واحدة
# ======================================
@st.cache_resource
def load_models():
    models = {
        "yolo_cover": None,
        "yolo_dewey": None,
        "yolo_barcode": None,
        "reader": None,
        "vectorizer": None,
        "classifier": None
    }

    # نماذج YOLO
    try:
        if os.path.exists(COVER_MODEL_PATH):
            models["yolo_cover"] = YOLO(COVER_MODEL_PATH)
    except Exception as e:
        st.sidebar.error(f"خطأ في تحميل نموذج الغلاف: {e}")

    try:
        if os.path.exists(DEWEY_MODEL_PATH):
            models["yolo_dewey"] = YOLO(DEWEY_MODEL_PATH)
    except Exception as e:
        st.sidebar.error(f"خطأ في تحميل نموذج ديوي: {e}")

    try:
        if os.path.exists(BARCODE_MODEL_PATH):
            models["yolo_barcode"] = YOLO(BARCODE_MODEL_PATH)
    except Exception as e:
        st.sidebar.error(f"خطأ في تحميل نموذج الباركود: {e}")

    # قارئ OCR
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        models["reader"] = easyocr.Reader(['ar', 'en'], gpu=(device == "cuda"))
    except Exception as e:
        st.sidebar.error(f"خطأ في تحميل EasyOCR: {e}")

    # نماذج النص
    try:
        if os.path.exists(VECTORIZER_PATH):
            models["vectorizer"] = load(VECTORIZER_PATH)
        if os.path.exists(CLASSIFIER_PATH):
            models["classifier"] = load(CLASSIFIER_PATH)
    except Exception as e:
        st.sidebar.error(f"خطأ في تحميل نموذج النص: {e}")

    return models

models = load_models()

# ======================================
# دوال مساعدة للنص وديوي والباركود
# ======================================
def classify_text(text: str):
    vec = models["vectorizer"]
    clf = models["classifier"]
    if not vec or not clf or not text:
        return "unknown", 0.0
    try:
        X = vec.transform([text])
        label = clf.predict(X)[0]
        if hasattr(clf, "decision_function"):
            score = float(np.max(clf.decision_function(X)))
        else:
            score = 0.0
        return label, score
    except Exception:
        return "unknown", 0.0

def extract_dewey(text: str) -> str:
    if not text:
        return ""
    text = text.replace(",", ".")
    matches = re.findall(r"\b\d{1,3}(?:\.\d{1,4})?\b", text)
    if not matches:
        return ""
    return sorted(matches, key=lambda x: len(x), reverse=True)[0]

def get_dewey_class(dewey_num: str) -> str:
    try:
        num = int(float(dewey_num))
        ranges = {
            (0, 100): "المعارف العامة",
            (100, 200): "الفلسفة وعلم النفس",
            (200, 300): "الديانات",
            (300, 400): "العلوم الاجتماعية",
            (400, 500): "اللغات",
            (500, 600): "العلوم البحتة",
            (600, 700): "العلوم التطبيقية",
            (700, 800): "الفنون والتسلية",
            (800, 900): "الأدب",
            (900, 1000): "التاريخ والجغرافيا"
        }
        for (start, end), name in ranges.items():
            if start <= num < end:
                return name
        return "غير محدد"
    except Exception:
        return "غير محدد"

def get_library_from_barcode(barcode: str) -> str:
    if barcode.startswith("01"):
        return "المكتبة الرئيسية"
    if barcode.startswith("02"):
        return "مكتبة اليافعين"
    if barcode.startswith("03"):
        return "مكتبة الطفل"
    return "غير محدد"

# ======================================
# دوال معالجة النماذج
# ======================================
def process_book_cover(image: Image.Image):
    """غلاف → عنوان + مؤلف باستخدام YOLO الغلاف + EasyOCR + SVM."""
    if not models["yolo_cover"] or not models["reader"]:
        return "النماذج غير متوفرة", "النماذج غير متوفرة", None

    yolo_model = models["yolo_cover"]
    reader     = models["reader"]

    img = image.convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    H, W = img_cv.shape[:2]

    temp_path = "temp_book_img.jpg"
    img.save(temp_path)
    results = yolo_model(temp_path, conf=0.35, imgsz=640, save=False)
    if os.path.exists(temp_path):
        os.remove(temp_path)

    res = results[0]
    title_text, author_text = "", ""
    title_score, author_score = -1e9, -1e9

    img_boxes = img_cv.copy()
    if res.boxes is not None:
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, min(W - 1, x1))
            x2 = max(0, min(W - 1, x2))
            y1 = max(0, min(H - 1, y1))
            y2 = max(0, min(H - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(img_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            crop = img_cv[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            ocr_texts = reader.readtext(crop, detail=0, paragraph=True)
            if not ocr_texts:
                continue
            text = " ".join([t for t in ocr_texts if isinstance(t, str)]).strip()
            if len(text) < 2:
                continue
            label, score = classify_text(text)
            if label == "title" and score > title_score:
                title_text, title_score = text, score
            elif label == "author" and score > author_score:
                author_text, author_score = text, score

    if not title_text:
        title_text = "لم يتم العثور على عنوان"
    if not author_text:
        author_text = "لم يتم العثور على مؤلف"

    img_boxes_pil = Image.fromarray(cv2.cvtColor(img_boxes, cv2.COLOR_BGR2RGB))
    return title_text, author_text, img_boxes_pil

def get_yolo_crops_for_dewey(img_bgr, results, max_crops=5):
    h, w = img_bgr.shape[:2]
    crops = []
    if not results or len(results) == 0:
        return [img_bgr]
    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return [img_bgr]
    boxes = res.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-areas)
    for idx in order[:max_crops]:
        x1, y1, x2, y2 = boxes[idx]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        if x2 > x1 and y2 > y1:
            crop = img_bgr[y1:y2, x1:x2]
            crops.append(crop)
    if not crops:
        crops.append(img_bgr)
    return crops

def process_dewey(image: Image.Image):
    """صورة رقم ديوي → رقم + تصنيف باستخدام YOLO ديوي + EasyOCR."""
    if not models["yolo_dewey"] or not models["reader"]:
        return "700.5", "الفنون والتسلية"

    yolo_model = models["yolo_dewey"]
    reader     = models["reader"]

    img = image.convert("RGB")
    img_np = np.array(img)

    if len(img_np.shape) == 3 and img_np.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    elif len(img_np.shape) == 3 and img_np.shape[2] == 1:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif len(img_np.shape) == 3:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_np

    try:
        results = yolo_model(img_bgr, imgsz=640, conf=0.25, verbose=False)
        crops = get_yolo_crops_for_dewey(img_bgr, results, max_crops=5)

        all_text = []
        best_num = ""
        best_len = 0
        for crop in crops:
            out = reader.readtext(crop, detail=0)
            if isinstance(out, list):
                all_text.extend(out)
                joined = " ".join(map(str, out))
                cand = extract_dewey(joined)
                if len(cand) > best_len:
                    best_len = len(cand)
                    best_num = cand

        if not best_num:
            best_num = "700.5"

        dewey_class = get_dewey_class(best_num)
        return best_num, dewey_class
    except Exception as e:
        return f"خطأ: {e}", "غير محدد"

def process_barcode(image: Image.Image):
    """صورة باركود → أرقام الباركود + اسم المكتبة باستخدام YOLO الباركود + تيسراكت."""
    if not models["yolo_barcode"] or not pytesseract_available:
        dummy = "0123456789"
        return dummy, get_library_from_barcode(dummy), None

    yolo_model = models["yolo_barcode"]

    img = image.convert("RGB")
    img_np = np.array(img)

    try:
        results = yolo_model(img_np)
    except Exception as e:
        return f"خطأ في YOLO: {e}", "غير محدد", None

    barcode_text = ""
    img_boxes = img_np.copy()

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            cv2.rectangle(img_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
            roi = img_np[y:y + h, x:x + w]
            if roi.size <= 0:
                continue
            try:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            except Exception:
                roi_rgb = roi
            pil_image = Image.fromarray(roi_rgb)
            try:
                raw_text = pytesseract.image_to_string(
                    pil_image,
                    config="--psm 6 --oem 3 -l eng"
                )
                digits = re.sub(r"[^0-9]", "", raw_text).strip()
                if digits:
                    barcode_text = digits
                    break
            except Exception:
                continue
        if barcode_text:
            break

    if not barcode_text:
        barcode_text = "0123456789"

    library = get_library_from_barcode(barcode_text)
    img_boxes_pil = Image.fromarray(cv2.cvtColor(img_boxes, cv2.COLOR_BGR2RGB))
    return barcode_text, library, img_boxes_pil

# ======================================
# Session State
# ======================================
if "books_data" not in st.session_state:
    st.session_state.books_data = []

if "current_book" not in st.session_state:
    st.session_state.current_book = {
        "title": "",
        "author": "",
        "barcode1": "",
        "barcode2": "",
        "dewey": "",
        "dewey_class": "",
        "library": "",
        "language": "",
        "condition": "",
        "year": "",
        "publisher": ""
    }

# ======================================
# الهيدر (شعار + اسم النظام)
# ======================================
st.markdown(f"""
<div class="main-header">
    <div class="header-title">
        <h1>المفهرس الذكي</h1>
    </div>
    <img src="data:image/png;base64,{logo_base64}" 
         class="header-logo" alt="شعار المكتبة">
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ======================================
# الشريط الجانبي
# ======================================
with st.sidebar:
    st.header("حالة النماذج")
    st.success("YOLO الغلاف جاهز")   if models["yolo_cover"]   else st.error("YOLO الغلاف غير متوفر")
    st.success("YOLO ديوي جاهز")     if models["yolo_dewey"]   else st.error("YOLO ديوي غير متوفر")
    st.success("YOLO الباركود جاهز") if models["yolo_barcode"] else st.error("YOLO الباركود غير متوفر")
    st.success("EasyOCR جاهز")       if models["reader"]       else st.error("EasyOCR غير متوفر")
    st.success("Tesseract جاهز")     if pytesseract_available  else st.error("Tesseract غير متوفر")
    st.markdown("---")
    st.metric("عدد الكتب المسجلة", len(st.session_state.books_data))

# ======================================
# التبويبات الخمسة
# ======================================

# --- تبويبات كمربعات أفقية بيضاء ---
tab_names = ["الغلاف", "ديوي", "الباركود", "السجل", "البيانات"]
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0
cols = st.columns(len(tab_names))
for i, name in enumerate(tab_names):
    if cols[i].button(name, key=f"tab_{i}", help=f"انتقل إلى {name}",
                      use_container_width=True,
                      type="primary" if st.session_state.active_tab == i else "secondary",
                      ):  # type="primary"/"secondary" متوفرة في streamlit>=1.25
        st.session_state.active_tab = i

# --- محتوى كل تبويب ---
if st.session_state.active_tab == 0:
    tab_cover = True
    tab_dewey = tab_barcode = tab_form = tab_data = False
elif st.session_state.active_tab == 1:
    tab_dewey = True
    tab_cover = tab_barcode = tab_form = tab_data = False
elif st.session_state.active_tab == 2:
    tab_barcode = True
    tab_cover = tab_dewey = tab_form = tab_data = False
elif st.session_state.active_tab == 3:
    tab_form = True
    tab_cover = tab_dewey = tab_barcode = tab_data = False
else:
    tab_data = True
    tab_cover = tab_dewey = tab_barcode = tab_form = False


# -------- تبويب الغلاف --------
if tab_cover:
    st.subheader("قراءة العنوان والمؤلف من غلاف الكتاب")

    src = st.radio("مصدر الصورة", ["رفع صورة", "الكاميرا"], horizontal=True, key="cover_src")
    img_file = None
    if src == "رفع صورة":
        img_file = st.file_uploader("صورة الغلاف", type=["jpg", "jpeg", "png"], key="cover_file")
    else:
        img_file = st.camera_input("التقاط صورة للغلاف", key="cover_cam")

    if img_file:
        image = Image.open(img_file)
        st.image(image, width=300)
        if st.button("استخراج العنوان والمؤلف", key="btn_cover"):
            with st.spinner("جاري التحليل..."):
                title, author, img_boxes = process_book_cover(image)
                st.session_state.current_book["title"] = title
                st.session_state.current_book["author"] = author
                st.success(f"العنوان: {title}")
                st.success(f"المؤلف: {author}")
                if img_boxes is not None:
                    st.image(img_boxes, caption="المربعات المكتشفة (YOLO)", width=300)

# -------- تبويب ديوي --------
if tab_dewey:
    st.subheader(" قراءة رقم ديوي وتحديد التصنيف")

    src = st.radio("مصدر الصورة", ["رفع صورة", "الكاميرا"], horizontal=True, key="dewey_src")
    img_file = None
    if src == "رفع صورة":
        img_file = st.file_uploader("صورة رقم ديوي", type=["jpg", "jpeg", "png"], key="dewey_file")
    else:
        img_file = st.camera_input("التقاط صورة لرقم ديوي", key="dewey_cam")

    if img_file:
        image = Image.open(img_file)
        st.image(image, width=300)
        if st.button("قراءة رقم ديوي", key="btn_dewey"):
            with st.spinner("جاري التحليل..."):
                dewey, d_class = process_dewey(image)
                st.session_state.current_book["dewey"] = dewey
                st.session_state.current_book["dewey_class"] = d_class
                st.success(f"رقم ديوي: {dewey}")
                st.success(f"التصنيف: {d_class}")

# -------- تبويب الباركود --------
if tab_barcode:
    st.subheader(" قراءة الباركود وتحديد المكتبة")
    target = st.radio("الباركود المستهدف", ["باركود 1", "باركود 2"], horizontal=True, key="barcode_target")
    src = st.radio("مصدر الصورة", ["رفع صورة", "الكاميرا"], horizontal=True, key="barcode_src")
    img_file = None
    if src == "رفع صورة":
        img_file = st.file_uploader("صورة الباركود", type=["jpg", "jpeg", "png"], key="barcode_file")
    else:
        img_file = st.camera_input("التقاط صورة للباركود", key="barcode_cam")

    if img_file:
        image = Image.open(img_file)
        st.image(image, width=300)
        if st.button("قراءة الباركود", key="btn_barcode"):
            with st.spinner("جاري التحليل..."):
                barcode, library, img_boxes = process_barcode(image)
                if target == "باركود 1":
                    st.session_state.current_book["barcode1"] = barcode
                else:
                    st.session_state.current_book["barcode2"] = barcode
                st.session_state.current_book["library"] = library
                st.success(f"الباركود: {barcode}")
                st.success(f"المكتبة: {library}")
                if img_boxes is not None:
                    st.image(img_boxes, caption="المربعات المكتشفة (YOLO)", width=300)
    else:
        st.info("اختاري/التقطي صورة للباركود.")

# -------- تبويب السجل (التسجيل اليدوي) --------
if tab_form:
    st.subheader("تسجيل بيانات الكتاب في السجل")

    with st.form("book_form"):
        c1, c2 = st.columns(2)
        with c1:
            title    = st.text_input("العنوان",   st.session_state.current_book["title"])
            author   = st.text_input("المؤلف",    st.session_state.current_book["author"])
            barcode1 = st.text_input("باركود 1",  st.session_state.current_book["barcode1"])
            barcode2 = st.text_input("باركود 2",  st.session_state.current_book["barcode2"])
            dewey    = st.text_input("رقم ديوي",  st.session_state.current_book["dewey"])
            publisher = st.text_input("دار النشر", st.session_state.current_book.get("publisher", ""))

        with c2:
            dewey_class = st.text_input("تصنيف ديوي", st.session_state.current_book["dewey_class"])
            library_options = ["", "المكتبة الرئيسية", "مكتبة اليافعين", "مكتبة الطفل"]
            current_lib = st.session_state.current_book["library"]
            idx = library_options.index(current_lib) if current_lib in library_options else 0
            library = st.selectbox("المكتبة", library_options, index=idx)

            languages_list = [
                "العربية", "الإنجليزية", "الفرنسية", "الألمانية", "الإسبانية", "التركية", "الأوردو", "الفارسية", "الصينية", "اليابانية", "أخرى"
            ]
            language = st.selectbox(
                "اختر لغة الكتاب",
                languages_list,
                index=languages_list.index(st.session_state.current_book.get("language", "العربية")) if st.session_state.current_book.get("language", "العربية") in languages_list else len(languages_list)-1
            )

            condition = st.selectbox(
                "حالة الكتاب",
                ["", "سليم", "غير مقبول", "لا يوجد له بيانات"],
                index=0
            )

            year = st.number_input("سنة النشر", 1990, 2030, 2024)

        b1, b2 = st.columns(2)
        submit = b1.form_submit_button("إضافة إلى الجدول", use_container_width=True)
        clear  = b2.form_submit_button("مسح الحقول",       use_container_width=True)

        if submit:
            if not title or not author or not barcode1 or not dewey or not dewey_class or not library or not condition:
                st.error("الرجاء تعبئة الحقول الأساسية قبل الإضافة.")
            else:
                book = {
                    "العنوان":      title,
                    "المؤلف":       author,
                    "باركود 1":     barcode1,
                    "باركود 2":     barcode2,
                    "رقم ديوي":     dewey,
                    "تصنيف ديوي":   dewey_class,
                    "المكتبة":      library,
                    "اللغة":        language,
                    "دار النشر":    publisher,
                    "حالة الكتاب":  condition,
                    "سنة النشر":    int(year),
                    "وقت التسجيل":  datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                st.session_state.books_data.append(book)
                st.session_state.current_book = {k: "" for k in st.session_state.current_book}
                st.success("تمت إضافة الكتاب إلى الجدول.")

        if clear:
            st.session_state.current_book = {k: "" for k in st.session_state.current_book}
            st.info("تم مسح الحقول.")

# -------- تبويب البيانات --------
if tab_data:
    st.subheader(" البيانات المسجلة")

    if st.session_state.books_data:
        df = pd.DataFrame(st.session_state.books_data)
        df.insert(0, "رقم", range(1, len(df) + 1))
        st.dataframe(df, use_container_width=True, height=420)

        c1, c2 = st.columns(2)
        csv_data = df.to_csv(index=False, encoding="utf-8-sig")
        c1.download_button(
            " تصدير إلى CSV",
            csv_data,
            file_name=f"books_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # تصدير إلى Excel
        import io
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Books')
        xlsx_data = output.getvalue()
        c1.download_button(
            " تصدير إلى Excel",
            xlsx_data,
            file_name=f"books_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        if c2.button("مسح جميع البيانات", use_container_width=True):
            st.session_state.books_data = []
            st.info("تم مسح جميع البيانات.")

        st.markdown("---")
        st.subheader(" إحصائيات سريعة")
        s1, s2, s3 = st.columns(3)
        s1.metric("إجمالي الكتب", len(df))
        if "المكتبة" in df and not df["المكتبة"].isna().all():
            s2.metric("أكثر مكتبة", df["المكتبة"].value_counts().index[0])
        if "تصنيف ديوي" in df and not df["تصنيف ديوي"].isna().all():
            s3.metric(" أكثر تصنيف", df["تصنيف ديوي"].value_counts().index[0])
    else:
        st.warning("لا توجد بيانات مسجلة حتى الآن. عند إضافة كتب ستظهر هنا.")

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#d0daff'>🌟 by Rahaf © 2025 | مكتبة حائل العامة</div>",
    unsafe_allow_html=True
)

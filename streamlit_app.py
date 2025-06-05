import streamlit as st
from PIL import Image, ImageDraw
import fitz  # PyMuPDF
import io
import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

st.set_page_config(layout="wide")
st.title("図面帯カットくん｜不動産営業の即戦力")
APP_VERSION = "v1.2.1"
st.markdown(f"#### 🏷️ バージョン: {APP_VERSION}")

st.markdown("📎 **PDFや画像をアップして、テンプレに図面を合成 → 高画質PDF出力できます！**")
st.markdown("🖼 **テンプレ画像は赤い四角の部分に自動で貼り付けられます（赤は合成後自動で消去）**")
st.markdown("⚠️ **テンプレ画像は300DPI以上推奨！印刷が綺麗になります。**")

uploaded_pdf = st.file_uploader("図面PDF または 画像", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=False)
uploaded_template = st.file_uploader("テンプレ画像（PNG）", type=["png"])

def auto_detect_drawing_area(image: Image.Image):
    np_img = np.array(image.convert("L"))
    edges = cv2.Canny(np_img, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = np_img.shape
    band_box = None
    max_area = 0
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if y > h * 0.6 and bw > w * 0.8 and area > max_area:
            band_box = (x, y, x + bw, y + bh)
            max_area = area
    if band_box:
        bx1, by1, bx2, by2 = band_box
        return (0, 0, w, by1)
    return (0, 0, w, h)

def find_red_area(template_img: Image.Image):
    img = template_img.convert("RGB")
    data = img.load()
    w, h = img.size
    min_x, min_y, max_x, max_y = w, h, 0, 0
    found = False
    for y in range(h):
        for x in range(w):
            r, g, b = data[x, y]
            if r > 200 and g < 60 and b < 60:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
                found = True
    if found:
        return (min_x, min_y, max_x + 1, max_y + 1)
    return None

def remove_red_area(template_img: Image.Image, red_area, fill=(255,255,255,255)):
    img = template_img.copy().convert("RGBA")
    x1, y1, x2, y2 = red_area
    for y in range(y1, y2):
        for x in range(x1, x2):
            r, g, b, *a = img.getpixel((x, y))
            if r > 200 and g < 60 and b < 60:
                img.putpixel((x, y), fill)
    return img

def generate_pdf(cropped: Image.Image, template: Image.Image):
    red_area = find_red_area(template)
    if red_area is None:
        return None
    x1, y1, x2, y2 = red_area
    area_w, area_h = x2 - x1, y2 - y1
    crop_w, crop_h = cropped.size
    scale = min(area_w / crop_w, area_h / crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    resized_crop = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    paste_x = x1 + (area_w - new_w) // 2
    paste_y = y1 + (area_h - new_h) // 2

    cleared_template = remove_red_area(template, red_area, fill=(255,255,255,255))
    combined = cleared_template.copy()
    combined.alpha_composite(resized_crop.convert("RGBA"), (paste_x, paste_y))

    img_buffer = io.BytesIO()
    combined = combined.convert("RGB")
    combined.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=(A4[1], A4[0]))
    c.drawImage(ImageReader(img_buffer), 0, 0, width=A4[1], height=A4[0])
    c.showPage()
    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer

if uploaded_pdf and uploaded_template:
    with st.spinner("処理中..."):
        # PDF or 画像読込
        if uploaded_pdf.name.lower().endswith(".pdf"):
            doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        else:
            img = Image.open(uploaded_pdf).convert("RGB")

        # 帯自動認識 → デフォ値設定
        selected_area = auto_detect_drawing_area(img)
        auto_x, auto_y, auto_x2, auto_y2 = selected_area
        auto_w, auto_h = auto_x2 - auto_x, auto_y2 - auto_y

        st.subheader("【1】帯認識・手動微調整")
        manual_mode = st.checkbox("手動で範囲を指定（自動認識値が初期値です。必要なら微調整してOK）")
        if manual_mode:
            x = st.number_input("X座標", min_value=0, max_value=img.width-1, value=auto_x)
            y = st.number_input("Y座標", min_value=0, max_value=img.height-1, value=auto_y)
            w = st.number_input("幅", min_value=1, max_value=img.width-x, value=auto_w)
            h = st.number_input("高さ", min_value=1, max_value=img.height-y, value=auto_h)
            cropped = img.crop((x, y, x + w, y + h))
        else:
            cropped = img.crop(selected_area)
        st.image(cropped, caption="図面トリミング範囲プレビュー", use_column_width=True)

        # 【2】塗りつぶし（色ピッカー＋スポイト仮）
        st.subheader("【2】画像の一部を塗りつぶす（ロゴ・社名隠し等）")
        fill_mode = st.checkbox("塗りつぶしON")
        fill_img = cropped.copy()
        if fill_mode:
            fx = st.number_input("塗りつぶし：左上X", min_value=0, max_value=fill_img.width-1, value=0, key="fx")
            fy = st.number_input("塗りつぶし：左上Y", min_value=0, max_value=fill_img.height-1, value=0, key="fy")
            fw = st.number_input("塗りつぶし：幅", min_value=1, max_value=fill_img.width-fx, value=50, key="fw")
            fh = st.number_input("塗りつぶし：高さ", min_value=1, max_value=fill_img.height-fy, value=20, key="fh")
            color_pick = st.color_picker("塗りつぶし色（スポイトで拾いたい色は一度画像上で確認→カラピで選択）", "#FFFFFF")
            if st.button("塗りつぶし実行"):
                draw = ImageDraw.Draw(fill_img)
                rgb = tuple(int(color_pick.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                draw.rectangle([fx, fy, fx+fw, fy+fh], fill=rgb)
                st.image(fill_img, caption="塗りつぶし後プレビュー", use_column_width=True)
                cropped = fill_img

        # 【3】PDF出力
        st.subheader("【3】PDF保存")
        template = Image.open(uploaded_template).convert("RGBA")
        result_pdf = generate_pdf(cropped, template)
        if result_pdf:
            st.success("PDF出力完了！以下からダウンロードできます：")
            st.download_button("📄 PDFをダウンロード", data=result_pdf, file_name="zumen_output.pdf", mime="application/pdf")
        else:
            st.error("テンプレ画像に赤枠が見つかりませんでした。")

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
import io
import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide")
st.title("図面帯カットくん｜不動産営業の即戦力")
APP_VERSION = "v1.3.3"
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

def draw_grid(image: Image.Image, grid_step=100, color=(0, 255, 0), width=2, label_color=(255,0,0)):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=28)
    except Exception:
        font = None
    for x in range(0, w, grid_step):
        draw.line([(x, 0), (x, h)], fill=color, width=width)
        draw.text((x+4, 8), str(x), fill=label_color, font=font)
    for y in range(0, h, grid_step):
        draw.line([(0, y), (w, y)], fill=color, width=width)
        draw.text((8, y+4), str(y), fill=label_color, font=font)
    if font:
        draw.text((w//2-40, 8), "横軸(px)", fill=(0,0,255), font=font)
        draw.text((8, h//2-20), "縦軸(px)", fill=(0,0,255), font=font)
    else:
        draw.text((w//2-40, 8), "横軸(px)", fill=(0,0,255))
        draw.text((8, h//2-20), "縦軸(px)", fill=(0,0,255))
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
        # 画像読込
        if uploaded_pdf.name.lower().endswith(".pdf"):
            doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            original_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        else:
            original_img = Image.open(uploaded_pdf).convert("RGB")

        # 軽量プレビュー画像生成
        PREVIEW_WIDTH = 800
        preview_img = original_img.resize(
            (PREVIEW_WIDTH, int(original_img.height * PREVIEW_WIDTH / original_img.width)),
            Image.LANCZOS
        )

        # 操作モード選択
        mode = st.radio("操作モードを選択", ["帯範囲指定", "塗りつぶし"])
        if 'coords' not in st.session_state or st.session_state.get('last_mode') != mode:
            st.session_state['coords'] = []
            st.session_state['last_mode'] = mode

        # プレビュー画像で2点クリック
        click = streamlit_image_coordinates(np.array(preview_img), key="preview")
        st.image(preview_img, caption="プレビュー画像（2点クリックで範囲指定）", width=PREVIEW_WIDTH)

        # クリック処理
        if click and len(st.session_state['coords']) < 2:
            st.session_state['coords'].append((click['x'], click['y']))
            st.info(f"{len(st.session_state['coords'])}点目: X={click['x']} Y={click['y']}")

        # 2点揃ったら範囲を赤枠でプレビュー
        if len(st.session_state['coords']) == 2:
            (x1, y1), (x2, y2) = st.session_state['coords']
            left, right = min(x1, x2), max(x1, x2)
            top, bottom = min(y1, y2), max(y1, y2)
            preview_draw = preview_img.copy()
            draw = ImageDraw.Draw(preview_draw)
            draw.rectangle([left, top, right, bottom], outline=(255,0,0), width=3)
            st.image(preview_draw, caption="選択範囲プレビュー（赤枠）", width=PREVIEW_WIDTH)

            # 元画像座標に変換
            real_left = int(left * original_img.width / PREVIEW_WIDTH)
            real_right = int(right * original_img.width / PREVIEW_WIDTH)
            real_top = int(top * original_img.height / preview_img.height)
            real_bottom = int(bottom * original_img.height / preview_img.height)

            if mode == "帯範囲指定":
                cropped = original_img.crop((0, min(real_top, real_bottom), original_img.width, max(real_top, real_bottom)))
                st.success("この範囲がテンプレートにコピーされます。")
            elif mode == "塗りつぶし":
                color_pick = st.session_state.get('last_color_pick', "#FFFFFF")
                color_pick = st.color_picker("塗りつぶし色（2点目クリックでスポイト／ここで変更も可）", color_pick)
                st.session_state['last_color_pick'] = color_pick
                fill_img = original_img.copy()
                draw = ImageDraw.Draw(fill_img)
                draw.rectangle([real_left, real_top, real_right, real_bottom], fill=color_pick)
                st.image(fill_img, caption="塗りつぶし後プレビュー", width=PREVIEW_WIDTH)
                cropped = fill_img

            if st.button("範囲リセット"):
                st.session_state['coords'] = []

            # PDF出力
            st.subheader("【PDF保存】")
            template = Image.open(uploaded_template).convert("RGBA")
            result_pdf = generate_pdf(cropped, template)
            if result_pdf:
                st.success("PDF出力完了！以下からダウンロードできます：")
                st.download_button("📄 PDFをダウンロード", data=result_pdf, file_name="zumen_output.pdf", mime="application/pdf")
            else:
                st.error("テンプレ画像に赤枠が見つかりませんでした。")
        else:
            st.info("2点クリックで範囲を指定してください。")

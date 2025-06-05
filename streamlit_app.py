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
APP_VERSION = "v1.2.6"
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

        # プレビューサイズ調整スライダー
        preview_width = st.slider("プレビュー画像の幅(px)", min_value=300, max_value=1200, value=600, step=50)

        st.subheader("【1】帯認識・手動微調整")
        manual_mode = st.checkbox("手動で範囲を指定（クリックで座標取得も可能）")
        if manual_mode:
            cropped = img.crop((auto_x, auto_y, auto_x + auto_w, auto_y + auto_h))
            grid_img = draw_grid(cropped, grid_step=100)
            st.write("下の画像をクリックすると、範囲指定の左上座標に反映されます。")
            # 画像クリックで座標取得
            coords = streamlit_image_coordinates(np.array(grid_img), key="manual_select")
            if coords is not None:
                mx, my = int(coords["x"]), int(coords["y"])
                st.info(f"クリック座標: 横位置={mx}, 縦位置={my}")
            else:
                mx, my = auto_x, auto_y

            x = st.number_input("範囲の左上 横位置（px）", min_value=0, max_value=img.width-1, value=mx, key="manual_x")
            y = st.number_input("範囲の左上 縦位置（px）", min_value=0, max_value=img.height-1, value=my, key="manual_y")
            w = st.number_input("幅（px）", min_value=1, max_value=img.width-x, value=auto_w, key="manual_w")
            h = st.number_input("高さ（px）", min_value=1, max_value=img.height-y, value=auto_h, key="manual_h")
            cropped = img.crop((x, y, x + w, y + h))
            grid_img = draw_grid(cropped, grid_step=100)
            st.image(grid_img, caption="手動選択範囲プレビュー（100pxごとに目安線・軸ラベル付き）", width=preview_width)
            st.success("この範囲でPDF生成可能！")
        else:
            cropped = img.crop(selected_area)
            grid_img = draw_grid(cropped, grid_step=100)
            st.image(grid_img, caption="自動認識範囲プレビュー（100pxごとに目安線・軸ラベル付き）", width=preview_width)

        # 【2】塗りつぶし（スポイト対応）
        st.subheader("【2】画像の一部を塗りつぶす（画像クリックで色取得も可能）")
        fill_mode = st.checkbox("塗りつぶしON")
        fill_img = cropped.copy()
        color_pick = "#FFFFFF"
        st.write("下の画像をクリックすると、その位置の色がカラーピッカーに反映されます。")
        coords2 = streamlit_image_coordinates(np.array(fill_img), key="fill_select")
        if coords2 is not None:
            fx, fy = int(coords2["x"]), int(coords2["y"])
            rgb = fill_img.getpixel((fx, fy))
            color_pick = '#%02x%02x%02x' % rgb
            st.info(f"クリック座標: 横位置={fx}, 縦位置={fy} / 色: {color_pick}")
        fx = st.number_input("塗りつぶし開始 横位置（px）", min_value=0, max_value=fill_img.width-1, value=0, key="fx")
        fy = st.number_input("塗りつぶし開始 縦位置（px）", min_value=0, max_value=fill_img.height-1, value=0, key="fy")
        fw = st.number_input("塗りつぶし 幅（px）", min_value=1, max_value=fill_img.width-fx, value=50, key="fw")
        fh = st.number_input("塗りつぶし 高さ（px）", min_value=1, max_value=fill_img.height-fy, value=20, key="fh")
        color_pick = st.color_picker("塗りつぶし色（画像クリックでスポイト）", color_pick)
        if fill_mode and st.button("塗りつぶし実行"):
            draw = ImageDraw.Draw(fill_img)
            rgb = tuple(int(color_pick.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            draw.rectangle([fx, fy, fx+fw, fy+fh], fill=rgb)
            st.image(fill_img, caption="塗りつぶし後プレビュー", use_container_width=True)
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

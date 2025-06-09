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
from datetime import datetime

st.set_page_config(layout="wide")
st.title("図面帯カットくん｜不動産営業の即戦力")
APP_VERSION = "v1.4.5"
st.markdown(f"#### 🏷️ バージョン: {APP_VERSION}")

st.markdown("📎 **PDFや画像をアップして、テンプレに図面を合成 → 高画質PDF出力できます！**")
st.markdown("🖼 **テンプレ画像は赤い四角の部分に自動で貼り付けられます（赤は合成後自動で消去）**")
st.markdown("⚠️ **テンプレ画像は300DPI以上推奨！印刷が綺麗になります。**")

# セッションステートの初期化
def init_session_state():
    """セッションステートを初期化"""
    defaults = {
        'processed_image': None,
        'original_image': None,
        'preview_image': None,
        'auto_detected_area': None,
        'current_mode': 'auto',
        'manual_coords': [],
        'fill_areas': [],
        'processing_step': 'upload',
        'last_uploaded_file': None,
        'template_image': None,
        'eyedropper_mode': False,
        'property_name': '',
        'property_price': ''
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

@st.cache_data
def load_and_process_image(file_data, file_name):
    """画像を読み込んで処理（キャッシュ機能付き）"""
    try:
        if file_name.lower().endswith(".pdf"):
            doc = fitz.open(stream=file_data, filetype="pdf")
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            original_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
        else:
            original_img = Image.open(io.BytesIO(file_data)).convert("RGB")
        
        # 画像サイズの検証
        if original_img.width == 0 or original_img.height == 0:
            raise ValueError("無効な画像サイズです")
        
        # プレビュー画像生成（800px幅に統一）
        PREVIEW_WIDTH = 800
        aspect_ratio = original_img.height / original_img.width
        preview_height = int(PREVIEW_WIDTH * aspect_ratio)
        preview_img = original_img.resize((PREVIEW_WIDTH, preview_height), Image.LANCZOS)
        
        return original_img, preview_img
    except Exception as e:
        st.error(f"画像の読み込みに失敗しました: {str(e)}")
        return None, None

def auto_detect_drawing_area(image: Image.Image):
    """図面領域を自動検出（改良版）"""
    try:
        np_img = np.array(image.convert("L"))
        
        # エッジ検出のパラメータを調整
        edges = cv2.Canny(np_img, 50, 150)
        
        # ノイズ除去
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = np_img.shape
        
        # 帯領域候補を検出
        band_candidates = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            
            # 条件を調整：下部60%以下、幅が70%以上、高さが画像の5%以上
            if (y > h * 0.6 and 
                bw > w * 0.7 and 
                bh > h * 0.05 and 
                area > 1000):
                band_candidates.append((x, y, x + bw, y + bh, area))
        
        if band_candidates:
            # 最も大きな帯を選択
            band_box = max(band_candidates, key=lambda x: x[4])
            bx1, by1, bx2, by2, _ = band_box
            # 帯より上の部分を図面領域とする
            detected_area = (0, 0, w, by1)
        else:
            # 帯が見つからない場合は画像全体の80%を図面領域とする
            detected_area = (0, 0, w, int(h * 0.8))
        
        # 検出された領域の妥当性をチェック
        x1, y1, x2, y2 = detected_area
        if x2 <= x1 or y2 <= y1:
            # 無効な領域の場合、画像全体の80%にフォールバック
            return (0, 0, w, int(h * 0.8))
        
        return detected_area
    except Exception as e:
        st.error(f"自動検出中にエラーが発生しました: {str(e)}")
        # エラー時は画像全体の80%を返す
        h, w = image.height, image.width
        return (0, 0, w, int(h * 0.8))

def validate_area(area, image_width, image_height):
    """選択領域の妥当性をチェック"""
    if not area:
        return False
    
    x1, y1, x2, y2 = area
    
    # 座標が有効な範囲内にあるかチェック
    if (x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height):
        return False
    
    # 幅と高さが正の値かチェック
    if x2 <= x1 or y2 <= y1:
        return False
    
    # 最小サイズをチェック（10px以上）
    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return False
    
    return True

def find_red_area(template_img: Image.Image):
    """テンプレート内の赤い領域を検出"""
    try:
        img = template_img.convert("RGB")
        np_img = np.array(img)
        
        # 赤色の範囲をHSVで定義（より正確な検出）
        hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
        
        # 赤色の範囲（HSV）
        lower_red1 = np.array([0, 120, 120])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 120])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        
        # 輪郭を検出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 最大の赤い領域を取得
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            red_area = (x, y, x + w, y + h)
            
            # 赤い領域の妥当性をチェック
            if validate_area(red_area, template_img.width, template_img.height):
                return red_area
        
        return None
    except Exception as e:
        st.error(f"赤い領域の検出中にエラーが発生しました: {str(e)}")
        return None

def remove_red_area(template_img: Image.Image, red_area, fill=(255,255,255,255)):
    """テンプレートから赤い領域を除去"""
    try:
        img = template_img.copy().convert("RGBA")
        if red_area:
            x1, y1, x2, y2 = red_area
            # 赤い領域を白で塗りつぶし
            draw = ImageDraw.Draw(img)
            draw.rectangle([x1, y1, x2, y2], fill=fill)
        return img
    except Exception as e:
        st.error(f"赤い領域の除去中にエラーが発生しました: {str(e)}")
        return template_img

def draw_preview_with_area(image: Image.Image, area, color=(255, 0, 0), label="選択領域"):
    """プレビュー画像に選択領域を描画"""
    try:
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        if area and st.session_state.original_image:
            x1, y1, x2, y2 = area
            # 座標をプレビューサイズに調整
            scale_x = image.width / st.session_state.original_image.width
            scale_y = image.height / st.session_state.original_image.height
            
            px1 = int(x1 * scale_x)
            py1 = int(y1 * scale_y)
            px2 = int(x2 * scale_x)
            py2 = int(y2 * scale_y)
            
            # 枠線を描画
            draw.rectangle([px1, py1, px2, py2], outline=color, width=3)
            
            # ラベルを描画
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            draw.text((px1 + 5, py1 + 5), label, fill=color, font=font)
        
        return img_copy
    except Exception as e:
        st.error(f"プレビュー描画中にエラーが発生しました: {str(e)}")
        return image

def apply_fill_areas(image: Image.Image, fill_areas):
    """複数の塗りつぶし領域を適用"""
    try:
        result_img = image.copy()
        draw = ImageDraw.Draw(result_img)
        
        for area_info in fill_areas:
            if len(area_info) >= 5:
                x1, y1, x2, y2, color = area_info
                # 領域の妥当性をチェック
                if validate_area((x1, y1, x2, y2), image.width, image.height):
                    draw.rectangle([x1, y1, x2, y2], fill=color)
        
        return result_img
    except Exception as e:
        st.error(f"塗りつぶし処理中にエラーが発生しました: {str(e)}")
        return image

def safe_crop_image(image: Image.Image, area):
    """安全に画像をクロップする"""
    try:
        if not validate_area(area, image.width, image.height):
            st.error("無効な選択領域です。範囲を再選択してください。")
            return None
        
        x1, y1, x2, y2 = area
        cropped = image.crop((x1, y1, x2, y2))
        
        # クロップ結果の検証
        if cropped.width == 0 or cropped.height == 0:
            st.error("クロップした画像のサイズが無効です。")
            return None
        
        return cropped
    except Exception as e:
        st.error(f"画像のクロップ中にエラーが発生しました: {str(e)}")
        return None

def safe_resize_preview(image: Image.Image, target_width):
    """安全に画像をリサイズしてプレビューを作成"""
    try:
        if image.width == 0 or image.height == 0:
            st.error("無効な画像サイズです。")
            return None
        
        aspect_ratio = image.height / image.width
        preview_height = int(target_width * aspect_ratio)
        
        # 最小サイズのチェック
        if preview_height < 1:
            preview_height = 1
        
        preview = image.resize((target_width, preview_height), Image.LANCZOS)
        return preview
    except Exception as e:
        st.error(f"プレビュー画像の作成中にエラーが発生しました: {str(e)}")
        return None

def generate_pdf(cropped: Image.Image, template: Image.Image):
    """PDF生成（エラーハンドリング強化）"""
    try:
        # 入力画像の検証
        if cropped.width == 0 or cropped.height == 0:
            return None, "処理済み画像のサイズが無効です。"
        
        red_area = find_red_area(template)
        if red_area is None:
            return None, "テンプレート画像に赤い領域が見つかりませんでした。"
        
        x1, y1, x2, y2 = red_area
        area_w, area_h = x2 - x1, y2 - y1
        crop_w, crop_h = cropped.size
        
        # ゼロ除算を防ぐ
        if crop_w == 0 or crop_h == 0 or area_w == 0 or area_h == 0:
            return None, "画像またはテンプレート領域のサイズが無効です。"
        
        # アスペクト比を保持してリサイズ
        scale = min(area_w / crop_w, area_h / crop_h)
        new_w = max(1, int(crop_w * scale))
        new_h = max(1, int(crop_h * scale))
        resized_crop = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 中央配置
        paste_x = x1 + (area_w - new_w) // 2
        paste_y = y1 + (area_h - new_h) // 2

        # テンプレートから赤い領域を除去
        cleared_template = remove_red_area(template, red_area)
        
        # 画像を合成
        combined = cleared_template.copy()
        if resized_crop.mode != 'RGBA':
            resized_crop = resized_crop.convert('RGBA')
        combined.alpha_composite(resized_crop, (paste_x, paste_y))

        # PDF生成
        img_buffer = io.BytesIO()
        combined_rgb = combined.convert("RGB")
        combined_rgb.save(img_buffer, format="PNG", dpi=(300, 300))
        img_buffer.seek(0)

        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=(A4[1], A4[0]))
        c.drawImage(ImageReader(img_buffer), 0, 0, width=A4[1], height=A4[0])
        c.showPage()
        c.save()
        pdf_buffer.seek(0)
        
        return pdf_buffer, "PDF生成成功"
    
    except Exception as e:
        return None, f"PDF生成エラー: {str(e)}"

def generate_filename():
    """物件情報を基にファイル名を生成"""
    today = datetime.now().strftime("%y-%m-%d")
    filename_parts = [today]
    
    if st.session_state.property_name:
        filename_parts.append(st.session_state.property_name)
    if st.session_state.property_price:
        filename_parts.append(st.session_state.property_price)
    
    if len(filename_parts) == 1:  # 日付のみの場合
        filename_parts.append("zumen_output")
    
    return "_".join(filename_parts) + ".pdf"

# メインUI
col1, col2 = st.columns(2)

with col1:
    st.header("📄 図面ファイル")
    uploaded_pdf = st.file_uploader(
        "図面PDF または 画像をアップロード",
        type=["pdf", "png", "jpg", "jpeg"],
        help="処理したい図面ファイルを選択してください"
    )

with col2:
    st.header("🖼️ テンプレート画像")
    uploaded_template = st.file_uploader(
        "テンプレート画像（PNG推奨）",
        type=["png", "jpg", "jpeg"],
        help="赤い四角が描かれたテンプレート画像をアップロードしてください"
    )

# 物件情報入力セクション
st.header("📝 物件情報（ファイル名に使用）")
col_prop1, col_prop2 = st.columns(2)
with col_prop1:
    property_name = st.text_input("物件名", value=st.session_state.property_name, placeholder="例：山田マンション")
    if property_name != st.session_state.property_name:
        st.session_state.property_name = property_name

with col_prop2:
    property_price = st.text_input("価格", value=st.session_state.property_price, placeholder="例：3980万円")
    if property_price != st.session_state.property_price:
        st.session_state.property_price = property_price

# ファイル名プレビュー
if property_name or property_price:
    today = datetime.now().strftime("%y-%m-%d")
    filename_parts = [today]
    if property_name:
        filename_parts.append(property_name)
    if property_price:
        filename_parts.append(property_price)
    preview_filename = "_".join(filename_parts) + ".pdf"
    st.info(f"📄 保存ファイル名プレビュー: {preview_filename}")

if uploaded_pdf and uploaded_template:
    # ファイルが変更された場合のみ処理
    file_changed = (st.session_state.last_uploaded_file != uploaded_pdf.name)
    
    if file_changed or st.session_state.original_image is None:
        with st.spinner("画像を読み込み中..."):
            file_data = uploaded_pdf.read()
            original_img, preview_img = load_and_process_image(file_data, uploaded_pdf.name)
            
            if original_img is None or preview_img is None:
                st.error("画像の読み込みに失敗しました。ファイルを確認してください。")
                st.stop()
            
            try:
                template_img = Image.open(uploaded_template).convert("RGBA")
                if template_img.width == 0 or template_img.height == 0:
                    st.error("テンプレート画像のサイズが無効です。")
                    st.stop()
            except Exception as e:
                st.error(f"テンプレート画像の読み込みに失敗しました: {str(e)}")
                st.stop()
            
            st.session_state.original_image = original_img
            st.session_state.preview_image = preview_img
            st.session_state.template_image = template_img
            st.session_state.last_uploaded_file = uploaded_pdf.name
            st.session_state.processing_step = 'auto_detect'
            st.session_state.fill_areas = []
            st.session_state.eyedropper_mode = False
    
    # ステップ1: 自動帯認識
    if st.session_state.processing_step == 'auto_detect':
        with st.spinner("図面領域を自動検出中..."):
            st.session_state.auto_detected_area = auto_detect_drawing_area(st.session_state.original_image)
            st.session_state.processing_step = 'review_auto'
    
    # ステップ2: 自動検出結果のレビュー
    if st.session_state.processing_step == 'review_auto':
        st.subheader("🤖 自動検出結果")
        st.markdown("""
        **📋 説明**: 自動で図面領域（帯より上の部分）を検出しました。
        - **緑枠**: 検出された図面領域
        - この領域が適切であれば「✅この領域でOK」をクリック
        - 調整が必要な場合は「🔧手動で調整」をクリック
        - 塗りつぶしが必要な場合は「🎨塗りつぶしモード」をクリック
        """)
        
        # プレビュー画像に検出領域を描画
        preview_with_area = draw_preview_with_area(
            st.session_state.preview_image, 
            st.session_state.auto_detected_area,
            color=(0, 255, 0),
            label="自動検出された図面領域"
        )
        
        st.image(preview_with_area, caption="自動検出結果（緑枠が図面領域）", use_column_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ この領域でOK", type="primary"):
                # 図面領域を確定し、最終ステップへ
                st.session_state.confirmed_drawing_area = st.session_state.auto_detected_area
                cropped = safe_crop_image(st.session_state.original_image, st.session_state.auto_detected_area)
                if cropped is not None:
                    st.session_state.processed_image = cropped
                    st.session_state.processing_step = 'final'
                    st.rerun()
        
        with col2:
            if st.button("🔧 手動で調整"):
                st.session_state.processing_step = 'manual_adjust'
                st.session_state.manual_coords = []
                st.rerun()
        
        with col3:
            if st.button("🎨 塗りつぶしモード"):
                # 図面領域を確定してから塗りつぶしモードへ
                st.session_state.confirmed_drawing_area = st.session_state.auto_detected_area
                st.session_state.processing_step = 'fill_mode'
                st.session_state.manual_coords = []
                st.rerun()
    
    # ステップ3: 手動調整
    elif st.session_state.processing_step == 'manual_adjust':
        st.subheader("🔧 手動で図面領域を調整")
        st.markdown("""
        **📋 説明**: 図面領域を手動で調整します。
        - **操作方法**: 画像上で**左上の点**と**右下の点**を順番にクリック
        - **1点目**: 図面領域の左上角をクリック
        - **2点目**: 図面領域の右下角をクリック
        - 通常は帯（下部の枠）より上の部分を選択します
        """)
        
        # クリック座標取得
        coordinates = streamlit_image_coordinates(
            np.array(st.session_state.preview_image),
            key="manual_select"
        )
        
        if coordinates and len(st.session_state.manual_coords) < 2:
            st.session_state.manual_coords.append((coordinates['x'], coordinates['y']))
            if len(st.session_state.manual_coords) == 1:
                st.success(f"✅ 1点目を選択しました: X={coordinates['x']}, Y={coordinates['y']}")
                st.info("続けて2点目（右下角）をクリックしてください")
            else:
                st.success(f"✅ 2点目を選択しました: X={coordinates['x']}, Y={coordinates['y']}")
        
        # 2点が選択された場合
        if len(st.session_state.manual_coords) == 2:
            (x1, y1), (x2, y2) = st.session_state.manual_coords
            
            # プレビューサイズから元画像サイズに変換
            scale_x = st.session_state.original_image.width / st.session_state.preview_image.width
            scale_y = st.session_state.original_image.height / st.session_state.preview_image.height
            
            real_x1 = int(min(x1, x2) * scale_x)
            real_y1 = int(min(y1, y2) * scale_y)
            real_x2 = int(max(x1, x2) * scale_x)
            real_y2 = int(max(y1, y2) * scale_y)
            
            manual_area = (real_x1, real_y1, real_x2, real_y2)
            
            # 領域の妥当性をチェック
            if validate_area(manual_area, st.session_state.original_image.width, st.session_state.original_image.height):
                # プレビュー表示
                preview_with_manual = draw_preview_with_area(
                    st.session_state.preview_image,
                    manual_area,
                    color=(255, 0, 0),
                    label="手動選択領域"
                )
                st.image(preview_with_manual, caption="手動選択結果（赤枠が確定される図面領域）", use_column_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("✅ この領域で確定", type="primary"):
                        # 図面領域を確定
                        st.session_state.confirmed_drawing_area = manual_area
                        cropped = safe_crop_image(st.session_state.original_image, manual_area)
                        if cropped is not None:
                            st.session_state.processed_image = cropped
                            st.session_state.processing_step = 'final'
                            st.rerun()
                
                with col2:
                    if st.button("🎨 塗りつぶしモード"):
                        # 図面領域を確定してから塗りつぶしモードへ
                        st.session_state.confirmed_drawing_area = manual_area
                        st.session_state.processing_step = 'fill_mode'
                        st.session_state.manual_coords = []
                        st.rerun()
                
                with col3:
                    if st.button("🔄 やり直し"):
                        st.session_state.manual_coords = []
                        st.rerun()
            else:
                st.error("選択された領域が無効です。もう一度選択してください。")
                if st.button("🔄 リセット"):
                    st.session_state.manual_coords = []
                    st.rerun()
    
    # ステップ4: 塗りつぶしモード
    elif st.session_state.processing_step == 'fill_mode':
        st.subheader("🎨 塗りつぶしモード")
        
        # 確定された図面領域を表示
        if 'confirmed_drawing_area' not in st.session_state:
            st.warning("図面領域が確定されていません。")
            st.stop()
        
        dx1, dy1, dx2, dy2 = st.session_state.confirmed_drawing_area
        st.markdown(f"""
        **📋 説明**: 確定された図面領域内で塗りつぶしを行います。
        - **図面領域**: ({dx1}, {dy1}) から ({dx2}, {dy2})
        - **操作方法**: 図面領域内で**左上の点**と**右下の点**を順番にクリック
        - **1点目**: 塗りつぶし範囲の左上角をクリック
        - **2点目**: 塗りつぶし範囲の右下角をクリック
        - 複数の範囲を塗りつぶしできます
        """)
        
        # 塗りつぶし色選択
        col_color1, col_color2 = st.columns([1, 1])
        with col_color1:
            # スポイトで取得した色があれば使用、なければデフォルト
            default_color = st.session_state.get('selected_color', "#FFFFFF")
            fill_color = st.color_picker("塗りつぶし色", value=default_color)
        with col_color2:
            eyedropper_active = st.checkbox("🎨 スポイトツール", value=st.session_state.eyedropper_mode)
            if eyedropper_active != st.session_state.eyedropper_mode:
                st.session_state.eyedropper_mode = eyedropper_active
                st.session_state.manual_coords = []  # モード変更時は座標をリセット
                st.rerun()
        
        if st.session_state.eyedropper_mode:
            st.info("🎨 スポイトモード：画像をクリックして色を取得します")
        
        # 確定された図面領域をクロップした画像で作業
        drawing_area_image = st.session_state.original_image.crop(st.session_state.confirmed_drawing_area)
        
        # 塗りつぶし済み領域を適用
        if st.session_state.fill_areas:
            # 図面領域内の相対座標に変換して塗りつぶしを適用
            relative_fill_areas = []
            for area in st.session_state.fill_areas:
                fx1, fy1, fx2, fy2, color = area
                # 図面領域内の相対座標に変換
                rel_x1 = fx1 - dx1
                rel_y1 = fy1 - dy1
                rel_x2 = fx2 - dx1
                rel_y2 = fy2 - dy1
                # 図面領域内にクリップ
                rel_x1 = max(0, min(rel_x1, drawing_area_image.width))
                rel_y1 = max(0, min(rel_y1, drawing_area_image.height))
                rel_x2 = max(0, min(rel_x2, drawing_area_image.width))
                rel_y2 = max(0, min(rel_y2, drawing_area_image.height))
                if rel_x2 > rel_x1 and rel_y2 > rel_y1:
                    relative_fill_areas.append((rel_x1, rel_y1, rel_x2, rel_y2, color))
            
            drawing_area_image = apply_fill_areas(drawing_area_image, relative_fill_areas)
        
        # プレビュー画像生成
        current_preview = safe_resize_preview(drawing_area_image, 600)
        if current_preview is None:
            st.error("プレビュー画像の生成に失敗しました。")
            st.stop()
        
        # 現在の状態を表示（図面領域のみ）とクリック座標取得を統合
        coordinates = streamlit_image_coordinates(
            np.array(current_preview),
            key="fill_select"
        )
        
        # キャプションを動的に設定
        if st.session_state.eyedropper_mode:
            caption = f"🎨 スポイトモード：色を取得したい場所をクリック（塗りつぶし領域: {len(st.session_state.fill_areas)}個）"
        elif len(st.session_state.manual_coords) == 1:
            caption = f"📍 1点目設定済み：2点目（右下角）をクリックしてください（塗りつぶし領域: {len(st.session_state.fill_areas)}個）"
        else:
            caption = f"図面領域内の塗りつぶし状況（塗りつぶし領域: {len(st.session_state.fill_areas)}個）"
        
        if coordinates:
            if st.session_state.eyedropper_mode:
                # スポイトモード：クリックした位置の色を取得
                x, y = coordinates['x'], coordinates['y']
                if 0 <= x < current_preview.width and 0 <= y < current_preview.height:
                    # プレビュー画像から色を取得
                    pixel_color = current_preview.getpixel((x, y))
                    if len(pixel_color) == 3:  # RGB
                        r, g, b = pixel_color
                        hex_color = f"#{r:02x}{g:02x}{b:02x}"
                        # セッション状態を更新してcolor_pickerに反映
                        st.session_state.selected_color = hex_color
                        # スポイトツールを自動解除して範囲指定モードに移行
                        st.session_state.eyedropper_mode = False
                        # 色を取得した座標を1点目として自動設定
                        st.session_state.manual_coords = [(x, y)]
                        st.success(f"🎨 色を取得しました: RGB({r}, {g}, {b}) / {hex_color}")
                        st.info("✅ 1点目が自動設定されました。続けて2点目（右下角）をクリックしてください。")
                        st.rerun()
            elif len(st.session_state.manual_coords) < 2:
                # 通常の範囲選択モード
                st.session_state.manual_coords.append((coordinates['x'], coordinates['y']))
                if len(st.session_state.manual_coords) == 1:
                    st.success(f"✅ 1点目を選択しました: X={coordinates['x']}, Y={coordinates['y']}")
                    st.info("続けて2点目（右下角）をクリックしてください")
                else:
                    st.success(f"✅ 2点目を選択しました: X={coordinates['x']}, Y={coordinates['y']}")
        
        # 2点が選択された場合（通常の範囲選択モードのみ）
        if not st.session_state.eyedropper_mode and len(st.session_state.manual_coords) == 2:
            (x1, y1), (x2, y2) = st.session_state.manual_coords
            
            # 図面領域内の座標に変換
            scale_x = drawing_area_image.width / current_preview.width
            scale_y = drawing_area_image.height / current_preview.height
            
            rel_x1 = int(min(x1, x2) * scale_x)
            rel_y1 = int(min(y1, y2) * scale_y)
            rel_x2 = int(max(x1, x2) * scale_x)
            rel_y2 = int(max(y1, y2) * scale_y)
            
            # 図面領域内でクランプ
            rel_x1 = max(0, min(rel_x1, drawing_area_image.width))
            rel_y1 = max(0, min(rel_y1, drawing_area_image.height))
            rel_x2 = max(0, min(rel_x2, drawing_area_image.width))
            rel_y2 = max(0, min(rel_y2, drawing_area_image.height))
            
            # 元画像での絶対座標に変換
            abs_x1 = rel_x1 + dx1
            abs_y1 = rel_y1 + dy1
            abs_x2 = rel_x2 + dx1
            abs_y2 = rel_y2 + dy1
            
            # 最小サイズのチェック
            if rel_x2 > rel_x1 and rel_y2 > rel_y1 and (rel_x2 - rel_x1) >= 5 and (rel_y2 - rel_y1) >= 5:
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("✅ 塗りつぶし実行"):
                        # RGB値に変換
                        hex_color = fill_color.lstrip('#')
                        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                        
                        # 絶対座標で保存
                        st.session_state.fill_areas.append((abs_x1, abs_y1, abs_x2, abs_y2, rgb_color))
                        st.session_state.manual_coords = []
                        st.success("塗りつぶしを追加しました！")
                        st.rerun()
                
                with col2:
                    if st.button("🔄 範囲リセット"):
                        st.session_state.manual_coords = []
                        st.rerun()
                
                with col3:
                    if st.button("📋 塗りつぶし完了"):
                        # 図面領域をクロップして塗りつぶしを適用
                        filled_image = apply_fill_areas(st.session_state.original_image, st.session_state.fill_areas)
                        final_cropped = filled_image.crop(st.session_state.confirmed_drawing_area)
                        st.session_state.processed_image = final_cropped
                        st.session_state.processing_step = 'final'
                        st.rerun()
            else:
                st.error("選択された領域が小さすぎます。より大きな範囲を選択してください。")
                if st.button("🔄 リセット"):
                    st.session_state.manual_coords = []
                    st.rerun()
        

        
        # 塗りつぶし領域の管理
        if st.session_state.fill_areas:
            col_mgmt1, col_mgmt2 = st.columns(2)
            with col_mgmt1:
                if st.button("🗑️ 最後の塗りつぶしを削除"):
                    st.session_state.fill_areas.pop()
                    st.rerun()
            
            with col_mgmt2:
                if st.button("🧹 全塗りつぶしをクリア"):
                    st.session_state.fill_areas = []
                    st.rerun()
            
            # 塗りつぶし領域の一覧表示
            with st.expander("塗りつぶし領域の詳細"):
                for i, area in enumerate(st.session_state.fill_areas):
                    x1, y1, x2, y2, color = area
                    # 図面領域内の相対座標も表示
                    rel_x1 = x1 - dx1
                    rel_y1 = y1 - dy1
                    rel_x2 = x2 - dx1
                    rel_y2 = y2 - dy1
                    st.write(f"領域 {i+1}:")
                    st.write(f"  - 元画像座標: ({x1}, {y1}) - ({x2}, {y2})")
                    st.write(f"  - 図面内座標: ({rel_x1}, {rel_y1}) - ({rel_x2}, {rel_y2})")
                    st.write(f"  - 色: RGB{color}")
        
        # 直接PDF生成ボタンを追加
        if st.session_state.fill_areas:
            st.subheader("📄 PDF生成")
            st.info("塗りつぶしを適用した状態でPDFを生成してダウンロードします。")
            
            # PDF生成処理を自動実行
            with st.spinner("PDFを生成中..."):
                # 現在の塗りつぶし状態の画像を使用
                current_filled_image = apply_fill_areas(st.session_state.original_image, st.session_state.fill_areas)
                pdf_buffer, message = generate_pdf(current_filled_image, st.session_state.template_image)
                
                if pdf_buffer:
                    st.success("✅ PDF生成完了！下のボタンからダウンロードできます。")
                    st.download_button(
                        "📥 塗りつぶし状態のPDFをダウンロード",
                        data=pdf_buffer.getvalue(),
                        file_name=generate_filename(),
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True
                    )
                else:
                    st.error(message)
    
    # ステップ5: 最終確認とPDF生成
    elif st.session_state.processing_step == 'final':
        st.subheader("📄 最終確認とPDF生成")
        
        # 処理済み画像のプレビュー
        if st.session_state.processed_image:
            # 安全なプレビュー生成
            final_preview = safe_resize_preview(st.session_state.processed_image, 600)
            
            if final_preview is not None:
                st.image(final_preview, caption="処理済み図面（テンプレートに合成される部分）", use_column_width=True)
                
                # PDF生成処理を自動実行
                with st.spinner("PDFを生成中..."):
                    pdf_buffer, message = generate_pdf(st.session_state.processed_image, st.session_state.template_image)
                    
                    if pdf_buffer:
                        st.success("✅ PDF生成完了！下のボタンからダウンロードできます。")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                "📥 PDFをダウンロード",
                                data=pdf_buffer.getvalue(),
                                file_name=generate_filename(),
                                mime="application/pdf",
                                type="primary",
                                use_container_width=True
                            )
                        
                        with col2:
                            if st.button("🔙 最初からやり直し", use_container_width=True):
                                # セッションステートをリセット
                                for key in ['processed_image', 'processing_step', 'manual_coords', 'fill_areas', 'auto_detected_area', 'confirmed_drawing_area', 'eyedropper_mode']:
                                    if key in st.session_state:
                                        del st.session_state[key]
                                st.session_state.processing_step = 'auto_detect'
                                st.rerun()
                    else:
                        st.error(message)
            else:
                st.error("プレビュー画像の生成に失敗しました。最初からやり直してください。")
                if st.button("🔙 最初からやり直し"):
                    # セッションステートをリセット
                    for key in ['processed_image', 'processing_step', 'manual_coords', 'fill_areas', 'auto_detected_area', 'confirmed_drawing_area', 'eyedropper_mode']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.session_state.processing_step = 'auto_detect'
                    st.rerun()

elif uploaded_pdf:
    st.info("テンプレート画像（PNG）もアップロードしてください。")
elif uploaded_template:
    st.info("図面PDF または 画像をアップロードしてください。")
else:
    st.info("図面ファイルとテンプレート画像をアップロードして開始してください。")

# サイドバーに操作ガイド
with st.sidebar:
    st.header("📖 操作ガイド")
    st.markdown("""
    ### 🚀 使い方
    1. **図面ファイル**と**テンプレート画像**をアップロード
    2. **自動検出結果**を確認
    3. 必要に応じて**手動調整**または**塗りつぶし**
    4. **PDF生成**してダウンロード
    
    ### ✨ 新機能 v1.4.5
    - 🎯 **図面領域確定システム**: 帯の自動認識/手動修正で範囲を確定
    - 🎨 **スポイトツール強化**: 色取得後に1点目を自動設定し、スムーズに範囲指定へ
    - 📝 **名前を付けて保存**: 日付+物件名+価格のファイル名自動生成
    - ⚡ **PDF生成の高速化**: 1クリックで生成からダウンロードまで完了
    - 🎨 **図面領域内塗りつぶし**: 確定された図面領域内でのみ塗りつぶし可能
    - 📊 **座標表示詳細化**: 元画像座標と図面内相対座標の両方を表示
    - 🔄 **改善されたフロー**: 図面領域確定 → 塗りつぶし → 出力の明確な流れ
    - 🖼️ **プレビュー画面最適化**: 塗りつぶしモードで1つのプレビューに統合
    - 🛡️ エラーハンドリング強化
    """)
    
    if st.session_state.get('processing_step'):
        st.info(f"現在のステップ: {st.session_state.processing_step}")
    
    # デバッグ情報（開発用）
    if st.checkbox("デバッグ情報を表示"):
        st.write("### デバッグ情報")
        if st.session_state.original_image:
            st.write(f"元画像サイズ: {st.session_state.original_image.width} x {st.session_state.original_image.height}")
        if st.session_state.processed_image:
            st.write(f"処理済み画像サイズ: {st.session_state.processed_image.width} x {st.session_state.processed_image.height}")
        st.write(f"塗りつぶし領域数: {len(st.session_state.fill_areas)}")
        if st.session_state.auto_detected_area:
            st.write(f"自動検出領域: {st.session_state.auto_detected_area}")
        if 'confirmed_drawing_area' in st.session_state:
            st.write(f"確定図面領域: {st.session_state.confirmed_drawing_area}")
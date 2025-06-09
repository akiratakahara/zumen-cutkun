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
APP_VERSION = "v1.4.1"
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
        'template_image': None
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

# ファイルアップロード
uploaded_pdf = st.file_uploader("図面PDF または 画像", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=False)
uploaded_template = st.file_uploader("テンプレ画像（PNG）", type=["png"])

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
    
    # ステップ1: 自動帯認識
    if st.session_state.processing_step == 'auto_detect':
        with st.spinner("図面領域を自動検出中..."):
            st.session_state.auto_detected_area = auto_detect_drawing_area(st.session_state.original_image)
            st.session_state.processing_step = 'review_auto'
    
    # ステップ2: 自動検出結果のレビュー
    if st.session_state.processing_step == 'review_auto':
        st.subheader("🤖 自動検出結果")
        
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
                st.session_state.processing_step = 'fill_mode'
                st.session_state.manual_coords = []
                st.rerun()
    
    # ステップ3: 手動調整
    elif st.session_state.processing_step == 'manual_adjust':
        st.subheader("🔧 手動で図面領域を調整")
        st.info("画像上で2点をクリックして範囲を指定してください")
        
        # クリック座標取得
        coordinates = streamlit_image_coordinates(
            np.array(st.session_state.preview_image),
            key="manual_select"
        )
        
        if coordinates and len(st.session_state.manual_coords) < 2:
            st.session_state.manual_coords.append((coordinates['x'], coordinates['y']))
            st.success(f"点 {len(st.session_state.manual_coords)}: X={coordinates['x']}, Y={coordinates['y']}")
        
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
                st.image(preview_with_manual, caption="手動選択結果（赤枠）", use_column_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ この領域で確定"):
                        cropped = safe_crop_image(st.session_state.original_image, manual_area)
                        if cropped is not None:
                            st.session_state.processed_image = cropped
                            st.session_state.processing_step = 'final'
                            st.rerun()
                
                with col2:
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
        st.info("複数の範囲を塗りつぶしできます。2点クリックで範囲を指定してください。")
        
        # 塗りつぶし色選択
        fill_color = st.color_picker("塗りつぶし色", value="#FFFFFF")
        
        # 現在の塗りつぶし領域を表示
        current_image = st.session_state.original_image.copy()
        if st.session_state.fill_areas:
            current_image = apply_fill_areas(current_image, st.session_state.fill_areas)
        
        # プレビュー画像更新
        current_preview = safe_resize_preview(current_image, 800)
        if current_preview is None:
            st.error("プレビュー画像の生成に失敗しました。")
            st.stop()
        
        # クリック座標取得
        coordinates = streamlit_image_coordinates(
            np.array(current_preview),
            key="fill_select"
        )
        
        if coordinates and len(st.session_state.manual_coords) < 2:
            st.session_state.manual_coords.append((coordinates['x'], coordinates['y']))
            st.success(f"点 {len(st.session_state.manual_coords)}: X={coordinates['x']}, Y={coordinates['y']}")
        
        # 2点が選択された場合
        if len(st.session_state.manual_coords) == 2:
            (x1, y1), (x2, y2) = st.session_state.manual_coords
            
            # 座標変換
            scale_x = st.session_state.original_image.width / current_preview.width
            scale_y = st.session_state.original_image.height / current_preview.height
            
            real_x1 = int(min(x1, x2) * scale_x)
            real_y1 = int(min(y1, y2) * scale_y)
            real_x2 = int(max(x1, x2) * scale_x)
            real_y2 = int(max(y1, y2) * scale_y)
            
            fill_area = (real_x1, real_y1, real_x2, real_y2)
            
            if validate_area(fill_area, st.session_state.original_image.width, st.session_state.original_image.height):
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("✅ 塗りつぶし実行"):
                        # RGB値に変換
                        hex_color = fill_color.lstrip('#')
                        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                        
                        st.session_state.fill_areas.append((real_x1, real_y1, real_x2, real_y2, rgb_color))
                        st.session_state.manual_coords = []
                        st.success("塗りつぶしを追加しました！")
                        st.rerun()
                
                with col2:
                    if st.button("🔄 範囲リセット"):
                        st.session_state.manual_coords = []
                        st.rerun()
                
                with col3:
                    if st.button("📋 塗りつぶし完了"):
                        filled_image = apply_fill_areas(st.session_state.original_image, st.session_state.fill_areas)
                        st.session_state.processed_image = filled_image
                        st.session_state.processing_step = 'final'
                        st.rerun()
            else:
                st.error("選択された領域が無効です。")
                if st.button("🔄 リセット"):
                    st.session_state.manual_coords = []
                    st.rerun()
        
        # 現在の状態を表示
        st.image(current_preview, caption=f"現在の状態（塗りつぶし領域: {len(st.session_state.fill_areas)}個）", use_column_width=True)
        
        if st.session_state.fill_areas:
            if st.button("🗑️ 最後の塗りつぶしを削除"):
                st.session_state.fill_areas.pop()
                st.rerun()
            
            if st.button("🧹 全塗りつぶしをクリア"):
                st.session_state.fill_areas = []
                st.rerun()
    
    # ステップ5: 最終確認とPDF生成
    elif st.session_state.processing_step == 'final':
        st.subheader("📄 最終確認とPDF生成")
        
        # 処理済み画像のプレビュー
        if st.session_state.processed_image:
            # 安全なプレビュー生成
            final_preview = safe_resize_preview(st.session_state.processed_image, 600)
            
            if final_preview is not None:
                st.image(final_preview, caption="処理済み図面（テンプレートに合成される部分）", use_column_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📄 PDF生成", type="primary"):
                        with st.spinner("PDFを生成中..."):
                            pdf_buffer, message = generate_pdf(st.session_state.processed_image, st.session_state.template_image)
                            
                            if pdf_buffer:
                                st.success(message)
                                st.download_button(
                                    "📥 PDFをダウンロード",
                                    data=pdf_buffer.getvalue(),
                                    file_name="zumen_output.pdf",
                                    mime="application/pdf"
                                )
                            else:
                                st.error(message)
                
                with col2:
                    if st.button("🔙 最初からやり直し"):
                        # セッションステートをリセット
                        for key in ['processed_image', 'processing_step', 'manual_coords', 'fill_areas', 'auto_detected_area']:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.session_state.processing_step = 'auto_detect'
                        st.rerun()
            else:
                st.error("プレビュー画像の生成に失敗しました。最初からやり直してください。")
                if st.button("🔙 最初からやり直し"):
                    # セッションステートをリセット
                    for key in ['processed_image', 'processing_step', 'manual_coords', 'fill_areas', 'auto_detected_area']:
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
    
    ### ✨ 新機能 v1.4.1
    - 🤖 改良された自動帯認識
    - 🎨 複数範囲塗りつぶし対応
    - ⚡ 高速読み込み（キャッシュ機能）
    - 🖼️ プレビュー画面最適化
    - 📱 直感的なUI
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
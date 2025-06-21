import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
import io
import cv2
import numpy as np
import re
import easyocr
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from streamlit_image_coordinates import streamlit_image_coordinates
from datetime import datetime
import warnings
import json
import os
from pathlib import Path
from collections import deque
import math
from scipy.signal import savgol_filter
warnings.filterwarnings('ignore')

# Import with fallback for streamlit-image-coordinates
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_IMAGE_COORDINATES = True
except ImportError:
    HAS_IMAGE_COORDINATES = False
    st.error("streamlit-image-coordinatesパッケージがインストールされていません。pip install streamlit-image-coordinatesを実行してください。")

st.set_page_config(layout="wide")
st.title("図面帯カットくん｜不動産営業の即戦力")
APP_VERSION = "v1.7.4"
st.markdown(f"#### 🏷️ バージョン: {APP_VERSION}")

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
        'property_price': '',
        'selected_color': '#FFFFFF',
        'confirmed_drawing_area': None,
        'learning_data_version': '1.0',
        'recent_predictions': deque(maxlen=10)  # 直近10件の予測を保存
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# 学習データの保存先
LEARNING_DATA_DIR = Path("learning_data")
LEARNING_DATA_FILE = LEARNING_DATA_DIR / "band_detection_data.json"

@st.cache_resource
def get_ocr_reader():
    """EasyOCRリーダーを初期化（キャッシュ）"""
    try:
        with st.spinner("初回起動時：AI（OCRモデル）をダウンロードしています...（数分かかる場合があります）"):
            # Warningを非表示にする
            import logging
            logging.getLogger('easyocr').setLevel(logging.ERROR)
            reader = easyocr.Reader(['ja', 'en'])
        st.success("✅ AI（OCRモデル）の準備が完了しました。")
        return reader
    except Exception as e:
        st.error(f"OCRエンジンの初期化に失敗しました。EasyOCRが正しくインストールされているか確認してください。: {e}")
        st.warning("`pip install easyocr` を実行してください。")
        return None

def migrate_old_learning_data():
    """古い形式の学習データを新しい形式に移行"""
    try:
        if not LEARNING_DATA_FILE.exists():
            return
        
        with open(LEARNING_DATA_FILE, 'r') as f:
            data = json.load(f)
        
        # 既に新しい形式の場合は何もしない
        if 'learning_records' in data:
            return
        
        # 古い形式から新しい形式に移行
        if 'band_positions' in data and 'image_features' in data:
            st.info("🔄 学習データを新しい形式に移行中...")
            
            new_data = {
                'version': '1.0',
                'learning_records': [],
                'metadata': {
                    'total_records': len(data['band_positions']),
                    'manual_corrections': 0,
                    'auto_detections': len(data['band_positions']),
                    'last_updated': datetime.now().isoformat()
                }
            }
            
            # 古いデータを新しい形式に変換
            for i, (position, features) in enumerate(zip(data['band_positions'], data['image_features'])):
                record = {
                    'id': i + 1,
                    'timestamp': datetime.now().isoformat(),
                    'band_position': position,
                    'image_features': features,
                    'confidence': 0.5,  # デフォルト自信度
                    'is_manual_correction': False,  # 古いデータは自動検出として扱う
                    'is_positive_example': True
                }
                new_data['learning_records'].append(record)
            
            # 新しい形式で保存
            with open(LEARNING_DATA_FILE, 'w') as f:
                json.dump(new_data, f, indent=2)
            
            st.success("✅ 学習データの移行が完了しました！")
    except Exception as e:
        st.error(f"学習データの移行に失敗しました: {str(e)}")

def init_learning_data():
    """学習データの初期化"""
    LEARNING_DATA_DIR.mkdir(exist_ok=True)
    
    # 既存データの移行を試行
    migrate_old_learning_data()
    
    if not LEARNING_DATA_FILE.exists():
        with open(LEARNING_DATA_FILE, 'w') as f:
            json.dump({
                'version': '1.0',
                'learning_records': [],
                'metadata': {
                    'total_records': 0,
                    'manual_corrections': 0,
                    'auto_detections': 0,
                    'last_updated': datetime.now().isoformat()
                }
            }, f, indent=2)

def extract_color_features(image: Image.Image, band_area=None):
    """画像から色特徴量を抽出"""
    try:
        # RGBに変換
        rgb_img = image.convert('RGB')
        np_img = np.array(rgb_img)
        
        # 色ヒストグラム計算（各チャンネル16ビン）
        hist_r = cv2.calcHist([np_img], [0], None, [16], [0, 256]).flatten()
        hist_g = cv2.calcHist([np_img], [1], None, [16], [0, 256]).flatten()
        hist_b = cv2.calcHist([np_img], [2], None, [16], [0, 256]).flatten()
        
        # 正規化
        hist_r = hist_r / np.sum(hist_r)
        hist_g = hist_g / np.sum(hist_g)
        hist_b = hist_b / np.sum(hist_b)
        
        # HSV変換して色相・明度を計算
        hsv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv_img)
        
        # 帯領域が指定されている場合はその領域のみ分析
        if band_area:
            x1, y1, x2, y2 = band_area
            h_band = h[y1:y2, x1:x2]
            s_band = s[y1:y2, x1:x2]
            v_band = v[y1:y2, x1:x2]
            
            # 帯領域の色相・明度の統計
            hue_mean = np.mean(h_band)
            hue_std = np.std(h_band)
            value_mean = np.mean(v_band)
            value_std = np.std(v_band)
        else:
            # 全体の色相・明度の統計
            hue_mean = np.mean(h)
            hue_std = np.std(h)
            value_mean = np.mean(v)
            value_std = np.std(v)
        
        return {
            'hist_r': hist_r.tolist(),
            'hist_g': hist_g.tolist(),
            'hist_b': hist_b.tolist(),
            'hue_mean': float(hue_mean),
            'hue_std': float(hue_std),
            'value_mean': float(value_mean),
            'value_std': float(value_std)
        }
    except Exception as e:
        st.error(f"色特徴量の抽出に失敗しました: {str(e)}")
        return None

def extract_image_features(image: Image.Image, band_area=None):
    """画像から特徴量を抽出（色特徴量追加版）"""
    try:
        # グレースケール変換
        gray = np.array(image.convert('L'))
        
        # エッジ検出
        edges = cv2.Canny(gray, 50, 150)
        
        # 基本特徴量
        basic_features = {
            'width': image.width,
            'height': image.height,
            'edge_density': np.mean(edges) / 255.0,
            'brightness': np.mean(gray) / 255.0
        }
        
        # 色特徴量を追加
        color_features = extract_color_features(image, band_area)
        if color_features:
            basic_features.update(color_features)
        
        return basic_features
    except Exception as e:
        st.error(f"特徴量の抽出に失敗しました: {str(e)}")
        return None

def calculate_similarity(features1, features2):
    """2つの特徴量セットの類似度を計算"""
    try:
        # 基本特徴量の類似度
        basic_diff = (
            abs(features1['width'] - features2['width']) / max(features1['width'], features2['width']) +
            abs(features1['height'] - features2['height']) / max(features1['height'], features2['height']) +
            abs(features1['edge_density'] - features2['edge_density']) +
            abs(features1['brightness'] - features2['brightness'])
        ) / 4.0
        
        # 色ヒストグラムの類似度（コサイン類似度）
        hist_similarity = 0
        if 'hist_r' in features1 and 'hist_r' in features2:
            hist_r_sim = np.dot(features1['hist_r'], features2['hist_r']) / (
                np.linalg.norm(features1['hist_r']) * np.linalg.norm(features2['hist_r']) + 1e-8
            )
            hist_g_sim = np.dot(features1['hist_g'], features2['hist_g']) / (
                np.linalg.norm(features1['hist_g']) * np.linalg.norm(features2['hist_g']) + 1e-8
            )
            hist_b_sim = np.dot(features1['hist_b'], features2['hist_b']) / (
                np.linalg.norm(features1['hist_b']) * np.linalg.norm(features2['hist_b']) + 1e-8
            )
            hist_similarity = (hist_r_sim + hist_g_sim + hist_b_sim) / 3.0
        
        # 色相・明度の類似度
        color_similarity = 0
        if 'hue_mean' in features1 and 'hue_mean' in features2:
            hue_diff = abs(features1['hue_mean'] - features2['hue_mean']) / 180.0
            value_diff = abs(features1['value_mean'] - features2['value_mean']) / 255.0
            color_similarity = 1.0 - (hue_diff + value_diff) / 2.0
        
        # 総合類似度（基本:色:色相 = 3:4:3の重み）
        total_similarity = (1.0 - basic_diff) * 0.3 + hist_similarity * 0.4 + color_similarity * 0.3
        
        return total_similarity
    except Exception as e:
        st.error(f"類似度計算に失敗しました: {str(e)}")
        return 0.0

def save_learning_data(band_position, image_features, confidence=0.5, is_manual_correction=False):
    """学習データの保存（フラグ・自信度付き）"""
    try:
        LEARNING_DATA_DIR.mkdir(exist_ok=True)
        if LEARNING_DATA_FILE.exists():
            with open(LEARNING_DATA_FILE, 'r') as f:
                data = json.load(f)
        else:
            data = {
                'version': '1.0',
                'learning_records': [],
                'metadata': {
                    'total_records': 0,
                    'manual_corrections': 0,
                    'auto_detections': 0,
                    'last_updated': datetime.now().isoformat()
                }
            }
        
        # 新しい学習レコードを作成
        record = {
            'id': len(data['learning_records']) + 1,
            'timestamp': datetime.now().isoformat(),
            'band_position': band_position,
            'image_features': image_features,
            'confidence': confidence,
            'is_manual_correction': is_manual_correction,
            'is_positive_example': True  # 正例として保存
        }
        
        # レコードを追加
        data['learning_records'].append(record)
        
        # メタデータを更新
        data['metadata']['total_records'] = len(data['learning_records'])
        if is_manual_correction:
            data['metadata']['manual_corrections'] += 1
        else:
            data['metadata']['auto_detections'] += 1
        data['metadata']['last_updated'] = datetime.now().isoformat()
        
        # データを保存
        with open(LEARNING_DATA_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        # 件数制限を自動適用（1,000件をデフォルト）
        manage_learning_data(1000)
        
        return True
    except Exception as e:
        st.error(f"学習データの保存に失敗しました: {str(e)}")
        return False

def predict_band_position(image: Image.Image, use_recent_average=True, min_confidence=0.3):
    """学習データを基に帯位置を予測（改良版）"""
    try:
        if not LEARNING_DATA_FILE.exists():
            return None, 0.0
        
        # 学習データの読み込み
        with open(LEARNING_DATA_FILE, 'r') as f:
            data = json.load(f)
        
        if not data['learning_records']:
            return None, 0.0
        
        # 現在の画像の特徴量を抽出
        current_features = extract_image_features(image)
        if not current_features:
            return None, 0.0
        
        # 類似度計算とソート
        similarities = []
        for record in data['learning_records']:
            if not record.get('is_positive_example', True):  # 負例は除外
                continue
                
            similarity = calculate_similarity(current_features, record['image_features'])
            
            # 手動修正データの重み付け（1.2倍）
            weight = 1.2 if record.get('is_manual_correction', False) else 1.0
            weighted_similarity = similarity * weight
            
            similarities.append({
                'record': record,
                'similarity': weighted_similarity,
                'confidence': record.get('confidence', 0.5)
            })
        
        if not similarities:
            return None, 0.0
        
        # 類似度でソート
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 直近N件の平均値を使用する場合
        if use_recent_average and len(similarities) >= 3:
            # 上位3件の平均位置を計算
            top_records = similarities[:3]
            avg_x1 = sum(r['record']['band_position'][0] for r in top_records) / len(top_records)
            avg_y1 = sum(r['record']['band_position'][1] for r in top_records) / len(top_records)
            avg_x2 = sum(r['record']['band_position'][2] for r in top_records) / len(top_records)
            avg_y2 = sum(r['record']['band_position'][3] for r in top_records) / len(top_records)
            
            # 平均類似度を計算
            avg_similarity = sum(r['similarity'] for r in top_records) / len(top_records)
            
            predicted_position = [int(avg_x1), int(avg_y1), int(avg_x2), int(avg_y2)]
            confidence = avg_similarity
        else:
            # 最も類似した1件を使用
            best_match = similarities[0]
            predicted_position = best_match['record']['band_position']
            confidence = best_match['similarity']
        
        # 信頼度が閾値を下回る場合は予測を無効にする
        if confidence < min_confidence:
            return None, confidence
        
        return predicted_position, confidence
    except Exception as e:
        st.error(f"帯位置の予測に失敗しました: {str(e)}")
        return None, 0.0

def detect_text_regions(image: Image.Image):
    """文字領域を検出（OCRライクな手法）"""
    try:
        # グレースケール変換
        gray = np.array(image.convert('L'))
        
        # 二値化（文字を強調）
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # モルフォロジー処理で文字を連結
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 輪郭検出
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        h, w = gray.shape
        
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            
            # 文字らしい領域の条件
            if (area > 50 and  # 最小サイズ
                area < w * h * 0.1 and  # 最大サイズ（画像の10%以下）
                bw > 5 and bh > 5 and  # 最小幅・高さ
                bw < w * 0.8 and bh < h * 0.3):  # 最大幅・高さ
                
                text_regions.append((x, y, x + bw, y + bh, area))
        
        return text_regions
    except Exception as e:
        st.error(f"文字領域検出に失敗しました: {str(e)}")
        return []

def detect_horizontal_lines(image: Image.Image):
    """水平線を検出"""
    try:
        # グレースケール変換
        gray = np.array(image.convert('L'))
        
        # エッジ検出
        edges = cv2.Canny(gray, 50, 150)
        
        # 水平線の検出（HoughLinesP）
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        horizontal_lines = []
        h, w = gray.shape
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # 水平線の判定（角度が10度以内）
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 10 or angle > 170:
                    # 線の長さが画像幅の30%以上
                    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    if line_length > w * 0.3:
                        horizontal_lines.append((x1, y1, x2, y2, line_length))
        
        return horizontal_lines
    except Exception as e:
        st.error(f"水平線検出に失敗しました: {str(e)}")
        return []

def analyze_background_color(image: Image.Image, region):
    """指定領域の背景色を分析"""
    try:
        x1, y1, x2, y2 = region
        region_img = image.crop((x1, y1, x2, y2))
        
        # RGBに変換
        rgb_img = region_img.convert('RGB')
        np_img = np.array(rgb_img)
        
        # 平均色を計算
        mean_color = np.mean(np_img, axis=(0, 1))
        
        # 色の一様性を計算（標準偏差が小さいほど一様）
        color_std = np.std(np_img, axis=(0, 1))
        uniformity = 1.0 - (np.mean(color_std) / 255.0)
        
        return {
            'mean_color': mean_color.tolist(),
            'uniformity': float(uniformity),
            'brightness': float(np.mean(mean_color) / 255.0)
        }
    except Exception as e:
        st.error(f"背景色分析に失敗しました: {str(e)}")
        return None

def detect_band_by_content(image: Image.Image, reader):
    """画像内のテキストコンテンツを分析して帯領域を検出"""
    try:
        np_img = np.array(image.convert('RGB'))
        h, w, _ = np_img.shape

        # 下半分をクロップしてOCRの対象範囲を限定
        scan_area_top = h // 2
        scan_area = np_img[scan_area_top:, :]

        ocr_results = reader.readtext(scan_area, detail=1, paragraph=False)

        # 帯によくあるキーワードと正規表現
        keywords = [
            '株式会社', '会社', '一級建築士', '免許', '登録', '保証', '協会',
            '所在地', '住所', '電話', 'TEL', 'FAX', 'メール'
        ]
        # 日本の電話番号 (フリーダイヤル含む), FAX番号のパターン
        phone_fax_pattern = re.compile(r'(\d{2,4}-\d{2,4}-\d{4}|\(0\d{1,4}\)\d{1,4}-\d{4}|0120-\d{2,3}-\d{3})')
        
        band_content_boxes = []
        for (bbox, text, prob) in ocr_results:
            if prob < 0.3: continue

            # キーワード検索
            if any(keyword in text for keyword in keywords):
                band_content_boxes.append(bbox)
                continue
            
            # 電話番号・FAX番号検索
            if phone_fax_pattern.search(text):
                band_content_boxes.append(bbox)

        if not band_content_boxes:
            return None, 0.0

        # 検出されたコンテンツのY座標の最小値（帯の上端候補）を特定
        min_y_in_scan_area = min([box[0][1] for box in band_content_boxes])
        
        # 元画像での絶対Y座標
        band_top_candidate_y = scan_area_top + min_y_in_scan_area

        # 候補ラインの上部で水平線を探す
        lines = detect_horizontal_lines(image)
        strongest_line_y = 0
        
        # 帯コンテンツの少し上から、図面の上部にかけての範囲で線を探す
        line_search_top = int(band_top_candidate_y - h * 0.1) 
        line_search_bottom = int(band_top_candidate_y + h * 0.05)
        
        relevant_lines = [
            y1 for x1, y1, x2, y2, length in lines 
            if y1 > line_search_top and y1 < line_search_bottom
        ]

        if relevant_lines:
            # 候補領域内で最も下にある線を帯の上端とする
            strongest_line_y = max(relevant_lines)
        
        # 線が見つかればそれを、なければテキストの開始位置を帯の上端とする
        final_band_top_y = strongest_line_y if strongest_line_y > 0 else band_top_candidate_y
        
        # 最終的な上端位置を微調整（少しだけ上にあげる）
        final_band_top_y = max(0, final_band_top_y - 10)

        # 帯の高さが妥当かチェック
        band_height = h - final_band_top_y
        if not (h * 0.05 < band_height < h * 0.5):
            return None, 0.0 # 帯の高さが5%未満か50%以上なら無効
        
        score = len(band_content_boxes) / len(ocr_results) if ocr_results else 0.0
        return (0, int(final_band_top_y), w, h), score

    except Exception as e:
        st.error(f"コンテンツ分析による帯検出に失敗しました: {str(e)}")
        return None, 0.0

def detect_footer_band_by_layout(image: Image.Image):
    """フッター帯をレイアウト分析（垂直プロファイル）で検出"""
    try:
        gray_img = image.convert('L')
        np_gray = np.array(gray_img)
        h, w = np_gray.shape

        # 1. テキストとエッジの垂直方向の投影プロファイルを作成
        _, binary = cv2.threshold(np_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        text_profile = np.sum(binary, axis=1)

        edges = cv2.Canny(np_gray, 50, 150, apertureSize=3)
        edge_profile = np.sum(edges, axis=1)

        # 2. プロファイルを正規化して結合
        if text_profile.max() > 0:
            text_profile = text_profile / text_profile.max()
        if edge_profile.max() > 0:
            edge_profile = edge_profile / edge_profile.max()
        
        # テキストの比重を高くする
        combined_profile = (text_profile * 0.8) + (edge_profile * 0.2)

        # 3. プロファイルを平滑化してノイズを除去
        window_size = max(11, int(h * 0.02))
        if window_size % 2 == 0: window_size += 1
        
        if window_size >= len(combined_profile):
             return None, 0.0 # 画像が小さすぎる
        
        smoothed_profile = savgol_filter(combined_profile, window_size, 3) # window, polyorder

        # 4. 画像下部からスキャンして帯の上端を探す
        scan_start_y = h - 1
        scan_end_y = int(h * 0.5) # 下半分のみを対象

        # 画像下部20%の最大値を基準とする
        bottom_area_start = int(h * 0.8)
        peak_in_bottom = smoothed_profile[bottom_area_start:].max()

        if peak_in_bottom < 0.1: # 下部に十分なコンテンツがない場合は失敗
            return None, 0.0

        # コンテンツが大幅に減少する点を「帯の上端」と判断
        threshold = peak_in_bottom * 0.35 

        band_top_y = 0
        for y in range(scan_start_y, scan_end_y, -1):
            if smoothed_profile[y] < threshold:
                # 安定した低コンテンツ領域か確認
                rows_to_check = int(h * 0.03) # 画像高さの3%をチェック
                area_above_y = y - rows_to_check
                if area_above_y < scan_end_y: continue

                if smoothed_profile[area_above_y:y].mean() < threshold * 1.2:
                    band_top_y = y
                    break
        
        if band_top_y == 0: # 明確な境界が見つからない
            return None, 0.0
            
        # 5. 検出結果をスコアリング
        area_above = smoothed_profile[scan_end_y:band_top_y]
        area_below = smoothed_profile[band_top_y:scan_start_y]
        avg_above = area_above.mean() if len(area_above) > 0 else 0
        avg_below = area_below.mean() if len(area_below) > 0 else 0
        
        # 帯部分とそれ以外の部分の差が大きいほど高スコア
        score = (avg_below - avg_above) / (peak_in_bottom + 1e-6)
        
        # 検出された帯の高さが妥当かチェック
        band_height = h - band_top_y
        if not (h * 0.05 < band_height < h * 0.4):
            return None, 0.0 # 帯の高さが5%未満か40%以上なら無効

        detected_band_region = (0, band_top_y, w, h)
        return detected_band_region, score

    except Exception as e:
        st.error(f"フッター帯のレイアウト分析に失敗しました: {str(e)}")
        return None, 0.0

def auto_detect_drawing_area(image: Image.Image):
    """図面領域を自動検出（コンテンツ分析AI）"""
    h, w = image.height, image.width
    reader = get_ocr_reader()
    if not reader:
        st.error("OCRエンジンが利用できないため、自動検出を中止します。")
        return (0, 0, w, int(h*0.8))


    # 1. 学習データからの予測を最優先
    predicted_drawing_area, confidence = predict_band_position(image)
    if predicted_drawing_area and confidence > 0.8: # 高信頼度の学習結果を優先
        st.session_state.recent_predictions.append({
            'position': predicted_drawing_area, 'confidence': confidence, 
            'method': 'learning', 'timestamp': datetime.now().isoformat()
        })
        return predicted_drawing_area

    # 2. コンテンツ分析で帯を検出
    band_region, content_score = detect_band_by_content(image, reader)
    if band_region and content_score > 0.05: # キーワードが少しでもあれば採用
        _, band_top_y, _, _ = band_region
        drawing_area = (0, 0, w, band_top_y)
        
        st.session_state.recent_predictions.append({
            'position': drawing_area, 'confidence': content_score, 
            'method': 'content_analysis', 'timestamp': datetime.now().isoformat()
        })
        return drawing_area
    
    # 3. レイアウト分析をフォールバックとして使用
    footer_region, layout_score = detect_footer_band_by_layout(image)
    if footer_region and layout_score > 0.4:
        _, band_top_y, _, _ = footer_region
        drawing_area = (0, 0, w, band_top_y)
        
        st.session_state.recent_predictions.append({
            'position': drawing_area, 'confidence': layout_score, 
            'method': 'layout_analysis_fallback', 'timestamp': datetime.now().isoformat()
        })
        return drawing_area

    # 4. 最終フォールバック
    fallback_area = (0, 0, w, int(h * 0.8))
    st.session_state.recent_predictions.append({
        'position': fallback_area, 'confidence': 0.1, 
        'method': 'fallback', 'timestamp': datetime.now().isoformat()
    })
    return fallback_area

def manage_learning_data(max_records=1000):
    """学習データの管理（件数制限、古いデータ削除）"""
    try:
        if not LEARNING_DATA_FILE.exists():
            return
        
        with open(LEARNING_DATA_FILE, 'r') as f:
            data = json.load(f)
        
        records = data['learning_records']
        
        # 件数制限を超えている場合、古いデータを削除
        if len(records) > max_records:
            # 手動修正データを優先保持
            manual_records = [r for r in records if r.get('is_manual_correction', False)]
            auto_records = [r for r in records if not r.get('is_manual_correction', False)]
            
            # 手動修正データは全て保持
            # 自動検出データは新しい順に制限内まで保持
            keep_auto_count = max_records - len(manual_records)
            if keep_auto_count > 0:
                # タイムスタンプでソート（新しい順）
                auto_records.sort(key=lambda x: x['timestamp'], reverse=True)
                auto_records = auto_records[:keep_auto_count]
            else:
                auto_records = []
            
            # 手動修正データを優先して結合
            records = manual_records + auto_records
            
            # メタデータを更新
            data['learning_records'] = records
            data['metadata']['total_records'] = len(records)
            data['metadata']['manual_corrections'] = len(manual_records)
            data['metadata']['auto_detections'] = len(auto_records)
            data['metadata']['last_updated'] = datetime.now().isoformat()
            
            # 保存
            with open(LEARNING_DATA_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        return False
    except Exception as e:
        st.error(f"学習データの管理に失敗しました: {str(e)}")
        return False

def export_learning_data():
    """学習データをエクスポート"""
    try:
        if not LEARNING_DATA_FILE.exists():
            return None
        
        with open(LEARNING_DATA_FILE, 'r') as f:
            data = json.load(f)
        
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"学習データのエクスポートに失敗しました: {str(e)}")
        return None

def import_learning_data(import_data):
    """学習データをインポート"""
    try:
        data = json.loads(import_data)
        
        # データ形式の検証
        if 'learning_records' not in data or 'metadata' not in data:
            st.error("無効な学習データ形式です")
            return False
        
        # バックアップを作成
        if LEARNING_DATA_FILE.exists():
            backup_file = LEARNING_DATA_FILE.with_suffix('.json.backup')
            with open(LEARNING_DATA_FILE, 'r') as f:
                backup_data = f.read()
            with open(backup_file, 'w') as f:
                f.write(backup_data)
        
        # 新しいデータを保存
        with open(LEARNING_DATA_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        st.success("✅ 学習データのインポートが完了しました！")
        return True
    except Exception as e:
        st.error(f"学習データのインポートに失敗しました: {str(e)}")
        return False

def clear_learning_data():
    """学習データをクリア"""
    try:
        if LEARNING_DATA_FILE.exists():
            # バックアップを作成
            backup_file = LEARNING_DATA_FILE.with_suffix('.json.backup')
            with open(LEARNING_DATA_FILE, 'r') as f:
                backup_data = f.read()
            with open(backup_file, 'w') as f:
                f.write(backup_data)
            
            # 初期化
            init_learning_data()
            st.success("✅ 学習データをクリアしました！（バックアップを作成済み）")
            return True
        return False
    except Exception as e:
        st.error(f"学習データのクリアに失敗しました: {str(e)}")
        return False

init_session_state()
init_learning_data()
get_ocr_reader()

@st.cache_data
def load_and_process_image(file_data, file_name):
    # アプリケーション起動時にscipyの存在をチェック
    try:
        import scipy
    except ImportError:
        st.warning("高精度な帯認識（レイアウト分析）のために、SciPyライブラリのインストールを推奨します。「pip install scipy」でインストールできます。")
    
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
        
        # AI予測の信頼度を表示
        if st.session_state.recent_predictions:
            latest_prediction = list(st.session_state.recent_predictions)[-1]
            confidence = latest_prediction['confidence']
            
            if confidence > 0.7:
                st.success(f"🎯 AI信頼度: {confidence:.1%} - 高信頼度の予測です")
            elif confidence > 0.5:
                st.info(f"📊 AI信頼度: {confidence:.1%} - 中程度の信頼度です")
            else:
                st.warning(f"⚠️ AI信頼度: {confidence:.1%} - 低信頼度です。手動調整を推奨します")
        
        st.info("緑枠の範囲でよろしいですか？")
        
        # プレビュー画像に検出領域を描画
        preview_with_area = draw_preview_with_area(
            st.session_state.preview_image, 
            st.session_state.auto_detected_area,
            color=(0, 255, 0),
            label="自動検出された図面領域"
        )
        
        st.image(preview_with_area, caption="自動検出結果（緑枠が図面領域）", use_container_width=True)
        
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
        st.info("左上→右下の順番で2点をクリックしてください")
        
        # クリック座標取得
        coordinates = streamlit_image_coordinates(
            np.array(st.session_state.preview_image),
            key="manual_select"
        )
        
        if coordinates and len(st.session_state.manual_coords) < 2:
            st.session_state.manual_coords.append((coordinates['x'], coordinates['y']))
            if len(st.session_state.manual_coords) == 1:
                st.success("1点目を選択")
            else:
                st.success("2点目を選択")
        
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
                st.image(preview_with_manual, caption="手動選択結果（赤枠が確定される図面領域）", use_container_width=True)
                
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
        
        # 手動調整時の学習データ保存処理を追加
        if st.session_state.processing_step == 'manual_adjust' and len(st.session_state.manual_coords) == 2:
            (x1, y1), (x2, y2) = st.session_state.manual_coords
            
            # プレビューサイズから元画像サイズに変換
            scale_x = st.session_state.original_image.width / st.session_state.preview_image.width
            scale_y = st.session_state.original_image.height / st.session_state.preview_image.height
            
            real_x1 = int(min(x1, x2) * scale_x)
            real_y1 = int(min(y1, y2) * scale_y)
            real_x2 = int(max(x1, x2) * scale_x)
            real_y2 = int(max(y1, y2) * scale_y)
            
            manual_area = (real_x1, real_y1, real_x2, real_y2)
            
            # 学習データの保存（手動修正フラグ付き）
            if validate_area(manual_area, st.session_state.original_image.width, st.session_state.original_image.height):
                image_features = extract_image_features(st.session_state.original_image, manual_area)
                if image_features:
                    # 手動修正として高自信度で保存
                    success = save_learning_data(
                        band_position=manual_area,
                        image_features=image_features,
                        confidence=0.9,  # 手動修正は高自信度
                        is_manual_correction=True
                    )
                    if success:
                        st.success("✅ 学習データを保存しました！（手動修正データとして高優先度で登録）")
                    else:
                        st.warning("⚠️ 学習データの保存に失敗しました")
    
    # ステップ4: 塗りつぶしモード
    elif st.session_state.processing_step == 'fill_mode':
        st.subheader("🎨 塗りつぶしモード")
        
        # 確定された図面領域を表示
        if 'confirmed_drawing_area' not in st.session_state:
            st.warning("図面領域が確定されていません。")
            st.stop()
        
        dx1, dy1, dx2, dy2 = st.session_state.confirmed_drawing_area
        st.info("💡 手順：🎨色取得 → 📍範囲選択 → ✅実行")
        
        # 操作モード選択（排他的）
        st.subheader("🔧 操作モード選択")
        mode = st.radio(
            "操作を選択してください：",
            ["🎨 スポイトツール（色取得）", "📍 範囲選択（塗りつぶし範囲指定）"],
            index=1 if not st.session_state.eyedropper_mode else 0,
            horizontal=True
        )
        
        # モード変更時の処理
        new_eyedropper_mode = (mode == "🎨 スポイトツール（色取得）")
        if new_eyedropper_mode != st.session_state.eyedropper_mode:
            st.session_state.eyedropper_mode = new_eyedropper_mode
            st.session_state.manual_coords = []  # モード変更時は座標をリセット
            st.rerun()
        
        # 現在のモードに応じた説明表示
        if st.session_state.eyedropper_mode:
            st.info("🎨 色を取得したい場所をクリック")
        else:
            st.info("📍 左上→右下の順で塗りつぶし範囲をクリック")
        
        # 塗りつぶし色選択
        # スポイトで取得した色があれば使用、なければデフォルト
        default_color = st.session_state.get('selected_color', "#FFFFFF")
        fill_color = st.color_picker("塗りつぶし色", value=default_color)
        
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
        
        # 操作状況の表示
        if st.session_state.eyedropper_mode:
            st.write("🎨 色を取得する場所をクリック")
        elif len(st.session_state.manual_coords) == 0:
            st.write("📍 1点目（左上）をクリック")
        elif len(st.session_state.manual_coords) == 1:
            st.write("📍 2点目（右下）をクリック")
        
        # 画像をクリック可能な形で表示
        try:
            coordinates = streamlit_image_coordinates(
                current_preview,
                key=f"image_coords_fill_{len(st.session_state.fill_areas)}"
            )
        except Exception as e:
            st.error(f"画像の表示中にエラーが発生しました: {str(e)}")
            coordinates = None
        
        # 座標クリックイベントの処理（モード別に完全分離）
        if coordinates:
            if st.session_state.eyedropper_mode:
                # 🎨 スポイトモード：色取得のみ（座標は一切保存しない）
                x, y = coordinates['x'], coordinates['y']
                if 0 <= x < current_preview.width and 0 <= y < current_preview.height:
                    # プレビュー画像から色を取得
                    pixel_color = current_preview.getpixel((x, y))
                    if len(pixel_color) == 3:  # RGB
                        r, g, b = pixel_color
                        hex_color = f"#{r:02x}{g:02x}{b:02x}"
                        # セッション状態を更新してcolor_pickerに反映
                        st.session_state.selected_color = hex_color
                        st.success(f"色を取得: {hex_color}")
                        st.rerun()
            else:
                # 📍 範囲選択モード：座標のみ保存（色取得は行わない）
                if len(st.session_state.manual_coords) < 2:
                    st.session_state.manual_coords.append((coordinates['x'], coordinates['y']))
                    if len(st.session_state.manual_coords) == 1:
                        st.success("1点目を選択")
                    else:
                        st.success("2点目を選択")
        
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
                # 塗りつぶしを適用してから図面領域で切り抜き
                filled_image = apply_fill_areas(st.session_state.original_image, st.session_state.fill_areas)
                current_filled_image = filled_image.crop(st.session_state.confirmed_drawing_area)
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
                st.image(final_preview, caption="処理済み図面（テンプレートに合成される部分）", use_container_width=True)
                
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
    
    ### ✨ 新機能 v1.7.4 - コンテンツ分析AI
    - 🤖 **OCR搭載**: 図面下部の文字を直接読み取り、内容を理解
    - 🔍 **キーワード検出**: 「株式会社」「電話番号」「免許番号」など帯特有の情報を自動で発見
    - 🎯 **情報クラスタリング**: 関連情報が密集する領域を「帯」として特定
    - 📏 **ハイブリッド分析**: テキスト情報と、それを区切る水平線を組み合わせ、帯の上端を正確に決定
    - 🧠 **3段階検出**: 学習→コンテンツ分析→レイアウト分析の順で高精度に検出
    
    ### 🔧 技術的改善
    - **EasyOCR**: 高精度な日本語OCRライブラリを導入
    - **正規表現**: 電話番号などの複雑なパターンを確実に抽出
    - **フォールバック**: コンテンツが見つからない場合も、レイアウト分析で対応
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
        
        # 学習データの統計情報
        if LEARNING_DATA_FILE.exists():
            try:
                with open(LEARNING_DATA_FILE, 'r') as f:
                    data = json.load(f)
                st.write("### 📊 学習データ統計")
                st.write(f"総レコード数: {data['metadata']['total_records']}")
                st.write(f"手動修正データ: {data['metadata']['manual_corrections']}")
                st.write(f"自動検出データ: {data['metadata']['auto_detections']}")
                st.write(f"最終更新: {data['metadata']['last_updated']}")
                st.write(f"データバージョン: {data.get('version', '1.0')}")
            except Exception as e:
                st.write(f"学習データ読み込みエラー: {str(e)}")
        
        # 予測履歴
        if st.session_state.recent_predictions:
            st.write("### 🔄 予測履歴（直近10件）")
            # `list()`でdequeのコピーを作成してから逆順にする
            for i, pred in enumerate(reversed(list(st.session_state.recent_predictions)[-5:])):  # 最新5件のみ表示
                method = pred.get('method', 'unknown')
                method_emoji = {
                    'learning': '🧠',
                    'content_analysis': '🤖',
                    'layout_analysis_fallback': '📊',
                    'fallback': '🔄'
                }.get(method, '❓')
                st.write(f"予測 {i+1}: {method_emoji} {method} - 信頼度 {pred.get('confidence', 0):.3f}")
        
        # 検出方法の詳細情報
        if st.session_state.recent_predictions:
            latest_prediction = list(st.session_state.recent_predictions)[-1]
            method = latest_prediction.get('method', 'unknown')
            
            st.write("### 🔍 最新検出詳細")
            if method == 'learning':
                st.info("🧠 学習データベースの予測を使用")
            elif method == 'content_analysis':
                st.success("🤖 コンテンツ分析AIによる検出を使用")
            elif method == 'layout_analysis_fallback':
                st.warning("📊 レイアウト分析（フォールバック）を使用")
            elif method == 'fallback':
                st.warning("🔄 最終フォールバックを使用")
            
            # 特徴ベース検出の詳細情報を表示
            if method == 'feature_based' and st.session_state.original_image:
                st.write("#### 📊 特徴分析結果")
                try:
                    # 文字領域と水平線を再検出して表示
                    text_regions = detect_text_regions(st.session_state.original_image)
                    horizontal_lines = detect_horizontal_lines(st.session_state.original_image)
                    
                    st.write(f"検出された文字領域数: {len(text_regions)}")
                    st.write(f"検出された水平線数: {len(horizontal_lines)}")
                    
                    if text_regions:
                        st.write("文字密度の高い領域を検出")
                    if horizontal_lines:
                        st.write("区切り線を検出")
                        
                except Exception as e:
                    st.write(f"特徴分析エラー: {str(e)}")
    
    # 学習データ管理セクション
    st.header("🗄️ 学習データ管理")
    
    # 現在の学習件数を表示
    if LEARNING_DATA_FILE.exists():
        try:
            with open(LEARNING_DATA_FILE, 'r') as f:
                data = json.load(f)
            metadata = data.get('metadata', {})
            total = metadata.get('total_records', 0)
            manual = metadata.get('manual_corrections', 0)
            auto = metadata.get('auto_detections', 0)
            st.info(f"**総学習数: {total}件**\n- 手動修正: {manual}件\n- 自動検出: {auto}件")
        except (json.JSONDecodeError, KeyError):
            # ファイルが空、または破損している場合
            st.warning("学習データの件数を読み込めませんでした。")

    # 学習データ管理の説明
    st.markdown("""
    ### 📋 学習データについて
    - **保存件数**: 理論上無制限（推奨: 1,000件程度）
    - **永続性**: ブラウザを閉じても保持されます
    - **自動反映**: 手動調整時にリアルタイムで学習
    - **優先度**: 手動修正データは高優先度で管理
    """)
    
    # 学習データ管理ボタン
    col_mgmt1, col_mgmt2 = st.columns(2)
    
    with col_mgmt1:
        if st.button("📤 学習データをエクスポート"):
            export_data = export_learning_data()
            if export_data:
                st.download_button(
                    "💾 JSONファイルをダウンロード",
                    data=export_data,
                    file_name=f"learning_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    with col_mgmt2:
        if st.button("🗑️ 学習データをクリア", type="secondary"):
            if clear_learning_data():
                st.rerun()
    
    # 学習データインポート
    st.subheader("📥 学習データをインポート")
    uploaded_learning_data = st.file_uploader(
        "学習データJSONファイルをアップロード",
        type=["json"],
        help="エクスポートした学習データをインポートできます"
    )
    
    if uploaded_learning_data:
        try:
            import_data = uploaded_learning_data.read().decode('utf-8')
            if st.button("✅ インポート実行"):
                if import_learning_data(import_data):
                    # scipyのインポートを促す
                    try:
                        import scipy
                    except ImportError:
                        st.warning("レイアウト分析機能にはSciPyが必要です。`pip install scipy`を実行してください。")
                    
                    # easyocrのインポートを促す
                    try:
                        import easyocr
                    except ImportError:
                        st.warning("コンテンツ分析機能にはEasyOCRが必要です。`pip install easyocr`を実行してください。")

                    st.rerun()
        except Exception as e:
            st.error(f"ファイルの読み込みに失敗しました: {str(e)}")
    
    # 学習データ件数制限設定
    st.subheader("⚙️ 学習データ設定")
    max_records = st.slider(
        "最大保存件数",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="この件数を超えると古い自動検出データが削除されます（手動修正データは優先保持）"
    )
    
    if st.button("🔧 件数制限を適用"):
        if manage_learning_data(max_records):
            st.success(f"✅ 学習データを{max_records}件に制限しました")
            st.rerun()
        else:
            st.info("ℹ️ 現在の件数は制限内です")
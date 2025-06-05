# 1. 自動帯検出
selected_area = auto_detect_drawing_area(img)
auto_x, auto_y, auto_x2, auto_y2 = selected_area
auto_w, auto_h = auto_x2 - auto_x, auto_y2 - auto_y

manual_mode = st.checkbox("手動で範囲を指定する（自動認識がおかしい場合）")
if manual_mode:
    st.write("※ 推奨値（自動検出結果）がデフォルトで入ってます。微調整してOKを押して下さい。")
    x = st.number_input("X座標", min_value=0, max_value=img.width-1, value=auto_x, step=1)
    y = st.number_input("Y座標", min_value=0, max_value=img.height-1, value=auto_y, step=1)
    w = st.number_input("幅", min_value=1, max_value=img.width-x, value=auto_w, step=1)
    h = st.number_input("高さ", min_value=1, max_value=img.height-y, value=auto_h, step=1)
    manual_crop = img.crop((x, y, x + w, y + h))
    cropped = manual_crop
    st.image(cropped, caption="手動選択範囲プレビュー", use_column_width=True)
    st.success("この範囲でPDF生成可能！")
else:
    cropped = img.crop(selected_area)
    st.image(cropped, caption="自動認識範囲プレビュー", use_column_width=True)
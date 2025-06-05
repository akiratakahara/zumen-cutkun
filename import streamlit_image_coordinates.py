import streamlit_image_coordinates

# grid_imgはPillow画像なのでNumPy配列に変換
coords = streamlit_image_coordinates.image_coordinates(np.array(grid_img), key="manual_select")
import streamlit as st
import pickle
import numpy as np

# Tiêu đề trang
st.title("Dự đoán Mức độ Nghiện Internet")

# Tải mô hình đã huấn luyện
with open("GDmodel_enc.rpk", "rb") as f:
    model = pickle.load(f) 
# Giao diện nhập liệu
st.header("Nhập thông tin người dùng:")

age = st.number_input("Tuổi", min_value=10, max_value=80, value=20)
study_time = st.number_input("Thời gian học mỗi ngày (giờ)", min_value=0.0, max_value=12.0, value=2.0)
online_time = st.number_input("Thời gian sử dụng Internet mỗi ngày (giờ)", min_value=0.0, max_value=24.0, value=5.0)
social_media = st.number_input("Thời gian dùng mạng xã hội mỗi ngày (giờ)", min_value=0.0, max_value=24.0, value=3.0)
sleep_time = st.number_input("Thời gian ngủ (giờ/ngày)", min_value=0.0, max_value=24.0, value=7.0)

# Nút dự đoán
if st.button("Dự đoán"):
    input_data = np.array([[age, study_time, online_time, social_media, sleep_time]])
    result = model.predict(input_data)
    st.success(f"Điểm dự đoán nghiện Internet: {result[0]:.2f}")

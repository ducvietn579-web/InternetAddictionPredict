import streamlit as st
import xgboost as xgb
import numpy as np
import pandas as pd
import joblib

# --- Load mô hình Machine Learning ---
try:
    rf_model, encoder = joblib.load("rfmodel_enc.rpk")
except Exception as e:
    st.error(f"Không thể load mô hình: {e}")
    rf_model, encoder = None, None

# --- Mapping cho dữ liệu dạng chữ ---
gender_map = {"Male": 0, "Female": 1}
academic_map = {"Undergraduate": 0, "Graduated": 1, "Highschool": 2}
relationship_map = {"Single": 0, "In a relationship": 1, "Complicated": 2}
platform_map = {"Youtube": 0, "Facebook": 1, "TikTok": 2, "Instagram": 3, "Other": 4}

# --- Giao diện chính ---
def main():
    st.title("🧠 Internet Addiction Prediction")
    st.write("Nhập thông tin bên dưới để dự đoán mức độ nghiện Internet:")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Giới tính", list(gender_map.keys()))
            academic = st.selectbox("Trình độ học vấn", list(academic_map.keys()))
            relationship = st.selectbox("Tình trạng mối quan hệ", list(relationship_map.keys()))
            platform = st.selectbox("Nền tảng sử dụng nhiều nhất", list(platform_map.keys()))
        with col2:
            sleep_hours = st.number_input("Số giờ ngủ mỗi đêm", min_value=0.0, max_value=12.0, value=7.0)
            mental_health = st.slider("Điểm sức khỏe tâm lý (1-10)", 1, 10, 5)
            usage_hours = st.number_input("Số giờ sử dụng Internet mỗi ngày", min_value=0.0, max_value=24.0, value=5.0)

        submit = st.form_submit_button("🔍 Dự đoán")

    if submit:
        if rf_model is None:
            st.error("Không thể dự đoán vì mô hình chưa được load đúng cách.")
        else:
            try:
                data = {
                    "Gender": gender_map[gender],
                    "Academic_Level": academic_map[academic],
                    "Sleep_Hours_Per_Night": sleep_hours,
                    "Relationship_Status": relationship_map[relationship],
                    "Mental_Health_Score": mental_health,
                    "Most_Used_Platform": platform_map[platform],
                    "Avg_Daily_Usage_Hours": usage_hours
                }

                X = pd.DataFrame([data])

                # Không cần encoder nữa
                prediction = rf_model.predict(X)[0]

                if prediction < 4:
                    level = "Thấp"
                elif prediction < 7:
                    level = "Trung bình"
                else:
                    level = "Cao"

                st.success(f"**Điểm dự đoán:** {round(prediction, 2)}")
                st.info(f"**Mức độ nghiện Internet:** {level}")

            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")
                st.write("📦 Dữ liệu đầu vào:", X)

if __name__ == '__main__':
    main()

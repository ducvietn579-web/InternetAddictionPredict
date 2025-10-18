import streamlit as st
import pandas as pd
import joblib

try:
    loaded = joblib.load("rfmodel_enc.rpk")
    st.write("📦 Kiểu dữ liệu load được:", type(loaded))

    # Nếu là tuple -> giải nén thành 2 phần
    if isinstance(loaded, tuple) and len(loaded) == 2:
        rf_model, encoder = loaded
        st.success("✅ Đã load thành công tuple (rf_model, encoder)")
        st.write("Kiểu model:", type(rf_model))
        st.write("Kiểu encoder:", type(encoder))
    else:
        rf_model, encoder = None, None
        st.error("⚠️ File không phải tuple (rf_model, encoder)")
except Exception as e:
    st.error(f"Không thể load mô hình: {e}")
    rf_model, encoder = None, None

def main():
    st.title("🧠 Internet Addiction Prediction")
    st.write("Nhập thông tin bên dưới để dự đoán mức độ nghiện Internet:")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Giới tính", ["Male", "Female"])
            academic = st.selectbox("Trình độ học vấn", ["Graduate", "Undergraduate", "High School"])
            relationship = st.selectbox("Tình trạng mối quan hệ", ["Single", "In a relationship", "Complicated"])
            platform = st.selectbox(
                                    "Nền tảng sử dụng nhiều nhất",["Instagram",
                                                                   "LINE",
                                                                   "VK",
                                                                   "Facebook",
                                                                   "LinkedIn",
                                                                   "WhatsApp",
                                                                   "TikTok",
                                                                   "KakaoTalk",
                                                                   "WeChat",
                                                                   "Twitter",
                                                                   "YouTube",
                                                                   "Snapchat"
                                                                ]
                                                            )
            sleep_hours = st.number_input("Số giờ ngủ mỗi đêm", min_value=0.0, max_value=12.0, value=7.0)
            mental_health = st.slider("Điểm sức khỏe tâm lý (1-10)", 1, 10, 5)
            usage_hours = st.number_input("Số giờ sử dụng Internet mỗi ngày", min_value=0.0, max_value=24.0, value=5.0)

            submit = st.form_submit_button("🔍 Dự đoán")

   if submit:
    if rf_model is None or encoder is None:
        st.error("Không thể dự đoán vì mô hình hoặc encoder chưa được load đúng cách.")
    else:
        try:
            data = pd.DataFrame([{
                "Gender": gender,
                "Academic_Level": academic,
                "Sleep_Hours_Per_Night": sleep_hours,
                "Relationship_Status": relationship,
                "Mental_Health_Score": mental_health,
                "Most_Used_Platform": platform,
                "Avg_Daily_Usage_Hours": usage_hours
            }])

            st.write("🧾 Dữ liệu gốc:", data)

            # Dùng encoder đã lưu để transform dữ liệu chữ
            X = encoder.transform(data)
            st.write("📊 Dữ liệu sau encoder:", X)

            # Dự đoán
            prediction = rf_model.predict(X)[0]

            # Xếp loại
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

if __name__ == '__main__':
    main()

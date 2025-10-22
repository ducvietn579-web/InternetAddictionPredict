import streamlit as st
import pandas as pd
import joblib

# --- Load mô hình ---
try:
    loaded = joblib.load("rfmodel (4).pkl")
    if isinstance(loaded, tuple) and len(loaded) == 2:
        rf_model, encoder = loaded
        st.success("Đã load thành công mô hình Random Forest và encoder")
    else:    
        st.error(" File không phải tuple (rf_model, encoder)")
        rf_model, encoder = None, None
except Exception as e:
    st.error(f"Không thể load mô hình: {e}")
    rf_model, encoder = None, None


def main():
    st.title("Internet Addiction Prediction")
    st.write("Nhập thông tin bên dưới để dự đoán mức độ nghiện Internet:")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Giới tính", ["Male", "Female"])
            academic = st.selectbox("Trình độ học vấn", ["Graduate", "Undergraduate", "Highschool"])
            relationship = st.selectbox("Tình trạng mối quan hệ", ["Single", "In a Relationship"])
            platform = st.selectbox("Nền tảng sử dụng nhiều nhất",
                                    ["Facebook", "Instagram", "TikTok", "YouTube", "Twitter", "Snapchat", "LinkedIn"])
        with col2:
            sleep_hours = st.number_input("Số giờ ngủ mỗi đêm", 0.0, 12.0, 7.0)
            mental_health = st.slider("Điểm sức khỏe tâm lý (1–10)", 1, 10, 5)
            usage_hours = st.number_input("Số giờ sử dụng Internet mỗi ngày", 0.0, 24.0, 5.0)
            conflict_over_internet = st.number_input("Xung đột khi sử dụng Internet", 0.0, 3.0, 0.0)
        submit = st.form_submit_button("🔍 Dự đoán")

    if submit:
        if rf_model is None or encoder is None:
            st.error("Không thể dự đoán vì mô hình chưa được load đúng cách.")
            return

        try:
            data = pd.DataFrame([{
                "Gender": gender,
                "Academic_Level": academic,
                "Sleep_Hours_Per_Night": sleep_hours,
                "Relationship_Status": relationship,
                "Mental_Health_Score": mental_health,
                "Most_Used_Platform": platform,
                "Avg_Daily_Usage_Hours": usage_hours,
                "Conflicts_Over_Social_Media": conflict_over_internet
            }])
            X = encoder.transform(data)
            prediction = rf_model.predict(X)[0]    
            level = "Thấp" if prediction < 4 else ("Trung bình" if prediction < 7 else "Cao")

            st.success(f"**Điểm dự đoán:** {round(prediction, 6)}")
            st.info(f"**Mức độ nghiện Internet:** {level}")

        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {e}")


if __name__ == "__main__":
    main()
